import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..storage.artifacts import publish, staging, verify_derived
from ..storage.identity import canonical_bytes, digest, load_json, write_json
from ..storage.lookup import readonly_database


DEFAULT_LEVELS = (("carrier", "quarter", "weekday", "bucket"), ("carrier", "quarter", "bucket"),
                  ("carrier", "quarter"), ("carrier",), ())


class NoMatchingPool(ValueError):
    pass


@dataclass(frozen=True)
class MatchingPolicy:
    levels: tuple[tuple[str, ...], ...] = DEFAULT_LEVELS
    bucket_hours: int = 6
    min_support: int = 30
    validation_reference: str | None = None
    version: str = "service-matching-v1"

    def __post_init__(self):
        if (type(self.bucket_hours) is not int or self.bucket_hours < 1 or 24 % self.bucket_hours
                or type(self.min_support) is not int or self.min_support < 1 or self.version != "service-matching-v1"
                or type(self.levels) is not tuple or not self.levels or self.levels[-1] != ()):
            raise ValueError("Invalid matching policy")
        previous = {"carrier", "quarter", "weekday", "bucket"}
        for level in self.levels:
            if type(level) is not tuple or len(set(level)) != len(level) or not set(level) <= previous:
                raise ValueError("Matching levels must progressively relax known criteria")
            previous = set(level)


def membership_hash(rows):
    value = hashlib.sha256(b"[")
    for index, row in enumerate(rows):
        if index:
            value.update(b",")
        value.update(canonical_bytes([row["source_sha256"], row["source_member"], row["source_ordinal"]]))
    value.update(b"]")
    return value.hexdigest()


def features(service, policy):
    return {"origin": service["origin"], "destination": service["destination"], "carrier": service["carrier"],
            "quarter": service["quarter"], "weekday": service["weekday"],
            "bucket": service["minute"] // (policy.bucket_hours * 60) if service["minute"] is not None else None}


def matching_query(target, level, years, policy):
    where = "split='historical' AND origin=? AND destination=? AND year IN (" + ",".join("?" for _ in years) + ")"
    args = [target["origin"], target["destination"], *years]
    for key in level:
        if target[key] is None:
            return None, None
        column = f"CAST(minute / {policy.bucket_hours * 60} AS INTEGER)" if key == "bucket" else key
        where += f" AND {column}=?"
        args.append(target[key])
    return where, args


def prepare_pools(catalog, schedule, output, policy, fitting_years):
    years = tuple(sorted(set(fitting_years)))
    catalog.require_ready("historical")
    historical = catalog.store.inventory("historical")["sources"]
    if (not years or any(type(year) is not int for year in years)
            or not set(years) <= {source["year"] for source in historical}):
        raise ValueError("Fitting years must have explicitly historical source coverage")
    sources = [source for source in historical if source["year"] in years]
    inventory = {"dataset_version": digest(sources), "sources": sources}
    identity = {"kind": "pools", "schema_version": 1, "builder_version": 1, "catalog": catalog.path.name,
                "schedule_sha256": schedule.reference.sha256, "fitting_inventory": inventory["dataset_version"],
                "fitting_years": list(years), "policy": json.loads(canonical_bytes(asdict(policy))),
                "membership_order": "service_instance_id"}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    levels, counters = Counter(), Counter()
    count = 0
    with sqlite3.connect(stage / "pools.sqlite") as database:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA cache_size=-8192")
        database.executescript("""
            CREATE TABLE pools (pool_id TEXT PRIMARY KEY, level TEXT, criteria TEXT, size INTEGER, sha256 TEXT, low_support INTEGER);
            CREATE TABLE members (pool_id TEXT, ordinal INTEGER, service_id TEXT, record_id TEXT,
                source_sha256 TEXT, source_member TEXT, source_ordinal INTEGER,
                PRIMARY KEY(pool_id, ordinal), UNIQUE(pool_id, service_id));
            CREATE TABLE targets (service_id TEXT PRIMARY KEY, request_key TEXT, pool_id TEXT);
            CREATE TABLE resolutions (request_key TEXT PRIMARY KEY, pool_id TEXT);
        """)
        for row in schedule.database.execute("SELECT service_id FROM flights ORDER BY service_id"):
            service = schedule.catalog.get(row[0])
            target = features(service, policy)
            request_key = digest(target)
            cached = database.execute("SELECT pool_id FROM resolutions WHERE request_key=?", (request_key,)).fetchone()
            if cached is not None:
                pool_id = cached[0]
            else:
                pool_id = None
                for index, level in enumerate(policy.levels):
                    where, args = matching_query(target, level, years, policy)
                    if where is None:
                        continue
                    size = catalog.database.execute("SELECT count(*) FROM services WHERE " + where, args).fetchone()[0]
                    if size < policy.min_support and index != len(policy.levels) - 1:
                        continue
                    if not size:
                        break
                    key = {name: target[name] for name in ("origin", "destination", *level)}
                    pool_id = digest({"catalog": catalog.path.name, "inventory": inventory["dataset_version"],
                                      "policy": identity["policy"], "level": index, "key": key})
                    if database.execute("SELECT 1 FROM pools WHERE pool_id=?", (pool_id,)).fetchone() is None:
                        for ordinal, member in enumerate(catalog.database.execute("SELECT service_id, evidence_source FROM services WHERE " + where + " ORDER BY service_id", args)):
                            ref = catalog.store.source_ref(member["evidence_source"], "historical")
                            database.execute("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (pool_id, ordinal, member["service_id"], member["evidence_source"],
                                 ref.source_sha256, ref.csv_member, ref.source_row_id))
                        sha256 = membership_hash(database.execute("SELECT * FROM members WHERE pool_id=? ORDER BY ordinal", (pool_id,)))
                        label = f"level_{index}:" + ("_".join(level) or "route_only")
                        database.execute("INSERT INTO pools VALUES (?, ?, ?, ?, ?, ?)",
                            (pool_id, label, json.dumps(level), size, sha256, size < policy.min_support))
                    break
                database.execute("INSERT INTO resolutions VALUES (?, ?)", (request_key, pool_id))
            database.execute("INSERT INTO targets VALUES (?, ?, ?)", (service["service_instance_id"], request_key, pool_id))
            count += 1
            if pool_id is None:
                counters["unsupported"] += 1
            else:
                pool = database.execute("SELECT * FROM pools WHERE pool_id=?", (pool_id,)).fetchone()
                criteria = json.loads(pool["criteria"])
                levels[pool["level"]] += 1
                counters["carrier_fallback"] += "carrier" not in criteria
                counters["lost_time_of_day"] += "bucket" not in criteria
                counters["low_support"] += bool(pool["low_support"])
            counters["missing_time_bucket"] += target["bucket"] is None
        database.commit()
        sizes = dict(database.execute("SELECT size, count(*) FROM pools GROUP BY size"))
    database.close()
    report = {"denominator": "unique scheduled target services", "request_count": count,
              "scope": "coverage diagnostics; not a validation-policy selection",
              "fallback_levels": {name: {"count": value, "fraction": value / count if count else 0} for name, value in sorted(levels.items())},
              "pool_size_distribution": {str(key): value for key, value in sorted(sizes.items())}}
    for key in ("unsupported", "carrier_fallback", "lost_time_of_day", "low_support", "missing_time_bucket"):
        report[key] = {"count": counters[key], "fraction": counters[key] / count if count else 0}
    write_json(stage / "quality.json", report)
    write_json(stage / "fitting_inventory.json", inventory)
    return publish(stage, final, identity)


class PreparedPools:
    def __init__(self, catalog, schedule, path):
        self.catalog, self.schedule, self.path = catalog, schedule, Path(path)
        self.manifest = verify_derived(path, "pools")
        identity = self.manifest["identity"]
        if identity["catalog"] != catalog.path.name or identity["schedule_sha256"] != schedule.reference.sha256:
            raise ValueError("Pool set references different catalog/schedule")
        catalog.require_ready("historical")
        self.inventory = load_json(self.path / "fitting_inventory.json")
        if (self.inventory["dataset_version"] != digest(self.inventory["sources"])
                or identity["fitting_inventory"] != self.inventory["dataset_version"]
                or any(source not in catalog.store.inventory("historical")["sources"] for source in self.inventory["sources"])):
            raise ValueError("Pool fitting inventory mismatch")
        self.report = load_json(self.path / "quality.json")
        self.database = readonly_database(self.path / "pools.sqlite")
        for pool in self.database.execute("SELECT * FROM pools"):
            rows = self.database.execute("SELECT * FROM members WHERE pool_id=? ORDER BY ordinal", (pool["pool_id"],))
            if membership_hash(rows) != pool["sha256"]:
                raise ValueError("Ordered pool membership hash mismatch")
            stats = self.database.execute("SELECT count(*), min(ordinal), max(ordinal) FROM members WHERE pool_id=?", (pool["pool_id"],)).fetchone()
            if tuple(stats) != (pool["size"], 0, pool["size"] - 1):
                raise ValueError("Pool ordinal index mismatch")

    def validation_report(self, service_ids):
        counts, levels, unique = Counter(), Counter(), set()
        for service_id in service_ids:
            service = self.schedule.catalog.get(service_id)
            if service["year"] >= 2025 or service["year"] in self.manifest["identity"]["fitting_years"]:
                raise ValueError("Validation requests must be pre-2025 and outside fitting years")
            target = self.database.execute("SELECT pool_id FROM targets WHERE service_id=?", (service_id,)).fetchone()
            if target is None:
                raise ValueError("Validation service not in prepared target scope")
            counts["requests"] += 1
            unique.add(service_id)
            if target[0] is None:
                counts["unsupported"] += 1
                continue
            pool = self.database.execute("SELECT * FROM pools WHERE pool_id=?", (target[0],)).fetchone()
            criteria = json.loads(pool["criteria"])
            levels[pool["level"]] += 1
            counts["carrier_fallback"] += "carrier" not in criteria
            counts["lost_time_of_day"] += "bucket" not in criteria
            counts["low_support"] += bool(pool["low_support"])
        total = counts["requests"]
        def metric(value):
            return {"count": value, "fraction": value / total if total else 0}
        return {"denominator": "explicit pre-2025 validation requests", "request_count": total,
                "unique_service_count": len(unique), "pool_set": self.path.name,
                "fallback_levels": {key: metric(value) for key, value in sorted(levels.items())},
                **{key: metric(counts[key]) for key in ("unsupported", "carrier_fallback", "lost_time_of_day", "low_support")}}

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
