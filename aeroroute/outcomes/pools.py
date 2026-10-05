import array
import hashlib
import json
import sqlite3
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..storage.artifacts import publish, staging, verify_derived
from ..storage.identity import canonical_bytes, digest, load_json, write_json
from ..storage.lookup import readonly_database
from ..catalog.services import public_service_id


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
    where = ("split='historical' AND schedule_issue IS NULL AND evidence_issue IS NULL AND origin=? "
             "AND destination=? AND year IN (" + ",".join("?" for _ in years) + ")")
    args = [target["origin"], target["destination"], *years]
    for key in level:
        if target[key] is None:
            return None, None
        column = f"CAST(minute / {policy.bucket_hours * 60} AS INTEGER)" if key == "bucket" else key
        where += f" AND {column}=?"
        args.append(target[key])
    return where, args


def policy_from_identity(value):
    return MatchingPolicy(levels=tuple(tuple(level) for level in value["levels"]), bucket_hours=value["bucket_hours"],
                          min_support=value["min_support"], validation_reference=value["validation_reference"],
                          version=value["version"])


def choose_pool(catalog, target, years, policy, catalog_name, inventory_version, policy_identity):
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
        pool_id = digest({"catalog": catalog_name, "inventory": inventory_version, "policy": policy_identity,
                          "membership": "omit-unresolved-services-v1", "level": index, "key": key})
        return {"pool_id": pool_id, "level": f"level_{index}:" + ("_".join(level) or "route_only"),
                "criteria": list(level), "size": size, "low_support": size < policy.min_support,
                "where": where, "args": args}
    return None


class MembershipCache:
    def __init__(self, budget):
        if type(budget) is not int or budget < 0:
            raise ValueError("Invalid membership cache budget")
        self.budget, self.items, self.size = budget, OrderedDict(), 0

    def get(self, pool_id):
        item = self.items.get(pool_id)
        if item is None:
            return None
        self.items.move_to_end(pool_id)
        return item

    def put(self, pool_id, keys, sha256):
        weight = len(keys) * keys.itemsize
        if not weight or weight > self.budget:
            return
        while self.items and self.size + weight > self.budget:
            self.size -= self.items.popitem(last=False)[1]["weight"]
        self.items[pool_id] = {"weight": weight, "keys": keys, "sha256": sha256}
        self.size += weight


def prepare_pools(catalog, schedule, output, policy, fitting_years):
    years = tuple(sorted(set(fitting_years)))
    if not catalog.manifest["identity"]["policy"].get("review_reference"):
        raise ValueError("Catalog is not schedule ready for historical; review diagnostics")
    historical = catalog.store.inventory("historical")["sources"]
    if (not years or any(type(year) is not int for year in years)
            or not set(years) <= {source["year"] for source in historical}):
        raise ValueError("Fitting years must have explicitly historical source coverage")
    sources = [source for source in historical if source["year"] in years]
    inventory = {"dataset_version": digest(sources), "sources": sources}
    identity = {"kind": "pools", "schema_version": 1, "builder_version": 2, "expansion": "lazy",
                "membership": "omit-unresolved-services-v1", "catalog": catalog.path.name,
                "schedule_sha256": schedule.reference.sha256, "fitting_inventory": inventory["dataset_version"],
                "fitting_years": list(years), "policy": json.loads(canonical_bytes(asdict(policy))),
                "membership_order": "service_instance_id"}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    divisor = policy.bucket_hours * 60
    year_filter = set(years)
    print("Loading scheduled flights", flush=True)
    pending = {bytes.fromhex(row[0]): None for row in schedule.database.execute("SELECT service_id FROM flights")}
    fine = Counter()
    print("Counting historical services once", flush=True)
    scanned = 0
    for row in catalog.database.execute(
            "SELECT service_id, split, year, origin, destination, carrier, quarter, weekday, minute, schedule_issue, evidence_issue FROM services"):
        scanned += 1
        minute = row["minute"]
        bucket = None if minute is None else minute // divisor
        service_id = row["service_id"]
        if isinstance(service_id, str):
            service_id = bytes.fromhex(service_id)
        if service_id in pending:
            pending[service_id] = (row["origin"], row["destination"], row["carrier"], row["quarter"], row["weekday"], bucket)
        if (row["split"] == "historical" and row["schedule_issue"] is None and row["evidence_issue"] is None
                and row["year"] in year_filter):
            fine[(row["origin"], row["destination"], row["carrier"], row["quarter"], row["weekday"], bucket)] += 1
        if scanned % 2000000 == 0:
            print(f"Catalog rows read: {scanned}", flush=True)
    print(f"Catalog rows read: {scanned}; historical keys: {len(fine)}", flush=True)
    if any(value is None for value in pending.values()):
        raise ValueError("Scheduled flight is absent from the service catalog")
    aggregates = {level: Counter() for level in policy.levels}
    names = ("origin", "destination", "carrier", "quarter", "weekday", "bucket")
    for key, size in fine.items():
        target = dict(zip(names, key))
        for level in policy.levels:
            if any(target[name] is None for name in level):
                continue
            aggregates[level][(target["origin"], target["destination"], *(target[name] for name in level))] += size
    levels, counters = Counter(), Counter()
    count = 0
    with sqlite3.connect(stage / "pools.sqlite") as database:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA cache_size=-8192")
        database.execute("""
            CREATE TABLE definitions (pool_id TEXT PRIMARY KEY, level TEXT NOT NULL, criteria TEXT NOT NULL,
                size INTEGER NOT NULL, low_support INTEGER NOT NULL)
        """)
        seen = {}
        print(f"Assigning {len(pending)} scheduled flights", flush=True)
        for spec in pending.values():
            target = dict(zip(names, spec))
            choice = None
            for index, level in enumerate(policy.levels):
                if any(target[name] is None for name in level):
                    continue
                size = aggregates[level].get((target["origin"], target["destination"], *(target[name] for name in level)), 0)
                if size < policy.min_support and index != len(policy.levels) - 1:
                    continue
                if not size:
                    break
                key = {name: target[name] for name in ("origin", "destination", *level)}
                pool_id = digest({"catalog": catalog.path.name, "inventory": inventory["dataset_version"], "policy": identity["policy"],
                                  "membership": "omit-unresolved-services-v1", "level": index, "key": key})
                choice = {"pool_id": pool_id, "level": f"level_{index}:" + ("_".join(level) or "route_only"),
                          "criteria": list(level), "size": size, "low_support": size < policy.min_support}
                break
            count += 1
            if choice is None:
                counters["unsupported"] += 1
            else:
                if choice["pool_id"] not in seen:
                    seen[choice["pool_id"]] = choice["size"]
                    database.execute("INSERT INTO definitions VALUES (?, ?, ?, ?, ?)",
                                     (choice["pool_id"], choice["level"], json.dumps(choice["criteria"]), choice["size"], int(choice["low_support"])))
                criteria = choice["criteria"]
                levels[choice["level"]] += 1
                counters["carrier_fallback"] += "carrier" not in criteria
                counters["lost_time_of_day"] += "bucket" not in criteria
                counters["low_support"] += bool(choice["low_support"])
            counters["missing_time_bucket"] += target["bucket"] is None
        database.commit()
        sizes = Counter(seen.values())
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
    def __init__(self, catalog, schedule, path, membership_budget=32 * 1024 ** 2):
        self.catalog, self.schedule, self.path = catalog, schedule, Path(path)
        self.manifest = verify_derived(path, "pools")
        identity = self.manifest["identity"]
        if (identity["catalog"] != catalog.path.name or identity["schedule_sha256"] != schedule.reference.sha256
                or identity.get("membership") != "omit-unresolved-services-v1"):
            raise ValueError("Pool set references different catalog/schedule")
        if not catalog.manifest["identity"]["policy"].get("review_reference"):
            raise ValueError("Catalog is not schedule ready for historical; review diagnostics")
        self.inventory = load_json(self.path / "fitting_inventory.json")
        if (self.inventory["dataset_version"] != digest(self.inventory["sources"])
                or identity["fitting_inventory"] != self.inventory["dataset_version"]
                or any(source not in catalog.store.inventory("historical")["sources"] for source in self.inventory["sources"])):
            raise ValueError("Pool fitting inventory mismatch")
        self.report = load_json(self.path / "quality.json")
        self.database = readonly_database(self.path / "pools.sqlite")
        self.sidecar = None
        if identity.get("builder_version", 1) >= 2:
            if catalog.manifest["identity"]["builder_version"] < 3:
                raise ValueError("Lazy pools require a typed service projection")
            self._memory = MembershipCache(membership_budget)
            self.cache_path = self.path.parent / f"{self.path.name}.membership-cache.sqlite"
            self.sidecar = sqlite3.connect(self.cache_path)
            self.sidecar.row_factory = sqlite3.Row
            self.sidecar.execute("PRAGMA cache_size=-8192")
            self.sidecar.execute("CREATE TABLE IF NOT EXISTS cached_hashes (pool_id TEXT PRIMARY KEY, size INTEGER NOT NULL, sha256 TEXT NOT NULL)")
            self.sidecar.commit()
            return
        for pool in self.database.execute("SELECT * FROM pools"):
            rows = self.database.execute("SELECT * FROM members WHERE pool_id=? ORDER BY ordinal", (pool["pool_id"],))
            if membership_hash(rows) != pool["sha256"]:
                raise ValueError("Ordered pool membership hash mismatch")
            stats = self.database.execute("SELECT count(*), min(ordinal), max(ordinal) FROM members WHERE pool_id=?", (pool["pool_id"],)).fetchone()
            if tuple(stats) != (pool["size"], 0, pool["size"] - 1):
                raise ValueError("Pool ordinal index mismatch")

    def resolve(self, service_id):
        identity = self.manifest["identity"]
        service = self.catalog.get(service_id)
        policy = policy_from_identity(identity["policy"])
        choice = choose_pool(self.catalog, features(service, policy), tuple(identity["fitting_years"]), policy,
                             self.catalog.path.name, self.inventory["dataset_version"], identity["policy"])
        if choice is None:
            raise NoMatchingPool("No historical matching pool for scheduled service")
        published = self.database.execute("SELECT * FROM definitions WHERE pool_id=?", (choice["pool_id"],)).fetchone()
        if published is None or published["size"] != choice["size"] or published["level"] != choice["level"]:
            raise ValueError("Resolved pool differs from the frozen definition")
        keys, sha256 = self._members(choice)
        return {**choice, "sha256": sha256, "keys": keys}

    def member(self, service_pk):
        row = self.catalog.database.execute(
            "SELECT service_id, archive_id, evidence_ordinal FROM services WHERE service_pk=?", (service_pk,)).fetchone()
        if row is None or row["evidence_ordinal"] is None:
            raise ValueError("Pool member has no authoritative evidence source")
        archive = self.catalog.store.archive(row["archive_id"])
        return {"service_pk": service_pk, "service_id": public_service_id(row["service_id"]),
                "record_id": digest([archive["source_sha256"], archive["member"], row["evidence_ordinal"]]),
                "source_sha256": archive["source_sha256"], "source_member": archive["member"],
                "source_ordinal": row["evidence_ordinal"]}

    def _members(self, choice):
        cached = self._memory.get(choice["pool_id"])
        if cached is not None:
            return cached["keys"], cached["sha256"]
        rows = []
        for member in self.catalog.database.execute(
                "SELECT service_pk, service_id, archive_id, evidence_ordinal FROM services WHERE " + choice["where"] + " ORDER BY service_id",
                choice["args"]):
            if member["evidence_ordinal"] is None:
                raise ValueError("Pool member has no authoritative evidence source")
            if type(member["service_pk"]) is not int or member["service_pk"] >= 2 ** 31:
                raise ValueError("Service key exceeds the validated int32 range")
            archive = self.catalog.store.archive(member["archive_id"])
            rows.append({"service_pk": member["service_pk"], "source_sha256": archive["source_sha256"],
                         "source_member": archive["member"], "source_ordinal": member["evidence_ordinal"]})
        if len(rows) != choice["size"]:
            raise ValueError("Pool membership count mismatch")
        sha256 = membership_hash(rows)
        stored = self.sidecar.execute("SELECT size, sha256 FROM cached_hashes WHERE pool_id=?", (choice["pool_id"],)).fetchone()
        if stored is not None and (stored["sha256"] != sha256 or stored["size"] != len(rows)):
            raise ValueError("Cached pool membership hash mismatch")
        if stored is None:
            self.sidecar.execute("INSERT INTO cached_hashes VALUES (?, ?, ?)", (choice["pool_id"], len(rows), sha256))
            self.sidecar.commit()
        keys = array.array("i", (row["service_pk"] for row in rows))
        self._memory.put(choice["pool_id"], keys, sha256)
        return keys, sha256

    def _metric_report(self, counts, levels, unique):
        total = counts["requests"]
        def metric(value):
            return {"count": value, "fraction": value / total if total else 0}
        return {"denominator": "explicit pre-2025 validation requests", "request_count": total,
                "unique_service_count": len(unique), "pool_set": self.path.name,
                "fallback_levels": {key: metric(value) for key, value in sorted(levels.items())},
                **{key: metric(counts[key]) for key in ("unsupported", "carrier_fallback", "lost_time_of_day", "low_support")}}

    def validation_report(self, service_ids):
        counts, levels, unique = Counter(), Counter(), set()
        if self.manifest["identity"].get("builder_version", 1) >= 2:
            identity = self.manifest["identity"]
            policy = policy_from_identity(identity["policy"])
            for service_id in service_ids:
                service = self.schedule.catalog.get(service_id)
                if service["year"] >= 2025 or service["year"] in identity["fitting_years"]:
                    raise ValueError("Validation requests must be pre-2025 and outside fitting years")
                if self.schedule.database.execute("SELECT 1 FROM flights WHERE service_id=?", (service_id,)).fetchone() is None:
                    raise ValueError("Validation service not in prepared target scope")
                counts["requests"] += 1
                unique.add(service_id)
                choice = choose_pool(self.catalog, features(service, policy), tuple(identity["fitting_years"]), policy,
                                     self.catalog.path.name, self.inventory["dataset_version"], identity["policy"])
                if choice is None:
                    counts["unsupported"] += 1
                    continue
                criteria = choice["criteria"]
                levels[choice["level"]] += 1
                counts["carrier_fallback"] += "carrier" not in criteria
                counts["lost_time_of_day"] += "bucket" not in criteria
                counts["low_support"] += bool(choice["low_support"])
            return self._metric_report(counts, levels, unique)
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
        return self._metric_report(counts, levels, unique)

    def close(self):
        if self.sidecar is not None:
            self.sidecar.close()
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
