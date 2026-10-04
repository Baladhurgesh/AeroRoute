import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..paths import RAW_DIR
from .archives import REQUIRED_COLUMNS, VALIDATION_VERSION, normalize_column, validate_archive


BASE_URL = "https://transtats.bts.gov/PREZIP/"
PREFIX = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018"
DEFAULT_OUTPUT = RAW_DIR


def archive_name(year, month):
    return f"{PREFIX}_{year}_{month}.zip"


def requested_months(start_year, end_year, month=None):
    if start_year < 2018 or end_year < start_year:
        raise ValueError("Years must start at 2018 or later and be in ascending order")
    if month is not None and not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    return [(year, selected) for year in range(start_year, end_year + 1)
            for selected in ([month] if month is not None else range(1, 13))]


def curl(*args):
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "20", "--max-time", "300",
        "--retry", "3", "--retry-connrefused", "--retry-max-time", "600",
        *map(str, args),
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True,
                              text=True, timeout=1000).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"BTS request failed: {(error.stderr or '')[-1000:].strip()}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("BTS request exceeded the total retry timeout") from error


def parse_listing(html):
    pattern = rf'(\d+)\s+<a href="[^\"]*/({re.escape(PREFIX)}_\d{{4}}_\d{{1,2}}\.zip)"'
    matches = re.findall(pattern, html, re.IGNORECASE)
    result = {name: int(size) for size, name in matches}
    if not result:
        raise ValueError("BTS directory did not contain the expected monthly archive listing")
    return result


def fetch_archive(url, path):
    headers = curl("--dump-header", "-", "--output", path, url)
    metadata = {}
    for line in headers.splitlines():
        if line.startswith("HTTP/"):
            metadata = {}
        name, separator, value = line.partition(":")
        if separator and name.lower() in {"etag", "last-modified"}:
            metadata[name.lower().replace("-", "_")] = value.strip()
    return metadata


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_manifest(path):
    if not path.exists():
        return {"version": 1, "dataset": PREFIX, "files": {}}
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or manifest.get("dataset") != PREFIX or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Unrecognized manifest format; existing manifest will not be overwritten")
    if any(not isinstance(entry, dict) for entry in manifest["files"].values()):
        raise ValueError("Invalid manifest entries")
    return manifest


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def check_disk_space(root, required_bytes, reserve=1024 ** 3):
    existing = root.resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing)[2]
    if free < required_bytes + reserve:
        raise OSError(f"Insufficient disk space: {free:,} bytes free; need {required_bytes + reserve:,} including reserve")
    return free


def process_month(year, month, root, expected_size, manifest, verify_only=False):
    key = f"{year}-{month:02d}"
    relative = Path(str(year)) / archive_name(year, month)
    path = root / relative
    url = BASE_URL + path.name
    entry = manifest["files"].get(key)
    if entry:
        expected = {"year": year, "month": month, "source_url": url,
                    "path": relative.as_posix(), "size_bytes": expected_size,
                    "status": "validated", "validation_version": VALIDATION_VERSION}
        if any(entry.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Conflicting manifest/source metadata for {key}; refusing to overwrite")
        if not isinstance(entry.get("row_count"), int) or entry["row_count"] <= 0:
            raise ValueError(f"Invalid row count in manifest for {key}")
    if verify_only and (not path.is_file() or not entry):
        raise FileNotFoundError(f"Missing validated archive or manifest entry for {key}")
    if path.exists():
        if path.stat().st_size != expected_size:
            raise ValueError(f"Existing archive size mismatch for {key}; refusing to overwrite")
        checksum = sha256_file(path)
        if entry and checksum != entry.get("sha256"):
            raise ValueError(f"Existing archive checksum mismatch for {key}; refusing to overwrite")
        if entry and not verify_only:
            return "skipped"
        validation = validate_archive(path, year, month)
        if verify_only:
            if any(entry.get(field) != value for field, value in validation.items()):
                raise ValueError(f"Archive validation differs from manifest for {key}")
            return "verified"
        metadata = {}
        action = "adopted"
    else:
        check_disk_space(root, expected_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".zip.part")
        metadata = fetch_archive(url, partial)
        if partial.stat().st_size != expected_size:
            raise ValueError(f"Downloaded archive size mismatch for {key}")
        validation = validate_archive(partial, year, month)
        checksum = sha256_file(partial)
        if entry and checksum != entry.get("sha256"):
            raise ValueError(f"Downloaded archive checksum differs from existing manifest for {key}")
        if path.exists():
            raise FileExistsError(f"Archive appeared during download: {path}; refusing to overwrite")
        partial.replace(path)
        action = "downloaded"
    manifest["files"][key] = {
        "year": year, "month": month, "path": relative.as_posix(),
        "source_url": url, "size_bytes": expected_size, "sha256": checksum,
        "source_etag": metadata.get("etag"),
        "source_last_modified": metadata.get("last_modified"),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "validated", "validation_version": VALIDATION_VERSION,
        **validation,
    }
    save_manifest(root / "manifest.json", manifest)
    return action
