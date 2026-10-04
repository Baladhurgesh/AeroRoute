import io
from functools import cache
from importlib import resources
from zoneinfo import ZoneInfo


@cache
def zone_bytes(key):
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("Invalid IANA timezone key")
    return resources.files("tzdata.zoneinfo").joinpath(*key.split("/")).read_bytes()


@cache
def pinned_zone(key):
    return ZoneInfo.from_file(io.BytesIO(zone_bytes(key)), key=key)
