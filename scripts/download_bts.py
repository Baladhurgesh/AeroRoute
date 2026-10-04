import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aeroroute.acquisition.bts import (
    BASE_URL, DEFAULT_OUTPUT, PREFIX, REQUIRED_COLUMNS, VALIDATION_VERSION,
    archive_name, check_disk_space, curl, fetch_archive, load_manifest,
    normalize_column, parse_listing, process_month, requested_months,
    save_manifest, sha256_file, validate_archive,
)
from aeroroute.cli.download import main, run


if __name__ == "__main__":
    sys.exit(run())
