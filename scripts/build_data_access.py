import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aeroroute.cli.build_data_access import main, periods, run
from aeroroute.reporting.access import run_pilot, verify_access


if __name__ == "__main__":
    sys.exit(run())
