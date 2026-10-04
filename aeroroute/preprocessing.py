from .acquisition.archives import read_source_rows
from .cli.normalize import main, run
from .normalization.datasets import publish_dataset, split_configuration
from .normalization.pipeline import (
    NORMALIZATION_VERSION, flush_batch, normalize_month, peak_rss_bytes,
    source_identity, transform_identity,
)
from .normalization.quality import collect_quality, fact_summary, new_quality
from .normalization.rows import mapping_zone, normalize_row, outcome_category
from .paths import ROOT


if __name__ == "__main__":
    raise SystemExit(run())
