"""Step 2: build clean daily panel and run the data-quality audit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.data_audit import audit
from mf.config import PROC
from mf.panel import build_and_save, reconstruct_benchmark

if __name__ == "__main__":
    long = build_and_save()
    print(f"panel: {len(long):,} rows, {long.code.nunique()} stocks")
    reconstruct_benchmark(long).to_frame().to_parquet(PROC / "bench_rec.parquet")
    res = audit(long)
    for k, v in res.items():
        if k != "missing_stocks":
            print(f"{k:35s} {v}")
