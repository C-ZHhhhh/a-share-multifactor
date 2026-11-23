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
    bench, cov = reconstruct_benchmark(long, return_coverage=True)
    bench.to_frame().to_parquet(PROC / "bench_rec.parquet")
    cov.to_parquet(PROC / "bench_coverage.parquet")
    print(f"benchmark coverage (members with a previous-day cap): mean {cov.coverage.mean():.1%}, "
          f"min {cov.coverage.min():.1%}")
    res = audit(long)
    import json
    json.dump({k: v for k, v in res.items() if k != "missing_stocks"}, open(PROC / "audit.json", "w"),
              ensure_ascii=False, indent=2, default=str)
    for k, v in res.items():
        if k != "missing_stocks":
            print(f"{k:35s} {v}")
