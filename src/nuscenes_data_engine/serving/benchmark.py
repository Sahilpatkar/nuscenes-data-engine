"""Latency sanity check for the serving API.

Times N sequential ``/predict`` calls (after one excluded warm-up) and prints p50/p95:

    uv run python -m nuscenes_data_engine.serving.benchmark --image app/samples/x.jpg -n 30
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import httpx


def run_benchmark(url: str, image: Path, n: int = 30) -> dict[str, float]:
    """POST ``image`` to ``{url}/predict`` ``n`` times sequentially; return latency ms."""
    data = image.read_bytes()
    with httpx.Client(timeout=120) as client:

        def call() -> float:
            start = time.perf_counter()
            resp = client.post(f"{url}/predict", files={"file": (image.name, data, "image/jpeg")})
            resp.raise_for_status()
            return (time.perf_counter() - start) * 1000

        call()  # warm-up (model + torch graph init) — excluded
        times = [call() for _ in range(n)]

    quantiles = statistics.quantiles(times, n=20)  # 5% steps: [9] = p50 area, [18] = p95
    return {
        "n": float(n),
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": quantiles[18],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Serving API base URL.")
    parser.add_argument("--image", type=Path, required=True, help="Image to POST repeatedly.")
    parser.add_argument("-n", type=int, default=30, help="Number of timed requests.")
    args = parser.parse_args()

    stats = run_benchmark(args.url, args.image, n=args.n)
    print(
        f"{int(stats['n'])} requests to {args.url}/predict — "
        f"p50 {stats['p50_ms']:.0f} ms, p95 {stats['p95_ms']:.0f} ms, "
        f"mean {stats['mean_ms']:.0f} ms"
    )


if __name__ == "__main__":
    main()
