"""Reproducible local vector-search benchmark for RustyRAG backends.

Measures a 384-dimensional corpus representative of the default all-minilm
embedding model. Results compare quantized turbovec search with exact float32
NumPy search. An optional embedded Qdrant run adds the current Qdrant client
implementation without requiring Docker.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from turbovec import IdMapIndex


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    return np.take_along_axis(candidates, order, axis=1)


def _percentiles(values: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(values, dtype=np.float64) * 1_000
    return {
        "mean_ms": round(float(milliseconds.mean()), 4),
        "p50_ms": round(float(np.percentile(milliseconds, 50)), 4),
        "p95_ms": round(float(np.percentile(milliseconds, 95)), 4),
        "qps": round(float(1_000 / milliseconds.mean()), 2),
    }


def _time_exact(vectors: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, float]]:
    timings: list[float] = []
    rankings: list[np.ndarray] = []
    for query in queries:
        started = time.perf_counter()
        ranking = _top_k((vectors @ query).reshape(1, -1), k)[0]
        timings.append(time.perf_counter() - started)
        rankings.append(ranking)
    return np.asarray(rankings), _percentiles(timings)


def benchmark_turbovec(
    vectors: np.ndarray,
    queries: np.ndarray,
    exact_rankings: np.ndarray,
    bit_width: int,
    k: int,
) -> dict[str, float]:
    ids = np.arange(len(vectors), dtype=np.uint64)
    with tempfile.TemporaryDirectory(prefix="rusty-rag-turbovec-") as temporary_directory:
        index_path = Path(temporary_directory) / "vectors.tvim"
        started = time.perf_counter()
        index = IdMapIndex(dim=vectors.shape[1], bit_width=bit_width)
        index.add_with_ids(vectors, ids)
        index.write(str(index_path))
        build_seconds = time.perf_counter() - started

        # Ensure the first-query initialization cost is not part of steady-state latency.
        index.search(queries[:1], k=k)
        timings: list[float] = []
        returned_rankings: list[np.ndarray] = []
        for query in queries:
            started = time.perf_counter()
            _, result_ids = index.search(query.reshape(1, -1), k=k)
            timings.append(time.perf_counter() - started)
            returned_rankings.append(result_ids[0])

        recall = np.mean(
            [
                len(set(expected).intersection(returned)) / k
                for expected, returned in zip(exact_rankings, returned_rankings)
            ]
        )
        metrics = _percentiles(timings)
        metrics.update(
            {
                "bit_width": bit_width,
                "build_ms": round(build_seconds * 1_000, 2),
                "index_bytes": index_path.stat().st_size,
                "recall_at_k": round(float(recall), 4),
            }
        )
        return metrics


def benchmark_qdrant(
    vectors: np.ndarray,
    queries: np.ndarray,
    exact_rankings: np.ndarray,
    k: int,
) -> dict[str, float]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    # In-memory mode avoids Windows file-lock dependencies while measuring the
    # same local search implementation used by QdrantClient.
    started = time.perf_counter()
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            collection_name="vectors",
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        batch_size = 1_000
        for start in range(0, len(vectors), batch_size):
            client.upsert(
                collection_name="vectors",
                points=[
                    PointStruct(id=index, vector=vector.tolist(), payload={})
                    for index, vector in enumerate(vectors[start : start + batch_size], start)
                ],
                wait=True,
            )
        build_seconds = time.perf_counter() - started

        client.query_points(collection_name="vectors", query=queries[0].tolist(), limit=k)
        timings: list[float] = []
        returned_rankings: list[np.ndarray] = []
        for query in queries:
            started = time.perf_counter()
            response = client.query_points(
                collection_name="vectors", query=query.tolist(), limit=k
            )
            timings.append(time.perf_counter() - started)
            returned_rankings.append(np.asarray([point.id for point in response.points]))

        recall = np.mean(
            [
                len(set(expected).intersection(returned)) / k
                for expected, returned in zip(exact_rankings, returned_rankings)
            ]
        )
        metrics = _percentiles(timings)
        metrics.update(
            {
                "build_ms": round(build_seconds * 1_000, 2),
                "recall_at_k": round(float(recall), 4),
            }
        )
        return metrics
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qdrant", action="store_true", help="Include embedded local Qdrant.")
    args = parser.parse_args()
    if args.dim % 8:
        parser.error("--dim must be divisible by 8 for turbovec")

    rng = np.random.default_rng(args.seed)
    vectors = _normalize(rng.standard_normal((args.vectors, args.dim), dtype=np.float32)).astype(np.float32)
    selected = rng.choice(args.vectors, size=args.queries, replace=False)
    queries = _normalize(vectors[selected] + rng.normal(0, 0.05, (args.queries, args.dim)).astype(np.float32))
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    exact_rankings, exact_metrics = _time_exact(vectors, queries, args.k)
    turbovec_results = [
        benchmark_turbovec(vectors, queries, exact_rankings, bit_width, args.k)
        for bit_width in (2, 4)
    ]
    raw_bytes = vectors.nbytes
    result = {
        "workload": {
            "vectors": args.vectors,
            "queries": args.queries,
            "dimension": args.dim,
            "k": args.k,
            "seed": args.seed,
            "raw_float32_bytes": raw_bytes,
        },
        "exact_float32": exact_metrics,
        "turbovec": [
            item | {"compression_ratio_vs_raw": round(raw_bytes / item["index_bytes"], 2)}
            for item in turbovec_results
        ],
    }
    if args.qdrant:
        result["qdrant_local"] = benchmark_qdrant(vectors, queries, exact_rankings, args.k)
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
