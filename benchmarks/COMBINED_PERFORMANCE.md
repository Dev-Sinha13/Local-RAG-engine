# Retrieval + KV-cache compression impact

Collected 2026-07-14 without downloading or loading a language model.

The two features optimize independent resources:

- **turbovec** compresses the persistent document-embedding index used during retrieval.
- **SHARD-style streaming quantization** compresses active per-token KV-cache storage during generation.

Their storage savings therefore apply to different components and must **not**
be multiplied into a single headline number.

## Measured results

### turbovec retrieval benchmark

Workload: 20,000 normalized synthetic 384-dimensional vectors, 200 queries,
and top-10 retrieval. Quality is Recall@10 against exact float32 ranking.
Full machine-readable results are in `turbovec_benchmark_results.json`.

| Backend | Recall@10 | p50 query latency | p95 query latency | QPS | Index storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact float32 NumPy | 1.000 | 2.06 ms | 3.83 ms | 469.5 | 30.72 MB raw |
| turbovec 2-bit | 0.564 | 1.89 ms | 2.94 ms | 483.3 | 2.16 MB (14.2x smaller) |
| turbovec 4-bit | 0.855 | 4.20 ms | 6.01 ms | 227.6 | 4.08 MB (7.52x smaller) |
| Embedded local Qdrant | 1.000 | 105.33 ms | 128.69 ms | 9.6 | not measured |

### Rust SHARD-style streaming quantizer benchmark

Workload: 50,000 deterministic synthetic token vectors, 128 dimensions,
compiled with `cargo test --release`. This measures only the portable Rust
codec: signed Hadamard rotation, Lloyd-Max quantization, packed codes, and
reconstruction. It does **not** measure a Llama model, PCA/VQ prefill cache,
fused attention, GPU memory, or text-generation quality.

| Codec width | Encode throughput | Decode throughput | Stored bytes/token | FP16 bytes/token | Storage reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-bit | 272,522 tokens/s | 815,946 tokens/s | 68 | 256 | 3.76x |
| 8-bit | 194,870 tokens/s | 854,384 tokens/s | 132 | 256 | 1.94x |

The stored size includes the packed code bytes and one `f32` norm per token.

## Combined runtime effect

For the recommended 4-bit turbovec mode, the persistent vector index drops
from 30.72 MB to 4.08 MB in this workload. During a compatible long-context
generation, the Rust streaming codec can independently reduce the token-vector
portion of an FP16 KV cache by 3.76x at 4-bit, or 1.94x at 8-bit.

The end-to-end request time is approximately:

```text
embedding + retrieval + prompt prefill + token decode
```

turbovec affects only **retrieval**. SHARD affects only **prefill/decode**.
At RustyRAG's default context of three 256-token chunks, SHARD should not be
expected to improve latency or materially reduce memory. It becomes relevant
when long contexts or high concurrent generation load make KV-cache capacity a
bottleneck.

## Not yet measured

No Llama model was downloaded or loaded, so these required end-to-end SHARD
metrics remain intentionally unmeasured:

- GPU VRAM reduction at 8K, 32K, and target production context lengths
- time to first token and decode tokens/second
- answer quality, retrieval-grounding quality, and long-context recall
- combined turbovec + SHARD request latency under concurrent load

When a compatible local Llama-3.1 model is available, collect those metrics by
comparing `LLM_BACKEND=ollama` with `LLM_BACKEND=shard` on the same document
set, prompts, context sizes, and GPU. The upstream SHARD implementation reports
10.0x KV compression at 8K and 11.2x at 32K for Llama-3.1-8B, but those figures
are not local RustyRAG measurements and are not asserted here.
