//! SHARD-style data-oblivious KV-cache streaming quantization.
//!
//! This is the portable portion of the SHARD design: normalize a token vector,
//! apply a signed orthonormal Hadamard transform, quantize with a Lloyd-Max
//! codebook for N(0, 1/d), and bit-pack the codes. It deliberately does not
//! implement Llama-specific PCA/VQ or fused attention; those require owning a
//! compatible model runtime and its attention kernels.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct ShardStreamQuantizer {
    dim: usize,
    bits: u8,
    signs: Vec<f32>,
    codebook: Vec<f32>,
}

#[pymethods]
impl ShardStreamQuantizer {
    /// Create a deterministic SHARD-style streaming quantizer.
    #[new]
    #[pyo3(signature = (dim, bits=8, seed=42))]
    fn new(dim: usize, bits: u8, seed: u64) -> PyResult<Self> {
        validate_parameters(dim, bits)?;
        Ok(Self {
            dim,
            bits,
            signs: make_signs(dim, seed),
            codebook: lloyd_max_codebook(dim, bits),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn bits(&self) -> u8 {
        self.bits
    }

    /// Encode vectors into packed quantization codes and their original norms.
    ///
    /// The input is one token vector per inner list. The returned byte strings
    /// use exactly `ceil(dim * bits / 8)` bytes per vector.
    fn encode(&self, vectors: Vec<Vec<f32>>) -> PyResult<(Vec<Vec<u8>>, Vec<f32>)> {
        let mut packed = Vec::with_capacity(vectors.len());
        let mut norms = Vec::with_capacity(vectors.len());

        for vector in vectors {
            if vector.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Expected vector dimension {}, got {}.",
                    self.dim,
                    vector.len()
                )));
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "Vectors must contain only finite values.",
                ));
            }

            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            let mut rotated = if norm > 0.0 {
                vector
                    .iter()
                    .zip(&self.signs)
                    .map(|(value, sign)| value / norm * sign)
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; self.dim]
            };
            hadamard_in_place(&mut rotated);
            let codes = rotated
                .iter()
                .map(|value| nearest_code(&self.codebook, *value) as u8)
                .collect::<Vec<_>>();
            packed.push(pack_codes(&codes, self.bits));
            norms.push(norm);
        }
        Ok((packed, norms))
    }

    /// Reconstruct vectors previously returned by `encode`.
    fn decode(&self, packed: Vec<Vec<u8>>, norms: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        if packed.len() != norms.len() {
            return Err(PyValueError::new_err(
                "Packed vectors and norms must have equal length.",
            ));
        }
        let expected_bytes = self.compressed_bytes_per_vector();
        let mut vectors = Vec::with_capacity(packed.len());
        for (bytes, norm) in packed.iter().zip(norms) {
            if bytes.len() != expected_bytes {
                return Err(PyValueError::new_err(format!(
                    "Expected {} packed bytes per vector, got {}.",
                    expected_bytes,
                    bytes.len()
                )));
            }
            if !norm.is_finite() || norm < 0.0 {
                return Err(PyValueError::new_err(
                    "Norms must be finite and non-negative.",
                ));
            }
            let codes = unpack_codes(bytes, self.dim, self.bits);
            let mut vector = codes
                .iter()
                .map(|code| self.codebook[*code as usize])
                .collect::<Vec<_>>();
            hadamard_in_place(&mut vector);
            for (value, sign) in vector.iter_mut().zip(&self.signs) {
                *value *= sign * norm;
            }
            vectors.push(vector);
        }
        Ok(vectors)
    }

    /// Return the packed code bytes required for one token vector.
    fn compressed_bytes_per_vector(&self) -> usize {
        (self.dim * self.bits as usize).div_ceil(8)
    }
}

fn validate_parameters(dim: usize, bits: u8) -> PyResult<()> {
    if dim == 0 || !dim.is_power_of_two() {
        return Err(PyValueError::new_err(
            "SHARD streaming quantization requires a positive power-of-two dimension.",
        ));
    }
    if !(2..=8).contains(&bits) {
        return Err(PyValueError::new_err("bits must be between 2 and 8."));
    }
    Ok(())
}

fn make_signs(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Fast Walsh-Hadamard transform normalized to be orthonormal and involutory.
fn hadamard_in_place(values: &mut [f32]) {
    let mut width = 1;
    while width < values.len() {
        for start in (0..values.len()).step_by(width * 2) {
            for offset in 0..width {
                let left = values[start + offset];
                let right = values[start + offset + width];
                values[start + offset] = left + right;
                values[start + offset + width] = left - right;
            }
        }
        width *= 2;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

fn nearest_code(codebook: &[f32], value: f32) -> usize {
    let insertion = codebook.partition_point(|centroid| *centroid < value);
    match insertion {
        0 => 0,
        index if index == codebook.len() => codebook.len() - 1,
        index => {
            if (value - codebook[index - 1]).abs() <= (codebook[index] - value).abs() {
                index - 1
            } else {
                index
            }
        }
    }
}

fn pack_codes(codes: &[u8], bits: u8) -> Vec<u8> {
    let mut output = vec![0_u8; (codes.len() * bits as usize).div_ceil(8)];
    let mut bit_offset = 0;
    for code in codes {
        let byte_index = bit_offset / 8;
        let intra_byte = bit_offset % 8;
        let value = *code as u16;
        output[byte_index] |= (value << intra_byte) as u8;
        if intra_byte + bits as usize > 8 {
            output[byte_index + 1] |= (value >> (8 - intra_byte)) as u8;
        }
        bit_offset += bits as usize;
    }
    output
}

fn unpack_codes(bytes: &[u8], count: usize, bits: u8) -> Vec<u8> {
    let mask = (1_u16 << bits) - 1;
    (0..count)
        .map(|index| {
            let bit_offset = index * bits as usize;
            let byte_index = bit_offset / 8;
            let intra_byte = bit_offset % 8;
            let mut value = (bytes[byte_index] as u16) >> intra_byte;
            if intra_byte + bits as usize > 8 {
                value |= (bytes[byte_index + 1] as u16) << (8 - intra_byte);
            }
            (value & mask) as u8
        })
        .collect()
}

fn normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Abramowitz-Stegun approximation; accurate enough for codebook construction.
fn normal_cdf(value: f64) -> f64 {
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let polynomial =
        (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782) * t
            + 0.319_381_530)
            * t;
    let cdf = 1.0 - normal_pdf(x) * polynomial;
    if value >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}

fn inverse_normal_cdf(probability: f64) -> f64 {
    let mut low = -8.0;
    let mut high = 8.0;
    for _ in 0..64 {
        let midpoint = (low + high) / 2.0;
        if normal_cdf(midpoint) < probability {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    (low + high) / 2.0
}

fn lloyd_max_codebook(dim: usize, bits: u8) -> Vec<f32> {
    let count = 1_usize << bits;
    let sigma = 1.0 / (dim as f64).sqrt();
    let mut centroids = (0..count)
        .map(|index| inverse_normal_cdf((index as f64 + 0.5) / count as f64) * sigma)
        .collect::<Vec<_>>();
    for _ in 0..100 {
        let mut bounds = Vec::with_capacity(count + 1);
        bounds.push(-1_000_000.0);
        bounds.extend((1..count).map(|index| (centroids[index - 1] + centroids[index]) / 2.0));
        bounds.push(1_000_000.0);
        let next = (0..count)
            .map(|index| {
                let lower = bounds[index] / sigma;
                let upper = bounds[index + 1] / sigma;
                let mass = normal_cdf(upper) - normal_cdf(lower);
                if mass < 1e-12 {
                    centroids[index]
                } else {
                    (normal_pdf(lower) - normal_pdf(upper)) * sigma / mass
                }
            })
            .collect::<Vec<_>>();
        let max_delta = centroids
            .iter()
            .zip(&next)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        centroids = next;
        if max_delta < 1e-9 {
            break;
        }
    }
    centroids.into_iter().map(|value| value as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_codes_round_trip() {
        let codes = vec![0, 1, 2, 3, 3, 2, 1, 0, 1, 2, 3];
        let packed = pack_codes(&codes, 2);
        assert_eq!(unpack_codes(&packed, codes.len(), 2), codes);
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let mut values = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0];
        let original = values.clone();
        hadamard_in_place(&mut values);
        hadamard_in_place(&mut values);
        for (actual, expected) in values.iter().zip(original) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn high_precision_round_trip_preserves_direction() {
        let quantizer = ShardStreamQuantizer::new(128, 8, 42).unwrap();
        let vector = (0..128)
            .map(|index| ((index as f32 * 0.17).sin()) + 0.2)
            .collect::<Vec<_>>();
        let (packed, norms) = quantizer.encode(vec![vector.clone()]).unwrap();
        let restored = quantizer.decode(packed, norms).unwrap().pop().unwrap();
        let dot = vector
            .iter()
            .zip(&restored)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let left_norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm = restored
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(dot / (left_norm * right_norm) > 0.99);
    }

    #[test]
    #[ignore = "Run explicitly to collect no-model SHARD streaming quantizer metrics."]
    fn benchmark_streaming_quantizer() {
        use std::time::Instant;

        const TOKENS: usize = 50_000;
        const DIM: usize = 128;
        let vectors = (0..TOKENS)
            .map(|token| {
                (0..DIM)
                    .map(|dimension| {
                        ((token as f32 * 0.013) + (dimension as f32 * 0.071)).sin() + 0.1
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for bits in [4, 8] {
            let quantizer = ShardStreamQuantizer::new(DIM, bits, 42).unwrap();
            let started = Instant::now();
            let (packed, norms) = quantizer.encode(vectors.clone()).unwrap();
            let encode_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let _decoded = quantizer.decode(packed, norms).unwrap();
            let decode_seconds = started.elapsed().as_secs_f64();
            let packed_bytes = quantizer.compressed_bytes_per_vector() + std::mem::size_of::<f32>();
            let fp16_bytes = DIM * std::mem::size_of::<u16>();
            println!(
                "SHARD_STREAM_METRIC bits={bits} tokens={TOKENS} dim={DIM} \
                 encode_tokens_per_second={:.2} decode_tokens_per_second={:.2} \
                 packed_bytes_per_token={packed_bytes} fp16_bytes_per_token={fp16_bytes} \
                 compression_ratio={:.2}",
                TOKENS as f64 / encode_seconds,
                TOKENS as f64 / decode_seconds,
                fp16_bytes as f64 / packed_bytes as f64,
            );
        }
    }
}
