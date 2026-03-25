# pallas-gpu-msm

**The first — and currently only — GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve.**

<p align="center">
  <img src="docs/benchmark.svg" alt="GPU vs CPU benchmark" width="720">
</p>

## What is this?

[Multi-Scalar Multiplication (MSM)](https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication) is the computational bottleneck in zero-knowledge proof systems. It computes:

```
result = s_1 * G_1 + s_2 * G_2 + ... + s_n * G_n
```

where `s_i` are scalars and `G_i` are elliptic curve points. In Halo2's IPA proving scheme, MSM accounts for **60-70% of total proving time**. For an attention layer with d=256, this means MSM alone takes over 100 seconds on CPU.

**pallas-gpu-msm** accelerates this operation on NVIDIA GPUs for the **Pallas curve** -- one half of the [Pasta curve cycle](https://electriccoin.co/blog/the-pasta-curves-for-halo-2-and-beyond/) used by [Halo2](https://zcash.github.io/halo2/), [Zcash](https://z.cash/), and other recursive proof systems.

Existing GPU MSM libraries (ICICLE, cuZK, Blitzar) only support BN254 and BLS12-381. **This crate is the first to bring GPU acceleration to Pallas.**

## Origin

This crate is the GPU MSM component of [ChainProve (nanoZkinference)](https://github.com/GeoffreyWang1117/nanoZkinference), a system for verifiable transformer inference using zero-knowledge proofs. It sits in the proving pipeline as:

```
ChainProve Proving Pipeline:

  Python API (nanozk_halo2)
      ↓
  Halo2 Backend (Rust/PyO3)
      ↓
  Forked halo2_proofs (MSM patched)
      ↓
  pallas-gpu-msm  ← this crate
      ↓
  CUDA Kernels (prebuilt)
```

Related paper:

> Zhaohui Wang. *Verifiable Transformer Inference on NanoGPT: A Layerwise zkML Prototype.* arXiv preprint, 2025.

## Features

- **GPU-accelerated MSM** for the Pallas curve on NVIDIA GPUs (Ampere, Ada, Hopper)
- **Drop-in replacement** for `halo2curves::msm::best_multiexp` -- same types, same API
- **Automatic fallback** to CPU when no GPU is available or for small inputs (< 8K points)
- **Prebuilt CUDA kernels** -- no CUDA Toolkit or nvcc required to build
- **Bit-exact correctness** -- GPU results match CPU across all tested input sizes (k=10..21)

## Landscape: No Existing Alternatives

As of March 2026, **no other library provides GPU-accelerated MSM for the Pallas curve**:

| Library | BN254 | BLS12-381 | Pallas |
|---------|-------|-----------|--------|
| [ICICLE](https://github.com/ingonyama-zk/icicle) (Ingonyama) | Yes | Yes | **No** |
| [cuZK](https://github.com/speakspeak/cuZK) | Yes | No | **No** |
| [Blitzar](https://github.com/spaceandtimelabs/blitzar) | Yes | Yes | **No** |
| [ec-gpu](https://github.com/filecoin-project/ec-gpu) | Yes | Yes | **No** |
| [halo2curves](https://github.com/privacy-scaling-explorations/halo2curves) | — | — | CPU only |
| **pallas-gpu-msm (this crate)** | — | — | **Yes** |

The entire Halo2/Zcash/Pasta ecosystem currently has no GPU MSM option. This crate is the only one.

## Performance

### Current Benchmark (v0.1.0)

RTX 3090 (Ampere, 82 SMs) vs Ryzen 9 5950X (32 threads):

| Input size | GPU (ms) | CPU (ms) | Speedup | Correct |
|-----------|----------|----------|---------|---------|
| 64K (k=16) | 44.9 | 47.7 | **1.06x** | OK |
| 128K (k=17) | 72.2 | 81.3 | **1.12x** | OK |
| 256K (k=18) | 117.5 | 152.8 | **1.30x** | OK |
| 512K (k=19) | 218.4 | 289.6 | **1.33x** | OK |
| 1M (k=20) | 417.0 | 491.4 | **1.18x** | OK |
| 2M (k=21) | 900.2 | 933.0 | **1.04x** | OK |

GPU wins at k >= 16 (64K+ points). Peak speedup **1.33x** at k=19. All results **bit-exact** (best-of-N runs).

### Progress vs. Early Development

| k | Early prototype | Current v0.1.0 | Improvement |
|---|----------------|----------------|-------------|
| 17 | 0.61x (GPU slower) | **1.12x** (GPU wins) | 1.8x |
| 19 | 0.84x | **1.33x** | 1.6x |
| 21 | 0.77x | **1.04x** | 1.4x |

### Reproduce

```bash
cargo run --release --example full_benchmark
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
pallas-gpu-msm = { git = "https://github.com/GeoffreyWang1117/pallas-gpu-msm" }
```

Use in your code:

```rust
use pallas_gpu_msm::{gpu_best_multiexp, is_gpu_available};

println!("GPU available: {}", is_gpu_available());

// Same signature as halo2curves::msm::best_multiexp
let result = gpu_best_multiexp(&scalars, &bases);
```

## Requirements

- **NVIDIA GPU**: compute capability >= 8.6 (RTX 3090, RTX 4090, A100, H100, etc.)
- **CUDA Runtime**: `libcudart` must be installed (comes with NVIDIA driver)
- **Rust**: 1.70+
- No nvcc or CUDA Toolkit needed -- kernels are prebuilt

## Supported GPUs

| Architecture | Example GPUs | Prebuilt |
|-------------|-------------|----------|
| sm_86 (Ampere) | RTX 3090, A6000, A100 | Yes |
| sm_89 (Ada) | RTX 4090, L40 | Yes |
| sm_90 (Hopper) | H100, H200 | Yes |

Override auto-detection: `CUDA_ARCH=sm_XX cargo build`

## API

```rust
/// Check if a CUDA GPU is available at runtime (cached after first call).
pub fn is_gpu_available() -> bool;

/// Compute MSM: result = sum(coeffs[i] * bases[i]).
/// GPU for inputs >= 8K points, CPU fallback otherwise.
/// Panics if coeffs.len() != bases.len().
pub fn gpu_best_multiexp(coeffs: &[Scalar], bases: &[Affine]) -> Point;
```

Types `Scalar`, `Affine`, `Point` are re-exported from `halo2curves::pasta::pallas`.

## Running Tests and Benchmarks

```bash
# Unit tests (includes GPU correctness verification)
cargo test --release

# Quick benchmark (k=10..21, GPU vs CPU, with correctness check)
cargo run --release --example full_benchmark

# Criterion benchmark (k=14,17,19, statistical analysis)
cargo bench
```

## Development Roadmap

### v0.1.0 (current) -- First GPU MSM for Pallas

- [x] Complete GPU MSM pipeline with bit-exact correctness
- [x] Rust FFI with automatic CPU fallback
- [x] GPU surpasses CPU at k >= 15 (up to 1.21x)
- [x] Prebuilt kernels for Ampere, Ada, Hopper
- [x] Open-source release

### v0.2.0 (target: 2026 Q2) -- 5-10x speedup

- [ ] Kernel-level parallelism improvements
- [ ] Better GPU occupancy and memory access patterns
- [ ] Target: consistent 5x+ over CPU at k=17-21

### v0.3.0 (target: 2026 Q3) -- Production-ready

- [ ] Additional kernel optimizations for throughput
- [ ] PyO3 Python bindings (`pip install pallas-gpu-msm`)
- [ ] Multi-GPU support for very large MSMs
- [ ] Vesta curve support (complete Pasta cycle)

### Long-term

- [ ] Target top-venue publication (TCHES / ASPLOS)
- [ ] Become the default GPU MSM backend for the Halo2/Pasta ecosystem

## Project Timeline

```
2025      Paper: "Verifiable Transformer Inference on NanoGPT" (arXiv)
2026 Q1   v0.1.0: First Pallas GPU MSM, GPU > CPU at large k
          VerifAI @ ICLR 2026 (accepted), BlockSys 2026 (submitted)
2026 Q2   v0.2.0: Major kernel improvements, target 5-10x
          ICICS 2026 submission
2026 Q3   v0.3.0: Python bindings, multi-GPU, Vesta curve
          Standalone GPU MSM paper draft
2026 Q4   Production release, community adoption
2027      Top-venue submission (TCHES / ASPLOS / CCS)
```

## Contributors

- **Zhaohui (Geoffrey) Wang** -- Design, development, and paper
- **Claude Opus (Anthropic)** -- Research assistance, experiments, and performance tuning

## License

[Apache License 2.0](LICENSE)

The CUDA kernels are distributed as prebuilt binaries only. See [NOTICE](NOTICE).

For collaboration or commercial inquiries: **zhaohui.geoffrey.wang@gmail.com**

## Citation

This project is part of the ChainProve/nanoZkinference system:

```bibtex
@article{nanogpt-zkinference,
  title={Verifiable Transformer Inference on NanoGPT: A Layerwise zkML Prototype},
  author={Zhaohui Wang},
  journal={arXiv preprint},
  year={2025}
}
```

```bibtex
@software{pallas_gpu_msm,
  title={pallas-gpu-msm: GPU-Accelerated MSM for the Pallas Curve},
  author={Zhaohui Wang},
  year={2026},
  url={https://github.com/GeoffreyWang1117/pallas-gpu-msm}
}
```
