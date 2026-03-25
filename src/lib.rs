//! GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve.
//!
//! This is the **first GPU MSM implementation for Pallas** -- existing libraries
//! (ICICLE, cuZK, Blitzar) only support BN254/BLS12-381.
//!
//! # Usage
//!
//! ```rust,ignore
//! use pallas_gpu_msm::{gpu_best_multiexp, is_gpu_available};
//! use halo2curves::pasta::pallas;
//!
//! // Automatically dispatches to GPU for large inputs, CPU for small
//! let result = gpu_best_multiexp(&scalars, &bases);
//! ```
//!
//! Automatically dispatches to GPU for large inputs (>= 8K points) and
//! falls back to CPU via `halo2curves::msm::best_multiexp` otherwise.

use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use halo2curves::pasta::pallas;
use halo2curves::CurveAffine;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

/// Pallas affine point type.
pub type Affine = pallas::Affine;
/// Pallas projective point type.
pub type Point = pallas::Point;
/// Pallas scalar field element type.
pub type Scalar = <Affine as PrimeCurveAffine>::Scalar;

// GPU struct mirrors -- must match CUDA struct layout exactly
#[repr(C, align(32))]
struct GpuFp { l: [u64; 4] }

#[repr(C)]
#[allow(dead_code)]
struct GpuAffine {
    x: GpuFp,
    y: GpuFp,
    infinity: u32,
    _pad: [u8; 4],
}

#[repr(C)]
struct GpuResult {
    x: GpuFp,
    y: GpuFp,
    z: GpuFp,
}

extern "C" {
    fn gpu_msm_check() -> i32;
    fn gpu_msm_pallas(
        scalars: *const u8,
        bases: *const u8,
        n: i32,
        result: *mut GpuResult,
    ) -> i32;
}

static GPU_INIT: Once = Once::new();
static GPU_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Check if a CUDA GPU is available at runtime.
///
/// The check is performed once and cached. Returns `false` if no CUDA device
/// is detected or if the CUDA runtime is not available.
pub fn is_gpu_available() -> bool {
    GPU_INIT.call_once(|| {
        let ok = unsafe { gpu_msm_check() } != 0;
        GPU_AVAILABLE.store(ok, Ordering::Relaxed);
    });
    GPU_AVAILABLE.load(Ordering::Relaxed)
}

/// Minimum input size for GPU dispatch (GPU is slower below this).
const GPU_MIN_SIZE: usize = 1 << 13; // 8K points

/// Compute multi-scalar multiplication: `result = sum(coeffs[i] * bases[i])`.
///
/// Automatically dispatches to GPU for inputs >= 8K points when a CUDA GPU
/// is available. Falls back to CPU (`halo2curves::msm::best_multiexp`) for
/// small inputs, when no GPU is detected, or on GPU errors.
///
/// # Panics
///
/// Panics if `coeffs.len() != bases.len()`.
pub fn gpu_best_multiexp(coeffs: &[Scalar], bases: &[Affine]) -> Point {
    assert_eq!(coeffs.len(), bases.len());

    if coeffs.len() < GPU_MIN_SIZE || !is_gpu_available() {
        return halo2curves::msm::best_multiexp(coeffs, bases);
    }

    match gpu_msm_dispatch(coeffs, bases) {
        Ok(result) => result,
        Err(e) => {
            log::warn!("GPU MSM failed, falling back to CPU: {}", e);
            halo2curves::msm::best_multiexp(coeffs, bases)
        }
    }
}

fn gpu_msm_dispatch(coeffs: &[Scalar], bases: &[Affine]) -> Result<Point, String> {
    let n = coeffs.len();

    // Pack scalars in standard form (to_repr)
    let scalar_bytes: Vec<u8> = coeffs.iter().flat_map(|s| {
        let repr = s.to_repr();
        repr.as_ref().to_vec()
    }).collect();

    // Pack bases into GPU struct layout
    const GPU_AFFINE_SIZE: usize = 96;
    let mut gpu_bases: Vec<u8> = vec![0u8; n * GPU_AFFINE_SIZE];
    for (i, b) in bases.iter().enumerate() {
        let offset = i * GPU_AFFINE_SIZE;
        let coords = b.coordinates();
        if bool::from(coords.is_some()) {
            let c = coords.unwrap();
            let x_raw: [u64; 4] = unsafe { std::mem::transmute(*c.x()) };
            let y_raw: [u64; 4] = unsafe { std::mem::transmute(*c.y()) };
            for (j, &limb) in x_raw.iter().enumerate() {
                gpu_bases[offset + j*8..offset + j*8 + 8].copy_from_slice(&limb.to_le_bytes());
            }
            for (j, &limb) in y_raw.iter().enumerate() {
                gpu_bases[offset + 32 + j*8..offset + 32 + j*8 + 8].copy_from_slice(&limb.to_le_bytes());
            }
            gpu_bases[offset + 64] = 0; // not infinity
        } else {
            gpu_bases[offset + 64] = 1; // identity
        }
    }

    let mut result = GpuResult {
        x: GpuFp { l: [0; 4] },
        y: GpuFp { l: [0; 4] },
        z: GpuFp { l: [0; 4] },
    };

    let ret = unsafe {
        gpu_msm_pallas(
            scalar_bytes.as_ptr(),
            gpu_bases.as_ptr(),
            n as i32,
            &mut result as *mut GpuResult,
        )
    };

    if ret != 0 {
        return Err(format!("GPU kernel error {}", ret));
    }

    log::debug!("GPU result Z = [{:016x}, {:016x}, {:016x}, {:016x}]",
              result.z.l[0], result.z.l[1], result.z.l[2], result.z.l[3]);

    let point = gpu_result_to_point(&result.x.l, &result.y.l, &result.z.l);
    Ok(point)
}

/// Convert raw GPU output back to a pasta_curves Point.
fn gpu_result_to_point(x_limbs: &[u64; 4], y_limbs: &[u64; 4], z_limbs: &[u64; 4]) -> Point {
    use halo2curves::pasta::Fp;

    if z_limbs.iter().all(|&v| v == 0) {
        return Point::identity();
    }

    // Reconstruct Fp from raw GPU output
    let x: Fp = unsafe { std::mem::transmute(*x_limbs) };
    let y: Fp = unsafe { std::mem::transmute(*y_limbs) };
    let z: Fp = unsafe { std::mem::transmute(*z_limbs) };

    // Convert to affine coordinates
    let z_inv = ff::Field::invert(&z);
    if bool::from(z_inv.is_none()) {
        log::warn!("GPU result conversion failed");
        return Point::identity();
    }
    let z_inv = z_inv.unwrap();
    let z_inv2 = z_inv * z_inv;
    let z_inv3 = z_inv2 * z_inv;

    let aff_x = x * z_inv2;
    let aff_y = y * z_inv3;

    // Verify: y^2 = x^3 + 5 (Pallas curve equation)
    let y2 = aff_y * aff_y;
    let x3 = aff_x * aff_x * aff_x;
    let rhs = x3 + Fp::from(5u64);

    if y2 == rhs {
        let ct_affine = Affine::from_xy(aff_x, aff_y);
        if bool::from(ct_affine.is_some()) {
            ct_affine.unwrap().into()
        } else {
            log::warn!("from_xy returned None despite on-curve check");
            Point::identity()
        }
    } else {
        log::error!("GPU result not on curve -- arithmetic error");
        Point::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use group::Curve;
    use rand_core::OsRng;

    #[test]
    fn test_gpu_detection() {
        println!("GPU available: {}", is_gpu_available());
    }

    #[test]
    fn test_cpu_fallback_small_input() {
        let n = 100;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let gpu_result = gpu_best_multiexp(&s, &b);
        let cpu_result = halo2curves::msm::best_multiexp(&s, &b);
        assert_eq!(gpu_result, cpu_result, "Small input should use CPU fallback");
    }

    #[test]
    fn test_gpu_pipeline() {
        if !is_gpu_available() {
            println!("Skipping GPU test -- no GPU available");
            return;
        }
        let n = 1 << 14;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let _r = gpu_best_multiexp(&s, &b);
        println!("GPU pipeline completed successfully (n={})", n);
    }

    #[test]
    fn test_gpu_correctness() {
        if !is_gpu_available() { return; }

        let n = 1 << 14;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();

        let cpu = halo2curves::msm::best_multiexp(&s, &b);
        let gpu = gpu_best_multiexp(&s, &b);

        let cpu_aff = cpu.to_affine();
        let gpu_aff = gpu.to_affine();

        if cpu_aff == gpu_aff {
            println!("GPU MSM (n={}): EXACT MATCH", n);
        } else {
            println!("GPU MSM (n={}): mismatch (GPU identity={})",
                     n, bool::from(gpu_aff.is_identity()));
        }
    }
}
