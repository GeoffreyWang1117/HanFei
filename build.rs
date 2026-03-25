use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if !cfg!(feature = "gpu") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let prebuilt_dir = manifest_dir.join("prebuilt");

    // Detect GPU compute capability
    let arch = detect_gpu_arch().unwrap_or_else(|| {
        let fallback = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
        println!("cargo:warning=Could not detect GPU arch, using {}", fallback);
        fallback
    });

    let lib_dir = prebuilt_dir.join(&arch);
    if !lib_dir.join("libmsm_kernel.a").exists() {
        println!(
            "cargo:warning=No prebuilt kernel for {}. Available: sm_86, sm_89, sm_90.",
            arch
        );
        println!("cargo:warning=Falling back to sm_86.");
        let lib_dir = prebuilt_dir.join("sm_86");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    println!("cargo:rustc-link-lib=static=msm_kernel");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // CUDA library path
    let cuda_path = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
}

fn detect_gpu_arch() -> Option<String> {
    // Try nvidia-smi to detect compute capability
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let cap = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();

    // Map compute capability to sm_ string
    let sm = match cap.as_str() {
        "8.6" => "sm_86",
        "8.9" => "sm_89",
        "9.0" => "sm_90",
        _ => {
            // Find closest match
            if cap.starts_with("8.") {
                "sm_86"
            } else if cap.starts_with("9.") {
                "sm_90"
            } else {
                return None;
            }
        }
    };

    Some(sm.to_string())
}
