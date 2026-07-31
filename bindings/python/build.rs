use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git_output(workspace: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_path(workspace: &Path, name: &str) -> Option<PathBuf> {
    git_output(
        workspace,
        &["rev-parse", "--path-format=absolute", "--git-path", name],
    )
    .map(PathBuf::from)
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.join("../..");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ALLOYGBM_BUILD_SOURCE_COMMIT");
    println!("cargo:rerun-if-env-changed=ALLOYGBM_BUILD_SOURCE_DIRTY");
    if let Some(head) = git_path(&workspace, "HEAD") {
        println!("cargo:rerun-if-changed={}", head.display());
    }
    if let Some(reference) = git_output(&workspace, &["symbolic-ref", "-q", "HEAD"])
        && let Some(reference_path) = git_path(&workspace, &reference)
    {
        println!("cargo:rerun-if-changed={}", reference_path.display());
    }

    let commit = env::var("ALLOYGBM_BUILD_SOURCE_COMMIT")
        .ok()
        .or_else(|| git_output(&workspace, &["rev-parse", "HEAD"]))
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".to_owned())
        .to_ascii_lowercase();
    let dirty = env::var("ALLOYGBM_BUILD_SOURCE_DIRTY")
        .ok()
        .unwrap_or_else(|| {
            git_output(
                &workspace,
                &["status", "--porcelain", "--untracked-files=no"],
            )
            .map_or_else(
                || "unknown".to_owned(),
                |status| (!status.is_empty()).to_string(),
            )
        });

    println!("cargo:rustc-env=ALLOYGBM_BUILD_SOURCE_COMMIT={commit}");
    println!("cargo:rustc-env=ALLOYGBM_BUILD_SOURCE_DIRTY={dirty}");
}
