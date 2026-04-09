use csbindgen;
use std::result::Result;

fn strip_struct_fields(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut in_struct_body = false;
    let mut awaiting_open = false;
    let mut depth: i32 = 0;

    for line in content.lines() {
        let trimmed = line.trim();

        if !in_struct_body && !awaiting_open {
            result.push_str(line);
            result.push('\n');
            if trimmed.starts_with("internal unsafe partial struct ") {
                awaiting_open = true;
            }
        } else if awaiting_open {
            result.push_str(line);
            result.push('\n');
            if trimmed == "{" {
                in_struct_body = true;
                awaiting_open = false;
                depth = 1;
            }
        } else {
            // inside struct body
            let opens = trimmed.chars().filter(|&c| c == '{').count() as i32;
            let closes = trimmed.chars().filter(|&c| c == '}').count() as i32;
            depth += opens - closes;
            if depth == 0 {
                result.push_str(line);
                result.push('\n');
                in_struct_body = false;
            }
            // else: skip member field lines
        }
    }
    result
}

fn update_readme_version() -> Result<(), Box<dyn std::error::Error>> {
    let version = std::env::var("CARGO_PKG_VERSION")?;
    let readme_path = "./README.md";
    let content = std::fs::read_to_string(readme_path)?;
    let updated = regex_replace_version(&content, &version);
    if updated != content {
        std::fs::write(readme_path, updated)?;
    }
    Ok(())
}

fn regex_replace_version(content: &str, version: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let prefix = "urng = \"";
    for line in content.lines() {
        if let Some(start) = line.find(prefix) {
            let after = &line[start + prefix.len()..];
            if let Some(end) = after.find('"') {
                let new_line = format!(
                    "{}{}{}{}",
                    &line[..start + prefix.len()],
                    version,
                    "\"",
                    &after[end + 1..]
                );
                result.push_str(&new_line);
                result.push('\n');
                continue;
            }
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Collect all `.c` files directly inside `dir` (non-recursive).
fn c_files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
        .collect()
}

fn link_testu01() {
    // TestU01 is compiled from source at build time using the cc crate.
    // This ensures ABI compatibility with whatever compiler Rust uses (cl.exe on MSVC).
    //
    // Set TESTU01_SRC_DIR to the root of the unpacked TestU01 source tree, e.g.:
    //   set TESTU01_SRC_DIR=C:\Downloads\TestU01\TestU01-1.2.3
    //
    // Only active for debug builds (cargo build / cargo test).
    let profile = std::env::var("PROFILE").unwrap_or_default();
    if profile != "debug" {
        return;
    }
    let Ok(src_dir) = std::env::var("TESTU01_SRC_DIR") else {
        println!(
            "cargo:warning=TESTU01_SRC_DIR not set; \
             testu01 module will not be available \
             (set TESTU01_SRC_DIR=<path/to/TestU01-1.2.3> to enable)"
        );
        return;
    };

    let src = std::path::Path::new(&src_dir);
    let include = src.join("include");
    // sincos() is POSIX/GNU only — MSVC CRT lacks it.
    // We compile a tiny compat shim into the testu01 library itself.
    let sincos_compat = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/testing/sincos_compat.c");
    let msvc_compat = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/testing/msvc_compat");

    let mut base = cc::Build::new();
    base.include(&include).warnings(false);
    if base.get_compiler().is_like_msvc() {
        // Prepend our compat dir so stub headers (unistd.h) shadow the missing
        // POSIX originals. config.h (in TestU01 include/) already has
        // HAVE_WINDOWS_H=1; HAVE_CONFIG_H tells each .c to include it.
        base.include(&msvc_compat)
            .define("HAVE_CONFIG_H", None);
    } else {
        base.flag("-std=gnu89");
    }

    // mylib — low-level utilities
    // On MSVC, num2.c contains `1.0 / 0.0` which is a hard error (C2124).
    // Patch it to HUGE_VAL in OUT_DIR before compilation.
    let mut mylib_files = c_files_in(&src.join("mylib"));
    let patched_num2 = if base.get_compiler().is_like_msvc() {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let num2_src = src.join("mylib/num2.c");
        let content = std::fs::read_to_string(&num2_src).expect("read num2.c");
        let patched = content.replace("1.0 / 0.0", "HUGE_VAL");
        let dest = std::path::PathBuf::from(&out_dir).join("num2_msvc.c");
        std::fs::write(&dest, patched).expect("write num2_msvc.c");
        mylib_files.retain(|p| p.file_name().and_then(|n| n.to_str()) != Some("num2.c"));
        Some(dest)
    } else {
        None
    };
    let mut mylib = base.clone();
    mylib.files(&mylib_files);
    if let Some(p) = &patched_num2 { mylib.file(p); }
    mylib.compile("mylib");

    // probdist — probability distributions (depends on mylib)
    base.clone().files(c_files_in(&src.join("probdist"))).compile("probdist");

    // testu01 — test batteries (depends on probdist + mylib)
    // Include sincos_compat.c on MSVC so the shim is linked into the lib.
    let mut testu01 = base.clone();
    testu01.files(c_files_in(&src.join("testu01")));
    if testu01.get_compiler().is_like_msvc() {
        testu01.file(&sincos_compat);
    }
    testu01.compile("testu01");

    println!("cargo:rustc-cfg=has_testu01");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    link_testu01();
    update_readme_version()?;

    let cs_path = "./target/release/RngNative.cs";

    csbindgen::Builder::default()
        // rng32 algorithm implementations (struct definitions)
        .input_extern_file("src/rng32/splitmix.rs")
        .input_extern_file("src/rng32/mersenne.rs")
        .input_extern_file("src/rng32/lcg.rs")
        .input_extern_file("src/rng32/pcg.rs")
        .input_extern_file("src/rng32/philox.rs")
        .input_extern_file("src/rng32/xorshift.rs")
        .input_extern_file("src/rng32/threefry.rs")
        .input_extern_file("src/rng32/squares.rs")
        .input_extern_file("src/rng32/xoshiro.rs")
        // cabi32 C ABI bindings
        .input_extern_file("src/cabi32/mersenne.rs")
        .input_extern_file("src/cabi32/lcg.rs")
        .input_extern_file("src/cabi32/pcg.rs")
        .input_extern_file("src/cabi32/philox.rs")
        .input_extern_file("src/cabi32/xorshift.rs")
        .input_extern_file("src/cabi32/splitmix.rs")
        .input_extern_file("src/cabi32/threefry.rs")
        .input_extern_file("src/cabi32/squares.rs")
        .input_extern_file("src/cabi32/xoshiro.rs")
        // rng64 algorithm implementations (struct definitions)
        .input_extern_file("src/rng64/splitmix.rs")
        .input_extern_file("src/rng64/mersenne.rs")
        .input_extern_file("src/rng64/twisted_gfsr.rs")
        .input_extern_file("src/rng64/lcg.rs")
        .input_extern_file("src/rng64/philox.rs")
        .input_extern_file("src/rng64/sfc.rs")
        .input_extern_file("src/rng64/xorshift.rs")
        .input_extern_file("src/rng64/cet.rs")
        .input_extern_file("src/rng64/xoshiro.rs")
        .input_extern_file("src/rng64/threefish.rs")
        .input_extern_file("src/rng64/xoroshiro.rs")
        .input_extern_file("src/rng64/biski.rs")
        // cabi64 C ABI bindings
        .input_extern_file("src/cabi64/mersenne.rs")
        .input_extern_file("src/cabi64/twisted_gfsr.rs")
        .input_extern_file("src/cabi64/lcg.rs")
        .input_extern_file("src/cabi64/philox.rs")
        .input_extern_file("src/cabi64/sfc.rs")
        .input_extern_file("src/cabi64/xorshift.rs")
        .input_extern_file("src/cabi64/cet.rs")
        .input_extern_file("src/cabi64/xoshiro.rs")
        .input_extern_file("src/cabi64/splitmix.rs")
        .input_extern_file("src/cabi64/threefish.rs")
        .input_extern_file("src/cabi64/xoroshiro.rs")
        .csharp_dll_name("urng")
        .generate_csharp_file(cs_path)?;

    let content = std::fs::read_to_string(cs_path)?;
    let stripped = strip_struct_fields(&content);
    std::fs::write(cs_path, stripped)?;

    Ok(())
}
