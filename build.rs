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
    // 改行コード (CRLF/LF) やファイル末尾の状態を保持するため、行分割せず
    // 生の文字列に対して `urng = "<version>"` の部分だけを置換する。
    let prefix = "urng = \"";
    let mut result = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(start) = rest.find(prefix) {
        let value_start = start + prefix.len();
        if let Some(rel_end) = rest[value_start..].find('"') {
            let value_end = value_start + rel_end;
            result.push_str(&rest[..value_start]);
            result.push_str(version);
            result.push('"');
            rest = &rest[value_end + 1..];
        } else {
            // 閉じクォートが無い場合はそのまま残す
            result.push_str(&rest[..value_start]);
            rest = &rest[value_start..];
        }
    }
    result.push_str(rest);
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // バージョン変更時に README.md を更新するため Cargo.toml の変更を追跡
    println!("cargo:rerun-if-changed=Cargo.toml");

    // docs.rs では read-only filesystem につき書き込みをスキップ
    if std::env::var("DOCS_RS").is_ok() {
        return Ok(());
    }

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
        .input_extern_file("src/rng32/jsf.rs")
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
        .input_extern_file("src/cabi32/jsf.rs")
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
