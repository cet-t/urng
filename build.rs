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
                let new_line = format!("{}{}{}{}", &line[..start + prefix.len()], version, "\"", &after[end + 1..]);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    update_readme_version()?;

    let cs_path = "./target/release/RngNative.cs";

    csbindgen::Builder::default()
        .input_extern_file("src/rng32.rs")
        .input_extern_file("src/cabi32.rs")
        .input_extern_file("src/rng64.rs")
        .input_extern_file("src/cabi64.rs")
        .input_extern_file("src/rng128.rs")
        .csharp_dll_name("urng")
        .generate_csharp_file(cs_path)?;

    let content = std::fs::read_to_string(cs_path)?;
    let stripped = strip_struct_fields(&content);
    std::fs::write(cs_path, stripped)?;

    Ok(())
}
