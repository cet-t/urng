use csbindgen;
use std::result::Result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    csbindgen::Builder::default()
        .input_extern_file("src/rng32.rs")
        .input_extern_file("src/rng64.rs")
        .input_extern_file("src/rng128.rs")
        .csharp_dll_name("urng")
        .generate_csharp_file("./target/release/RngNative.cs")?;

    Ok(())
}
