use csbindgen;

fn main() {
    csbindgen::Builder::default()
        .input_extern_file("src/rng32.rs")
        .input_extern_file("src/rng64.rs")
        .input_extern_file("src/rng128.rs")
        .csharp_dll_name("rng_core")
        .generate_csharp_file("./target/release/RngNative.cs")
        .unwrap();
}
