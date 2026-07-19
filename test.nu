def main [var: string] {
    match $var {
        "scalar" | "iter" => {
            cargo run --release --example $"($var)-test" -q
        }
        "vector" => {
            cargo run --release --example $"($var)-test" --features "cabi,simd" -q
        }
        "wide" => {
            cargo run --release --example $"($var)-test" --features "wide" -q
        }
        _ => {
            print $"unknown variant: ($var)"
        }
    }
}
