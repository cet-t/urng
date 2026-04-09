# testing モジュール

debug ビルド限定の統計的品質検査ハーネス。  
リリースビルド・配布 cdylib には含まれない。

---

## 対応ツール

| ツール | 方式 | 対応ビット幅 |
|--------|------|-------------|
| [PracRand](http://pracrand.sourceforge.net/) | subprocess (stdin pipe) | 32 / 64 bit |
| [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) | FFI (静的リンク) | 32 / 64 bit |

---

## 環境構築

### PracRand

#### 1. ビルド

```bash
# Linux / macOS
git clone https://sourceforge.net/projects/pracrand/ pracrand
cd pracrand
g++ -O3 -o RNG_test RNG_test.cpp src/*.cpp -Iinclude -std=c++11

# Windows (MinGW / MSYS2)
g++ -O3 -o RNG_test.exe RNG_test.cpp src/*.cpp -Iinclude -std=c++11
```

#### 2. 環境変数

```bash
# バイナリのフルパスを指定（PATH に入っている場合は不要）
export PRACRAND_PATH=/path/to/RNG_test       # Linux / macOS
set PRACRAND_PATH=C:\tools\pracrand\RNG_test.exe  # Windows
```

---

### TestU01

#### 1. ビルド（静的ライブラリを生成）

```bash
# Linux / macOS
tar xf TestU01-xxx.tar.gz
cd TestU01-xxx
./configure --prefix=$HOME/.local
make && make install
# → $HOME/.local/lib/ に libtestu01.a / libprobdist.a / libmylib.a が生成される
```

```bash
# Windows (MinGW / MSYS2)
./configure --prefix=/c/tools/testu01
make && make install
```

#### 2. 環境変数（ビルド前に設定）

```bash
export TESTU01_LIB_DIR=$HOME/.local/lib          # Linux / macOS
set TESTU01_LIB_DIR=C:\tools\testu01\lib         # Windows
```

> `TESTU01_LIB_DIR` が設定されていない場合は `testu01` モジュール自体がコンパイルされない（ビルドエラーにはならない）。

#### 3. リンクされるライブラリ

`build.rs` が以下を自動的にリンクする:

```
libtestu01.a
libprobdist.a
libmylib.a
```

---

## ビルド

```bash
# PracRand のみ使う場合（テスト品質フラグなし）
cargo build

# TestU01 も使う場合
set TESTU01_LIB_DIR=C:\tools\testu01\lib
cargo build
```

> `testing` モジュールは `debug_assertions` が有効なビルド（`cargo build` / `cargo test`）にのみ含まれる。  
> `cargo build --release` では除外される。

---

## 使い方

### PracRand

```rust
use urng::prelude::*;
use urng::testing::pracrand::{self, DataSize};

// 64-bit RNG を 1 GB 分テスト
let rng = Xoshiro256Pp::new(42);
let result = pracrand::run64(rng, DataSize::GB(1)).unwrap();

result.print();               // RNG_test の出力をそのまま表示
assert!(result.passed);       // FAIL が含まれていなければ true
```

```rust
// 32-bit RNG
let rng = Pcg32::new(42);
let result = pracrand::run32(rng, DataSize::MB(256)).unwrap();
assert!(result.passed);
```

**DataSize バリアント:**

| バリアント | 説明 |
|-----------|------|
| `DataSize::MB(n)` | n メガバイト |
| `DataSize::GB(n)` | n ギガバイト |
| `DataSize::TB(n)` | n テラバイト |

---

### TestU01

```rust
use urng::prelude::*;
use urng::testing::testu01::{self, Battery};

// 64-bit RNG に SmallCrush バッテリーを実行
let rng = Xoshiro256Pp::new(42);
let result = testu01::run64(rng, Battery::SmallCrush);

println!("battery : {}", result.battery);
println!("tests   : {}", result.n_tests);
println!("failed  : {}", result.n_failed);
assert!(result.passed());
```

```rust
// 32-bit RNG
let rng = Pcg32::new(42);
let result = testu01::run32(rng, Battery::SmallCrush);
assert!(result.passed());
```

**Battery バリアント:**

| バリアント | テスト数 | 所要時間（目安） |
|-----------|---------|----------------|
| `Battery::SmallCrush` | 15 | 数秒 |
| `Battery::Crush` | 96 | 数十分 |
| `Battery::BigCrush` | 106 | 数時間 |

> TestU01 の結果は stdout に直接出力される（ライブラリの仕様）。  
> `result.p_values` で各テストの p 値を参照できる（インデックスは 1 始まり）。

---

## テストから呼ぶ例

```rust
// tests/quality.rs
#[test]
fn xoshiro256pp_pracrand_1gb() {
    use urng::prelude::*;
    use urng::testing::pracrand::{self, DataSize};

    let rng = Xoshiro256Pp::new(0);
    let result = pracrand::run64(rng, DataSize::GB(1))
        .expect("RNG_test not found; set PRACRAND_PATH");
    assert!(result.passed, "PracRand FAIL:\n{}", result.output);
}

#[test]
#[cfg(has_testu01)]
fn xoshiro256pp_smallcrush() {
    use urng::prelude::*;
    use urng::testing::testu01::{self, Battery};

    let rng = Xoshiro256Pp::new(0);
    let r = testu01::run64(rng, Battery::SmallCrush);
    assert!(r.passed(), "SmallCrush failed {}/{} tests", r.n_failed, r.n_tests);
}
```

---

## feature フラグへの移行

将来 `testing` を `--features` で公開したい場合、変更は 2 行:

```toml
# Cargo.toml
[features]
testing = []
```

```rust
// src/lib.rs
#[cfg(any(debug_assertions, feature = "testing"))]
pub mod testing;
```

モジュール内部は一切変更不要。

---

## モジュール構成

```
src/testing/
├── mod.rs        — モジュール宣言・ドキュメント
├── stream.rs     — ByteStream32 / ByteStream64 (Read アダプタ)
├── pracrand.rs   — PracRand subprocess ラッパー
└── testu01.rs    — TestU01 FFI バインディング (has_testu01 cfg)
```
