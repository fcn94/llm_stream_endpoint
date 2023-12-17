FEATURE := phi-v2

# Check if processor supports AVX2 feature
ifeq ($(shell grep avx2 /proc/cpuinfo | wc -l), 0)
  HAS_AVX2 := no
else
  HAS_AVX2 := yes
endif

# Build rules for AVX2-supported processor
ifeq ($(HAS_AVX2), yes)
  RUSTFLAGS_CHAIN := RUSTFLAGS="-C target-feature=+avx2"
endif

# Build rules for non-AVX2-supported processor
ifeq ($(HAS_AVX2), no)
  RUSTFLAGS_CHAIN :=
endif

build :
ifeq ($(FEATURE), phi-v2)
	$(RUSTFLAGS_CHAIN) cargo build --release --features phi-v2
else
	$(RUSTFLAGS_CHAIN) cargo build --release --features mistral
endif

run :
	./target/release/llm_stream --quantized --temperature 0.1

clean :
	cargo clean

update :
	cargo update

