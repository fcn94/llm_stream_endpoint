FEATURE := phi-v2

# Enable to define a customized repo
MODEL_REPO :=
# Enable to define a customized repo
TOKENIZER_REPO :=
# Enable to define a customized quantized model
MODEL_FILE :=

ifneq ($(MAKECMDGOALS),clean)

endif

build :
ifeq ($(FEATURE), phi-v2)
	cargo build --release --features phi-v2
else
	cargo build --release --features mistral
endif

run :
	@if [ -n "$(MODEL_REPO)" -a -n "$(MODEL_FILE)" -a -n "$(TOKENIZER_REPO)"  ]; then \
  		echo "Building using Model Repo: $(MODEL_REPO) and Model file: $(MODEL_FILE) and Tokenizer Repo :  $(TOKENIZER_REPO) "; \
		./target/release/llm_stream --quantized --temperature 0.1 --model-id=$(MODEL_REPO) --model-file=$(MODEL_FILE) --tokenizer-id=$(TOKENIZER_REPO); \
    else \
       echo "Building using default Value for $(FEATURE)";\
       ./target/release/llm_stream --quantized --temperature 0.1; \
    fi

clean :
	cargo clean

update :
	cargo update

