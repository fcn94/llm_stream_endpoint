FEATURE := phi-v2

# Enable to define a customized repo
MODEL_REPO :=
# Enable to define a customized repo
TOKENIZER_REPO :=
# Enable to define a customized quantized_llm model
MODEL_FILE :=
# Enable to define a customized quantized_llm model
WEIGHT_FILES :=
# Enable to select one of the context type : general, classifier, sql, math
CONTEXT_TYPE := general

ifneq ($(MAKECMDGOALS),clean)

endif


# Build quantized llm , in gguf format , either for phi-2 ( default), or mistral, or llama
# Majority of quantized in huggingface are using llama2 formalism
build :
ifeq ($(FEATURE), phi-v2)
	echo "Building using default Value for $(FEATURE)";\
	cargo build --release --features phi-v2
else ifeq ($(FEATURE), mistral)
	echo "Building using default Value for $(FEATURE)";\
	cargo build --release --features mistral
else ifeq ($(FEATURE), llama)
	echo "Building using default Value for $(FEATURE)";\
    cargo build --release --features llama
endif


# run based on files from huggingface
run :
	@if [ -n "$(MODEL_REPO)" -a -n "$(MODEL_FILE)" -a -n "$(TOKENIZER_REPO)"  ]; then \
  		echo "Running using Model Repo: $(MODEL_REPO) and Model file: $(MODEL_FILE) and Tokenizer Repo :  $(TOKENIZER_REPO) "; \
		./target/release/llm_stream  --temperature 0.1 --model-id=$(MODEL_REPO) --model-file=$(MODEL_FILE) --tokenizer-id=$(TOKENIZER_REPO) --context-type=$(CONTEXT_TYPE); \
    else \
       echo "Running using default values";\
       ./target/release/llm_stream  --temperature 0.1 --context-type=$(CONTEXT_TYPE); \
    fi

# run based on a model downaloaded locally
run_local :
	@if [ -n "$(WEIGHT_FILES)"   ]; then \
  		echo "Building using Model Local File: $(WEIGHT_FILEs) "; \
		./target/release/llm_stream  --temperature 0.1 --weight-files=$(WEIGHT_FILES) --context-type=$(CONTEXT_TYPE); \
    else \
       echo "Building using default values";\
       ./target/release/llm_stream  --temperature 0.1 --context-type=$(CONTEXT_TYPE); \
    fi


clean :
	cargo clean

update :
	cargo update

