# llm_stream_endpoint
This repository is a minimal llm rust api streaming endpoint.

I wanted to do a minimalist service to interact with a LLM, in a streaming mode

I tested websockets and server sent events and bot were overkills

I finally opted for a simple HTTP streaming endpoint.

This repository describes it in its simplistic form.

It is a very simple Rest Streaming Endpoint using :
* Rust
* Warp
* Candle

It does not cover GPU, only CPU ( eventhough GPU adjustments should be straightforwards)
It uses mistral or phi-2


# How to use the service


> The selection of the model is activated by a feature, either mistral or phi-2 ( by default) 
>
> A Makefile facilitates clean,update, build,run
> 
> Prior to execution , please run :
>
> *make clean*
> 
> and then 
> 
> *make update*

\
\
To build the service , just type
> With Phi-2 , type :
> 
> *make build*
> 
> or
> 
> With mistral, type :
> 
> *make FEATURE=mistral build*

\
\
Then, to run it, 
> *make run*

\
\
And get these logs at launch, in my x86
> avx: true, neon: false, simd128: false, f16c: false
>
> temp: 0.10 repeat-penalty: 1.10 repeat-last-n: 64
>
> retrieved the files in 102.103µs
> loaded the model in 558.235206ms

\
\
Once launched, to use the API, you can
> * From a linux terminal, use curl
>  * curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"Where is located Paris ?"}'
> * From a browser, a very simple UI is available at :
>  * http://127.0.0.1:3030/

\
\
# New : You can now specify a custom model
Provided these models are compatible with phi-2 or mistral , you can specify your own huggingface repo and quantized file
\
> You can type
> 
> make run MODEL_REPO="YOUR CUSTOM HUGGINGFACE REPO" MODEL_FILE="YOUR QUANTIZED FILE"

This is useful should you be willing to run a fine tuned version of either phi-2 or mistral

\
\
# References
* This is heavily inspired by one of the example from candle repository
https://github.com/huggingface/candle/tree/main/candle-examples/examples/mistral
