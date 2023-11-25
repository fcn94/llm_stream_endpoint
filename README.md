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
It uses mistral or phi-puffin


# How to use the service

>*** IMPORTANT CHANGE ***
>
>***The selection is activated by a feature, either mistral or puffin***


To run it, just type
> cargo run --release --features mistral
> 
> or
> 
> cargo run --release --features puffin

Using RUSTFLAGS to activate some features like AVX2
> RUSTFLAGS="-C target-feature=+avx2" cargo run --release --features puffin  -- --temperature 0.1

And get these logs at launch, in my case
> avx: true, neon: false, simd128: false, f16c: false
>
> temp: 0.10 repeat-penalty: 1.10 repeat-last-n: 64
>
> retrieved the files in 102.103µs
> loaded the model in 558.235206ms


Once launched, to use the API, you can
> * From a linux terminal, use curl
>  * curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"Where is located Paris ?"}'
> * From a browser, a very simple UI is available at :
>  * http://127.0.0.1:3030/

  
# What I am looking for

I wanted to do a minimalistic service to interact with an LLM in a streaming mode

I am eager to get feedbacks, about potential technical mistakes I may have gone through, and any potential suggestion of improvements


# ROADMAP
I am planning to have a service that would make it easier to connect to different models

I will enable injection of a system prompt, and proper formatting of prompt depending on the model


# References
* This is heavily inspired by one of the example from candle repository
https://github.com/huggingface/candle/tree/main/candle-examples/examples/mistral
