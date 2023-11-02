# llm_stream_endpoint
I wanted to do a minimalist service to interact with a LLM, in a streaming mode

I tested websockets and server sent events and bot were overkills

I finally opted for a simple HTTP streaming endpoint.

This repository describes it in its simplistic form.

It is a very simple Rest Streaming Endpoint using :
* Rust
* Warp
* Candle

It does not cover GPU, only CPU ( eventhough GPU adjustments should be straightforwards)
It uses a quantized version of Mistral 

# How to use the service

To run it, just type
> cargo run --release

Once launched, to use the API, you can
* From a linux terminal, use curl
  * curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"Where is located Paris ?"}'
* From a browser, a very simple UI is available at :
  * http://127.0.0.1:3030/

# What I am looking for

I wanted to do a minimalistic service to interact with an LLM in a streaming mode

I am eager to get feedbacks, about potential technical mistakes I may have gone through, and any potential suggestion of improvements


# ROADMAP
I am planning to have a service that would make it easier to connect to different models




# References
* This is heavily inspired by one of the example from candle repository
https://github.com/huggingface/candle/tree/main/candle-examples/examples/mistral
