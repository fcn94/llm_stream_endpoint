pub mod device;
pub mod token_output_stream;



#[cfg(not(feature = "llama"))]
pub mod llm;

#[cfg(feature = "mistral")]
pub mod mistral_llm;

#[cfg(feature = "phi-v2")]
pub mod phi_v2_llm;

#[cfg(feature = "llama")]
pub mod llama_llm;
#[cfg(feature = "llama")]
pub mod quantized_llm;