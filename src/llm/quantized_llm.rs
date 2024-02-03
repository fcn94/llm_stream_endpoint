use anyhow::{Error as E, Result};
use candle::Device;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use crate::llm::token_output_stream::TokenOutputStream;
use crate::args_init::args::Args;


use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

#[derive( Debug,Clone)]
pub struct QuantizedLlmPackage {
    pub model_type:String,
    pub model_weights:ModelWeights,
    pub device:Device,
    pub tokenizer:Tokenizer,
    pub seed:u64,
    pub temperature:f64,
    pub top_p:f64,
    pub repeat_penalty:f32,
    pub repeat_last_n:usize,
    pub sample_len:usize,
}

pub struct QuantizedTextGeneration {
    pub model_type:String,
    pub model_weights: ModelWeights,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}


pub trait QuantizedLLM {
    fn initialize(&self,args_init: Args) ->  Result<QuantizedLlmPackage> ;
}


pub fn generate( quantized_llm_package:QuantizedLlmPackage,prompt:&str,tx:UnboundedSender<String>,context:&str) -> Result<()> {
    let mut pipeline = QuantizedTextGeneration::new(
        quantized_llm_package.model_type,
        quantized_llm_package.model_weights,
        quantized_llm_package.tokenizer,
        quantized_llm_package.seed,
        Some(quantized_llm_package.temperature),
        Some(quantized_llm_package.top_p),
        quantized_llm_package.repeat_penalty,
        quantized_llm_package.repeat_last_n,
        &quantized_llm_package.device,
    );
    pipeline.run(prompt, quantized_llm_package.sample_len,quantized_llm_package.seed,Some(quantized_llm_package.temperature),Some(quantized_llm_package.top_p),tx,context)?;
    Ok(())
}