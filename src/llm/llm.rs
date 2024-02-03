use anyhow::{Error as E, Result};
use candle::Device;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use crate::llm::token_output_stream::TokenOutputStream;
use crate::args_init::args::Args;

// todo: to be put under feature to spot right LLM
#[cfg(feature = "mistral")]
use crate::llm::mistral_llm::mistral_initialization::Model;

#[cfg(feature = "phi-v2")]
use crate::llm::phi_v2_llm::phi_v2_initialization::Model;


#[derive( Debug,Clone)]
pub struct LlmPackage {
    pub model:Model,
    pub device:Device,
    pub tokenizer:Tokenizer,
    pub seed:u64,
    pub temperature:f64,
    pub top_p:f64,
    pub repeat_penalty:f32,
    pub repeat_last_n:usize,
    pub sample_len:usize,
}


pub struct TextGeneration {
    pub model: Model,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

pub trait LLM {
    fn initialize(&self,args_init: Args) ->  Result<LlmPackage> ;
}


pub fn generate( llm_package:LlmPackage,prompt:&str,tx:UnboundedSender<String>,context:&str) -> Result<()> {
    let mut pipeline = TextGeneration::new(
        llm_package.model,
        llm_package.tokenizer,
        llm_package.seed,
        Some(llm_package.temperature),
        Some(llm_package.top_p),
        llm_package.repeat_penalty,
        llm_package.repeat_last_n,
        &llm_package.device,
    );
    pipeline.run(prompt, llm_package.sample_len,tx,context)?;
    Ok(())
}