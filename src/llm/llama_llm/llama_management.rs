use anyhow::{ Result};
use candle::{ Device, Tensor};
use candle_transformers::generation::LogitsProcessor;


use tokenizers::Tokenizer;
use crate::llm::token_output_stream::TokenOutputStream;
use tokio::sync::mpsc::{UnboundedSender};

use crate::llm::quantized_llm::QuantizedTextGeneration;

use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

impl QuantizedTextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model_type:String,
        model_weights: ModelWeights,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {

        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        Self {
            model_type,
            model_weights,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub(crate) fn run(&mut self, prompt: &str, sample_len: usize,seed:u64,temperature:Option<f64>,top_p:Option<f64>, tx:UnboundedSender<String>,context:&str) -> Result<()> {

        self.tokenizer.clear();

        let mut pre_prompt_tokens = vec![];

        // Text Generation Prompt for Mistral
        let prompt=format!("<s>[INST]{}[/INST]",prompt.trim());

        let tokens = self.tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;

        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();

        let to_sample = sample_len.saturating_sub(1);

        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };

        let mut all_tokens = vec![];
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        let start_prompt_processing = std::time::Instant::now();

        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model_weights.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };

        let prompt_dt = start_prompt_processing.elapsed();

        all_tokens.push(next_token);

        if let Some(t) =  self.tokenizer.next_token(next_token)? {
            let _ = tx.send(t.to_string());
        }

        // Retrieve eos token
        let eos_token = get_eos_token(self.model_type.clone()).unwrap();
        let eos_token = *self.tokenizer.tokenizer().get_vocab(true).get(eos_token.as_str()).unwrap();

        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;

        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model_weights.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            if let Some(t) =  self.tokenizer.next_token(next_token)? {
                let _ = tx.send(t.to_string());
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) =  self.tokenizer.decode_rest().map_err(candle::Error::msg)? {
            let _ = tx.send(rest.to_string());
        }

        let dt = start_post_prompt.elapsed();

        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );

        Ok(())

    }

}

fn get_eos_token(which:String) -> Result<String> {

    let eos_token=match which.as_str() {
        "mistral" => "</s>".to_string(),
        "open_chat" => "<|end_of_turn|>".to_string(),
        _ => "</s>".to_string(),
    };

    Ok(eos_token.to_string())
}