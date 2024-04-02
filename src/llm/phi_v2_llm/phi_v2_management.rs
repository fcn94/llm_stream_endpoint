use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use tokenizers::Tokenizer;
use tokio::sync::mpsc::{UnboundedSender};
use crate::llm::llm::TextGeneration;


use crate::llm::phi_v2_llm::phi_v2_initialization::{ Model};



impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Model,
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
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub(crate) fn run(&mut self, prompt: &str, sample_len: usize, tx:UnboundedSender<String>,context:&str) -> Result<()> {


        // Text Generation Prompt for phi-2
        let prompt=format!("Context:{}.\nInstruct: {}.\nOutput:",context.trim(),prompt.trim());


        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();



        let mut generated_tokens = 0usize;



        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };


        let start_gen = std::time::Instant::now();


        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Quantized(m) => m.forward(&input)?,
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;


            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token {
                break;
            }


            if let t = self.tokenizer.decode(&[next_token], true).map_err(E::msg)? {
                let _ = tx.send(t.to_string());
            }



        }

            let dt = start_gen.elapsed();

            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );

            Ok(())

    }

}
