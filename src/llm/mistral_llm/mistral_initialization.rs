#![feature(const_trait_impl)]

use anyhow::{Error as E, Result};

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use crate::args_init::args::Args;
use crate::llm::device::device;
use crate::llm::llm::{LLM, LlmPackage};


#[derive(Debug, Clone)]
pub enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}


pub struct LlmModel;

impl LLM for LlmModel {
    fn initialize(&self, args_init: Args) -> Result<LlmPackage> {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let _guard = if args_init.tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            args_init.temperature,
            args_init.repeat_penalty,
            args_init.repeat_last_n
        );

        let start = std::time::Instant::now();
        let api = Api::new()?;


        let repo = api.repo(Repo::with_revision(
            args_init.model_id,
            RepoType::Model,
            args_init.revision,
        ));

        let tokenizer_filename = repo.get(args_init.tokenizer_file.as_str())?;

        let filenames = match args_init.weight_files {
            Some(files) => files
                .split(',')
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>(),
            None => {
                if args_init.quantized {
                    vec![repo.get("model-q4k.gguf")?]
                    //vec![repo.get("arithmo-mistral_llm-7b.Q4_K_S.gguf")?]
                } else {
                    vec![
                        repo.get("pytorch_model-00001-of-00002.safetensors")?,
                        repo.get("pytorch_model-00002-of-00002.safetensors")?,
                    ]
                }
            }
        };

        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let start = std::time::Instant::now();
        let config = Config::config_7b_v0_1(args_init.use_flash_attn);
        let (model, device) = if args_init.quantized {
            let filename = &filenames[0];
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
            let model = QMistral::new(&config, vb)?;
            (Model::Quantized(model), Device::Cpu)
        } else {
            let device = device(args_init.cpu)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            let model = Mistral::new(&config, vb)?;
            (Model::Mistral(model), device)
        };

        println!("loaded the model in {:?}", start.elapsed());

        Ok(LlmPackage {
            model,
            device,
            tokenizer,
            seed: args_init.seed,
            temperature: args_init.temperature,
            top_p: args_init.top_p,
            repeat_penalty: args_init.repeat_penalty,
            repeat_last_n: args_init.repeat_last_n,
            sample_len: args_init.sample_len,
        })
    }


}

