#![feature(const_trait_impl)]

use std::path::PathBuf;
use anyhow::{Error as E, Result};

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use hf_hub::api::sync::ApiRepo;
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

        /**********************************************************************/
        // Tracing Initialization
        /**********************************************************************/

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

        /**********************************************************************/
        // End Initialization
        /**********************************************************************/

        /**********************************************************************/
        // Retrieve Model Files and Tokenizer
        /**********************************************************************/
        let start = std::time::Instant::now();

        let api = Api::new()?;

        let repo_model = api.repo(Repo::with_revision(
            args_init.model_id,
            RepoType::Model,
            args_init.revision.clone(),
        ));

        let model_filenames = get_filenames_model(&repo_model, args_init.weight_files, args_init.quantized, args_init.model_file)?;

        let repo_tokenizer = api.repo(Repo::with_revision(
            args_init.tokenizer_id,
            RepoType::Model,
            args_init.revision,
        ));

        let tokenizer_filename = repo_tokenizer.get(args_init.tokenizer_file.as_str())?;

        println!("retrieved the files in {:?}", start.elapsed());

        /**********************************************************************/
        // End Retrieval Model Files and Tokenizer Files
        /**********************************************************************/

        /**********************************************************************/
        // Construction LLM Package
        /**********************************************************************/

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let start = std::time::Instant::now();
        let config = Config::config_7b_v0_1(args_init.use_flash_attn);
        let (model, device) = if args_init.quantized {
            let filename = &model_filenames[0];
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
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_filenames, dtype, &device)? };
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

fn get_filenames_model(repo:&ApiRepo, weight_files:Option<String>, quantized:bool,model_file:Option<String>) -> Result<Vec<PathBuf>> {
    Ok(match weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if quantized {
                vec![repo.get(model_file.unwrap().as_str())?]
            } else {
                vec![
                    repo.get("pytorch_model-00001-of-00002.safetensors")?,
                    repo.get("pytorch_model-00002-of-00002.safetensors")?,
                ]
            }
        }
    })
}

