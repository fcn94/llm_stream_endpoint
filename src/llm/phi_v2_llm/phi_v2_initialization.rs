#![feature(const_trait_impl)]

use std::path::PathBuf;
use anyhow::{Error as E, Result};


use candle_transformers::models::mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle::{Device};
use hf_hub::{api::sync::Api, Repo, RepoType};
use hf_hub::api::sync::ApiRepo;
use tokenizers::Tokenizer;
use crate::args_init::args::Args;
use crate::llm::device::device;
use crate::llm::llm::{LLM, LlmPackage};


#[derive(Debug, Clone)]
pub enum Model {
    Quantized(QMixFormer),
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

        let model_filenames = get_filenames_model(&repo_model, args_init.weight_files, args_init.model_file)?;

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


        let config = Config::v2();

        // We will only process quantized models
        let (model, device_model) = {
            let device_model = device(false)?;
            let filename = &model_filenames[0];
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename,&device_model)?;
            let model = QMixFormer::new_v2(&config, vb)?;

            (Model::Quantized(model), device_model)
        };


        /**********************************************************************/
        // End Construction LLM Package
        /**********************************************************************/



        println!("loaded the model in {:?}", start.elapsed());

        Ok(LlmPackage {
            model,
            device:device_model,
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

fn get_filenames_model(repo:&ApiRepo, weight_files:Option<String>, model_file:Option<String>) -> Result<Vec<PathBuf>> {
    Ok(match weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
                vec![repo.get(model_file.unwrap().as_str())?]
            }
    })
}