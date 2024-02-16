#![feature(const_trait_impl)]

use std::path::PathBuf;
use anyhow::{Error as E, Result};


use candle::{Device};
use candle::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};
use hf_hub::api::sync::ApiRepo;
use tokenizers::Tokenizer;
use crate::args_init::args::Args;
use crate::llm::device::device;
use crate::llm::quantized_llm::{QuantizedLLM, QuantizedLlmPackage};





pub struct QuantizedLlmModel;

impl QuantizedLLM for QuantizedLlmModel {
    fn initialize(&self, args_init: Args) -> Result<QuantizedLlmPackage> {

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

        let model_filenames = get_filenames_model(&repo_model, args_init.local_model_file, args_init.model_file)?;

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


        let model_path = model_filenames[0].clone();


        let mut file = std::fs::File::open(&model_path)?;

        let gguf_model_content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in gguf_model_content.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }

        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            gguf_model_content.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );

        // CPU
        let device = device(true)?;

        let (model_weights, device) = (ModelWeights::from_gguf(gguf_model_content, &mut file, &device)?, Device::Cpu);


        /**********************************************************************/
        // End Construction LLM Package
        /**********************************************************************/

        println!("loaded the model in {:?}", start.elapsed());

        Ok(QuantizedLlmPackage {
            model_type:args_init.model_type,
            model_weights,
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

fn get_filenames_model(repo:&ApiRepo, weight_files:Option<String>,model_file:Option<String>) -> Result<Vec<PathBuf>> {
    Ok( vec![repo.get(model_file.unwrap().as_str())?])
}



fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}