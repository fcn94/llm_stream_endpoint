#![feature(const_trait_impl)]

use std::string::ToString;
use anyhow::{Error as E, Result};

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use crate::llm::device::device;


const CPU:bool = true;
const TRACING:bool = false;
const USE_FLASH_ATTENTION:bool = true;
pub const TEMPERATURE:Option<f64> = Some(0.2);
pub const TOP_P:Option<f64> = Some(0.3);
pub const SEED:u64 = 299792458;

const MODEL_ID: &str = "lmz/candle-mistral";

const REVISION: &str= "main";
const TOKENIZER_FILE: &str = "tokenizer.json";
const WEIGHTS_FILE:Option< &str> = None;
const QUANTIZED:bool = true;
pub const REPEAT_PENALTY:f32 = 1.1;
pub const REPEAT_LAST_N:usize = 64;

#[derive(Debug, Clone)]
pub enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}

#[derive( Debug,Clone)]
pub struct Llm_Package {
    pub model:Model,
    pub device:Device,
    pub tokenizer:Tokenizer,
}

pub fn llm_initialize() ->  Result<Llm_Package>  {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if TRACING {
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
        TEMPERATURE.unwrap_or(0.),
        REPEAT_PENALTY,
        REPEAT_LAST_N
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;


    let repo = api.repo(Repo::with_revision(
        MODEL_ID.to_string(),
        RepoType::Model,
        REVISION.to_string(),
    ));

    let tokenizer_filename = repo.get(TOKENIZER_FILE)?;

    let filenames = match WEIGHTS_FILE {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if QUANTIZED {
                vec![repo.get("model-q4k.gguf")?]
                //vec![repo.get("arithmo-mistral-7b.Q4_K_S.gguf")?]
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
    let config = Config::config_7b_v0_1(USE_FLASH_ATTENTION);
    let (model, device) = if QUANTIZED {
        let filename = &filenames[0];
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
        let model = QMistral::new(&config, vb)?;
        (Model::Quantized(model), Device::Cpu)
    } else {
        let device = device(CPU)?;
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

    Ok(Llm_Package{
        model,
        device,
        tokenizer,
    })
}