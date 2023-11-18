use clap::{ Parser};


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long,default_value_t=true)]
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long,default_value_t=false)]
    pub tracing: bool,

    #[arg(long,default_value_t=true)]
    pub use_flash_attn: bool,

    /// The temperature used to generate samples.
    #[arg(long,default_value_t=0.2)]
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long,default_value_t=0.3)]
    pub top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 200)]
    pub sample_len: usize,

    #[cfg(feature = "mistral")]
    #[arg(long, default_value = "lmz/candle-mistral")]
    pub model_id: String,

    #[cfg(feature = "puffin")]
    #[arg(long, default_value = "lmz/candle-quantized-phi")]
    pub model_id: String,

    #[arg(long, default_value = "main")]
    pub revision: String,

    #[arg(long)]
    pub model_file: Option<String>,
    
    #[arg(long,default_value="tokenizer.json")]
    pub tokenizer_file: String,

    #[arg(long)]
    pub weight_files: Option<String>,

    #[arg(long,default_value_t=true)]
    pub quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
}

impl Args {
    #[allow(clippy::too_many_arguments)]
    pub fn new() -> Self {
        let args_init = Args::parse();
        args_init
    }
}