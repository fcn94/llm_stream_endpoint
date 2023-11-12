use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 200)]
    sample_len: usize,

    // todo : should we manage a path of a gguf file, or via a feature
    #[arg(long, default_value = "lmz/candle-mistral")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

impl Args {
    #[allow(clippy::too_many_arguments)]
    pub fn new() -> Self {

        let args = Args::parse();

        Self {
            cpu: false,
            tracing: false,
            use_flash_attn: false,
            //prompt: "".to_string(),
            temperature: None,
            top_p: None,
            seed: 0,
            sample_len: 0,
            model_id: "".to_string(),
            revision: "".to_string(),
            tokenizer_file: None,
            weight_files: None,
            quantized: false,
            repeat_penalty: 0.0,
            repeat_last_n: 0,
        }
    }

}