#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::convert::Infallible;
use warp::{Filter};

use tokio_stream::wrappers::{UnboundedReceiverStream};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{ UnboundedReceiver, UnboundedSender};

use std::{fs, thread};
use std::sync::{Arc, Mutex};

use bytes::{Bytes};

use futures_util::{Stream, StreamExt};
use hyper::Body;
use serde::{Deserialize, Serialize};


// declared as pub
use llm_stream::llm::llm_initialization::{llm_initialize, Llm_Package};
use llm_stream::llm::llm_mgt::generate;


pub const SAMPLE_LEN:usize = 200;

const NB_WORKERS:usize = 2;

const PROMPT: &str = "Where is Mona Lisa ?";

#[derive(Serialize, Deserialize, Debug,Clone)]
pub struct Prompt {
    pub query: String,
}


#[tokio::main]
async fn main() ->anyhow::Result<()> {


    pretty_env_logger::init();

    /**************************************************************/
    // Create a new Tokio runtime on a dedicated thread
    /**************************************************************/
    let dedicated_runtime = Arc::new(Mutex::new(None));

    // Create a dedicated Tokio runtime on a separate thread
    let dedicated_thread_handle = tokio::task::spawn_blocking({
        let dedicated_runtime = dedicated_runtime.clone();
        move || {
            *dedicated_runtime.lock().unwrap() = Some(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(NB_WORKERS)
                    .enable_all()
                    .build()
                    .expect("Failed to create dedicated runtime"),
            );
        }
    });

    // Get a handle of this dedicated thread
    dedicated_thread_handle.await.expect("Failed to spawn dedicated thread");

    /**************************************************************/
    // Initialization Chain
    /**************************************************************/
    // Todo : Ensure initialization by command line
    // Todo: Ensure connection to a specific gguf tokenizer/model

    /**************************************************************/
    // Initialization llm model
    /**************************************************************/

    // Retrieve Model, Device, Tokenizer
    let llm_package=llm_initialize().unwrap();

    // retrieves root html page
    let index_text= fs::read_to_string("./site/index.html")?;

    // API route
    let routes = warp::path("token_stream")
        .and(warp::post())
        .and(prompt_json_body())
        .map( move |prompt :Prompt| {

            // Create a new channel for each request
            let (tx, rx):(UnboundedSender<String>,UnboundedReceiver<String>)  = mpsc::unbounded_channel();
            let rx_stream = UnboundedReceiverStream::new(rx);

            let llm_package_clone=llm_package.clone();

            // Clone the Arc for the closure
            let dedicated_runtime_clone = dedicated_runtime.clone();

            // Spawn a Tokio task in the dedicated runtime for this specific channel
            let _ = dedicated_runtime_clone.lock().unwrap().as_ref().unwrap().spawn(async move {
                process_generation(llm_package_clone, prompt.query, tx).await;
            });

            let event_stream = rx_stream.map(  move |token| {
                Ok(Bytes::from(token))
            });

            event_stream

    })
        .then(handler_stream);


    // Route to retrieve the html page
    let routes_index=warp::get().map(move || warp::reply::html(index_text.clone()));

    warp::serve(routes.or(routes_index)).run(([127, 0, 0, 1], 3030)).await;

    Ok(())
}

/*****************************************************************/
// route handlers
/*****************************************************************/

async fn handler_stream(
    body: impl Stream<Item = Result< Bytes, Infallible>> + Unpin + Send + Sync + 'static,
) -> Result<hyper::Response<Body>, Infallible> {
    let body= hyper::Body::wrap_stream(body);
    Ok(warp::reply::Response::new(body))
}


fn prompt_json_body() -> impl Filter<Extract = (Prompt,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(1024 * 16)
        .and(warp::body::json())
}


async fn process_generation(llm_package:Llm_Package,prompt:String,tx: UnboundedSender<String>) {
    let _ = generate(llm_package.model, llm_package.device, llm_package.tokenizer, prompt.as_str(), SAMPLE_LEN,tx);
}