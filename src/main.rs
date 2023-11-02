#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::convert::Infallible;
use warp::{Filter};

use tokio_stream::wrappers::{UnboundedReceiverStream};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{ UnboundedReceiver, UnboundedSender};

use std::{fs, thread};

use bytes::{Bytes};

use futures_util::{Stream, StreamExt};
use hyper::Body;
use serde::{Deserialize, Serialize};


// declared as pub
use llm_stream::llm::llm_initialization::{llm_initialize};
use llm_stream::llm::llm_mgt::generate;


pub const SAMPLE_LEN:usize = 200;
const PROMPT: &str = "Where is Mona Lisa ?";

#[derive(Serialize, Deserialize, Debug,Clone)]
pub struct Prompt {
    pub query: String,
}


#[tokio::main]
async fn main() ->anyhow::Result<()> {


    pretty_env_logger::init();

    /**************************************************************/
    // Initialization llm model
    /**************************************************************/

    let (model,device,tokenizer)=llm_initialize().unwrap();

    // retrieves root html page
    let index_text= fs::read_to_string("./site/index.html")?;


    // API route
    let routes = warp::path("token_stream")
        .and(warp::post())
        .and(prompt_json_body())
        .map( move |prompt :Prompt| {

        let (tx, rx):(UnboundedSender<String>,UnboundedReceiver<String>)  = mpsc::unbounded_channel();

        let rx_stream = UnboundedReceiverStream::new(rx);

            let llm_triple =(model.clone(), device.clone(), tokenizer.clone());

            let _handler = thread::spawn(move || {
                let _ = generate(llm_triple.0, llm_triple.1, llm_triple.2, prompt.query.as_str(), SAMPLE_LEN,tx);
            });

         let event_stream = rx_stream.map(move |token| {
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

