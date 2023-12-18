//! The primary purpose of this crate is to introduce the FromAPI trait,
//! which allows a struct to be obtained via an http API call to fathom.sextant-ai.com
//!
//!

use std::env;//string::ToString;
use serde::de::DeserializeOwned;
use hyperactive::{client, err::{HypErr, ApiKeyError}};

const BASE_URL: &'static str = "http://127.0.0.1:14080/api";
pub const CHRONO_FMT: &'static str = "%Y-%m-%d";


// return the value of the environment variable FATHOM_KEY
fn get_api_key() -> Result<String, ApiKeyError> {
    match env::var("FATHOM_KEY") {
        Ok(key) => Ok(key),
        Err(_) => Err(ApiKeyError::MissingEnv("FATHOM_KEY".to_string())),
    }
}


pub trait DataType {
    fn data_type() -> &'static str;
}

pub fn url_pk(data_type: &str) -> String {
    format!("{}/pk?data_type={}", BASE_URL, &data_type)
}


pub async fn call_api<T: DeserializeOwned>(url: &str) -> Result<T, HypErr> {
    let api_key = get_api_key()?;
    let t: T = client::get(url, Some(&api_key)).await?;
    Ok(t)
}