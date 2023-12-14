
use std::vec::Vec;
use serde::{Serialize, Deserialize};
use crate::core::TickerType;




/// A ticker denotes a trackable, and typicaly tradeable, entity as listed by a brokerage
/// Most tickers are just the ticker for a given stock or ETF, but FX ratios and aggregated market indicators 
/// are included as well
#[derive(Serialize, Deserialize)]
pub struct Ticker {
    /// The type of ticker- Security, Indicator, or FX
    pub ttype: TickerType,
    /// The unique string, typically 2-5 characters, that identifies the ticker within an exchange 
    pub ticker: String,
    /// The name of the company or indicator 
    pub name: String,
}

/// The UsageStats struct captures the use of a ticker either as input to predictive models or as
/// (an) output from predictive models. Therefore, many tickers will be associated with a
/// UsageStats struct both for their use as an input and their appearance as an output. 
/// NOTE: The models considered for incorporation into these statistics are only those that show at
/// least some small promise of predictive power during the timeframe of the data to which they
/// were applied. This is approximately 1% of the total model permutations that have been tried.
/// In other words, at least 99% of the models that are trialed are completely useless:
/// This struct summarizes the use of a ticker in the 1% of models that might not be completely
/// useless.
#[derive(Serialize, Deserialize)]
pub struct UsageStats {
    /// The number of models containing this ticker in the specified manner (input vs output)
    count: u16,
    /// The maximum r2 value achieved on training data in any model 
    max_r2_tr: f64,
    /// The maximum r2 value achieved on cross-validation data in any model 
    max_r2_cv: f64,
    /// The maximum score achieved on any model (score is a function of both training and cv
    /// performance, higher is better)
    max_score: f64,
}


/// The TickerDetail struct captures a significant amout of information about a ticker: 
#[derive(Serialize, Deserialize)]
pub struct TickerDetail {
    /// The ticker itself
    pub ticker: Ticker,
    /// average daily volume, typically calculated over the past couple years 
    pub avg_vol: Option<f64>,
    /// A three-character symbol indicating the exchange
    pub exchange: String,
    /// statistics on the useage of this ticker as the input to a selected model
    pub input_stats: Option<UsageStats>,
    /// statistics on the useage of this ticker as the output from a selected model
    pub output_stats: Option<UsageStats>,
}

