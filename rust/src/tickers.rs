
use std::vec::Vec;
use serde::{Serialize, Deserialize};
use chrono::NaiveDate;

#[derive(Serialize, Deserialize)]
pub struct DailyChange {
    /// The date 
    pub date: NaiveDate,
    /// the closing price for the day 
    pub close: f64,
    /// the percentage change from the previous day 
    pub pct_change: f64,
    /// The sigma level of the change 
    pub sigma: f64,
}

/// The TickerType enum captures that most tickers represent a stock/security,
/// But some are a market indicator (i.e. index) and some are a FX ratio 
pub enum TickerType {
    /// A stock or ETF
    Stock,
    /// A market indicator 
    Indicator,
    /// A Foreign eXchange rate
    FX,
}

/// A ticker is a symbol, typically two to five characters, indicating a security that can be
/// traded or index that can be referenced via most brokerage / trading platforms 
pub struct Ticker {
    /// The type of ticker
    pub ticker_type: TickerType,
    /// The unique symbol identifying the ticker 
    pub symbol: String,
    /// The name of the stock / indicator / FX 
    pub name: String,
}

/// The price history variable captures the daily change in price levels for a ticker
pub struct PriceHistory {
    /// The ticker in question
    pub ticker: Ticker, 
    /// A vector of DailyChange price movements over some period 
    pub history: Vec<DailyChange>,
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
    score: f64,
}


/// The TickerDetail struct captures a significant amout of information about a ticker: 
pub mod TickerDetail {

