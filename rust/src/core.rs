//! Information which is needed across other models, where inclusion in one might make for circular imports, 
//! is included in the core module.
//! For instance, SymbolMetaData is provided when returning autocomplete results for tickers and abstractions,
//! but trying to include it in just one could lead to circular references 
//! 
//! 

use serde::{Serialize, Deserialize};
use chrono::NaiveDate;


/// A ticker is a symbol, typically two to five characters, indicating a security that can be
/// traded or index that can be referenced via most brokerage / trading platforms 
/// The TickerType enum captures that most tickers represent a stock/security,
/// But some are a market indicator (i.e. index) and some are a FX ratio 
#[derive(Serialize, Deserialize)]
pub enum TickerType {
    /// A stock or ETF
    Stock,
    /// A market indicator 
    Indicator,
    /// A Foreign eXchange rate
    FX,
}

impl TickerType {
    pub fn from_str(s: &str) -> Self {
        match s.as_ref() {
            "stock" => TickerType::Stock,
            "indicator" => TickerType::Indicator,
            "fx" => TickerType::FX,
            _ => TickerType::Stock, // this is the overwhelming majority
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum AbstractionType {
    SubCentroid,
    Centroid,
    MegaCentroid,
    EigenVector,
}

#[derive(Serialize, Deserialize)]
pub enum SymbolType {
    Real(TickerType),
    Abstract(AbstractionType),
}


#[derive(Serialize, Deserialize)]
pub struct Symbol {
    /// The type of ticker
    pub stype: SymbolType,
    /// The unique ticker (for a real ticker) or symbol (for an abstraction)
    pub symbol: String,
    /// The name of the stock / indicator / FX / abstraction
    pub name: String,
}

/// this struct is provided as metadata when returning autocomplete results for tickers and abstractions

#[derive(Serialize, Deserialize)]
pub struct SymbolMetaData {
    /// The symbol itself
    pub symbol: Symbol,
    /// The count of selected models (i.e. those with some hint of predictive power) where this symbol is used as an input
    pub input_count: i64,
    /// The count of selected models (i.e. those with some hint of predictive power) where this symbol is used as an output
    pub output_count: i64,
}


/// The TimeFrame struct captures the fact that you typically want to analye price movement 
/// (including PCA and clustering analyses) over a given timeframe
#[derive(Serialize, Deserialize)]
pub struct TimeFrame {
    /// This represents the last trading day used to calculate an abstraction
    pub end_date: NaiveDate, 
    /// The lenght of time preceeding the end_date over which price action is considered 
    pub years_history: i16,
}


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



/// The price history variable captures the daily change in price levels for a ticker/symbol
#[derive(Serialize, Deserialize)]
pub struct PriceHistory {
    /// The symbol in question
    pub symbol: Symbol, 
    /// The TimeFrame for which price action is being provided
    pub time_config: TimeFrame,
    /// A vector of DailyChange price movements over some period 
    pub history: Vec<DailyChange>,
}