
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

pub enum StockType {
    /// A stock or ETF
    Stock,
    /// A market indicator 
    Indicator,
    /// A Foreign eXchange rate
    FX,
}

pub struct Stock {
    stype: StockType,
    symbol: String,
    name: String,
}

pub struct PriceHistory {
    stock: Stock, 
    history: Vec<DailyChange>,
}


