//! *"Abstractions are sometimes more real than the things they are used to describe."* - Jordan
//! Peterson
//!
//! This module is concerned with abstractions about the market. From the standpoint of a trader,
//! these are things that are perhaps more real, or at least more meaningful, than the price action
//! of an individual stock. A *"rising tide lifting all boats"* (to steal an abstraction form
//! Warren Buffett) is a very simple example of an abstraction. A more nuanced abstraction is
//! grouping stocks into sectors to aid in asset allocation- i.e. selecting allocation among those
//! sectors is the first and more important question, and selection within them is secondary.
//!
//! In the context of the fathom_sdk, abstractions come in two main flavors:
//! 
//! **Clustering**- this is the grouping of securities that tend to move together. This is similar
//! to sectors, but rather more granular. The universe of ~4,300 tradeable securities are broken
//! into 256 correlated clusters of ~17 securities each, and those 256 clusters are in turn
//! clustered into 16 *"megaclusters"* of ~16 clusters each. 
//!
//! **Principal Component Analysis** - This is the breakdown of price action across all securities
//! (or a set of them) into a ranked set of orthogonal factors. What do these represent? That is a
//! harder question. One of the top factors has to look a lot like the "tide" that Buffett
//! described. The strength of the dollar is probably near the top too. Further down there is probably a
//! component in there which correlates to geopolitical tension, and another one someplace for how good
//! (or bad) the weather conditions have been for harvest in each continent. It would take some
//! effort to apply these labels to the half of them, but they come out naturally as eigenvectors
//! in the PCA analysis of stock market movement. It is likely that these are the most meaningful
//! market indicators of all, as they represent similar hopes and fears ruminating in the minds of
//! millions of traders and investors that drive them to similar action with similar securities.
//! Remember, the abstraction can be more real than the thing it describes.
//!
//! This module contains structs that capture these abstractions so they can be investigated and,
//! more importantly, used as inputs into machine learning models to predict future price action.

use vec::Vec;
use serde::{Serialize, Deserialize};
use chrono::NaiveDate;


#[derive(Serialize, Deserialize)]
pub struct TimeFrame {
    /// This represents the last trading day used to calculate an abstraction
    pub end_date: NaiveDate, 
    /// The lenght of time preceeding the end_date over which price action is considered 
    pub years_history: i16,
}


#[derive(Serialize, Deserialize)]
pub struc AbstractionContext {
    /// a unique CHAR(16) string identifier for this abstraction context 
    pub abst_ctx_id: String,
    /// A name describing the intended scope of this abstraction (i.e. S&P 500 members 2019-2023)
    pub descrip: String,
    /// The timeframe for this abstraction
    pub timeframe: TimeFrame,
    /// a list of input stocks used in this abstraction context - i.e. those to be clustered and
    /// used for Principal Component Analysis 
    pub symbols: Vec<String>,
}

