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

use std::vec::Vec;
use serde::{Serialize, Deserialize};
use chrono::NaiveDate; 
use crate::core::TimeFrame;


/// Abstraction, be it a clustering centroid or principal component of eigenvector decomposition, 
/// The AbstractRow struct gives the raw output value on a given day, its normalized sigma value, 
/// The weight fraction of input tickers upon which it was originally based that were available as of the 
/// specified date to calculate a new value 
#[derive(Serialize, Deserialize, Debug)]
pub struct AbstractRow {
    pub date: NaiveDate,
    pub raw: f32,
    pub sigma: f32,
    pub weight_frac: f32
}

/// This struct is designed to capture the price history of an abstraction, in one row
#[derive(Serialize, Deserialize, Debug)]
pub struct AbstractPriceHistory {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate, 
    pub history: Vec<AbstractRow>,
}

/// When a ticker is used as an input within an abstraction context,
/// The index in which it appears as an input matrix is recorded for PCA purposes
/// The index will be unique within an AbstractionContext and run
/// from 0,1,2,...,n 
#[derive(Serialize, Deserialize)]
pub struct AbstractionInput {
    /// The ticker for a stock/indicator/fx used as an input to the abstraction
    pub ticker: String,
    /// The index in which the term appears in an input matrix for PCA purposes
    pub index: u16,
}


/// When a ticker us used as an input to an abstraction, the only thing that matters is the index 
/// (and that only so you can have the same ordering along one asix of the price action matrix)
/// In contrast, for a specific abstraction (EigenVector or (Sub/Mega)Centroid), 
/// There are two additional items that should be known- weight and cosine similarity 
#[derive(Serialize, Deserialize)]
pub struct SyntheticTickerInput {
    /// The ticker for the input
    pub ticker: String,
    /// For any given trading day, the sigma level of the synthetic will be the weighted average of its input tickers. 
    /// Here is the weight of one ticker in that weighted average 
    pub weight: f32,
    /// The cosine similarity (-1 to +1) of the input ticker to the output synthetic
    pub cossim: f32,
}

/// A clustering context takes the input price action from an abstraction context
/// Using cosine similarity as a distance metric:
/// 1) Tickers are assigned to subclusters,
/// 2) Subclusters are assigned to clusters,
/// 3) Subclusters are assigned to megaclusters
/// In each case, the price history sigma is used as input (i.e. % price change/stdev of % price change),
/// And the centroid is calculated to be a simple average of its input assignees
/// That is, you get the price action of the centroid 
#[derive(Serialize, Deserialize)]
pub struct ClusteringContext {
    /// a unique CHAR(10) string identifying the clustering context
    pub clst_ctx_id: String,
    /// the number of subclusters used 
    pub ct_subc: i16,
    /// the number of clusters used 
    pub ct_clst: i16,
    /// the number of megaclusters used 
    pub ct_mega: i16,
}


/// Two key inputs define an abstraction context:
/// 1) A set of tickers whose price action you wish to describe 
/// 2) A timeframe over which their price action is considered 
#[derive(Serialize, Deserialize)]
pub struct AbstractionContext {
    /// a unique CHAR(10) string identifier for this abstraction context 
    pub abst_ctx_id: String,
    /// A name describing the intended scope of this abstraction (i.e. S&P 500 members 2019-2023)
    pub descrip: String,
    /// The timeframe for this abstraction
    pub timeframe: TimeFrame,
    /// a list of input stocks used in this abstraction context - i.e. those to be clustered and
    /// used for Principal Component Analysis 
    pub inputs: Vec<AbstractionInput>,
    /// A list of any clustering operations that have been calculated within this abstraction context, typically just one 
    pub clustering: Vec<ClusteringContext>,
}

/// Given the price action of any set of tickers over a specified timeframe, 
/// The combined price action can be broken down into Eigenvectors, also known as principal component analysis 
/// This struct captures key information about an eigenvector (or principal component) of an abstraction
#[derive(Serialize, Deserialize)]
pub struct Eigenvector {
    /// A synthetic, arbitrary symbol used to track this component so it can be used as an input to
    /// models like a ticker
    pub symbol: String,
    /// The abstraction context id in which this analysis was performed 
    pub abst_ctx_id: String,
    /// The index/rank of this component in the PCA, 1 is most important, 2 second, etc.
    pub rank: u16,
    /// The eigenvalue associated with this component 
    pub eigenvalue: f32,
    /// the percent of variance described by this component, defined as 100*eigenvalue/(sum
    /// eigenvalues)
    pub pct_variance: f32,
}


/// This struct represents a breakdown of an eigenvector and its inputs 
#[derive(Serialize, Deserialize)]
pub struct EigenvectorDetail {
    pub eigenvector: Eigenvector,
    pub inputs: Vec<SyntheticTickerInput>,
}


/// The approach taken to clusterin in fathom is as follows:
/// 1) tickers that move (very roughly) together are grouped into SubClusters, 512 by default
/// 2) SubClusters that move (very roughly) together are grouped into Clusters, 64 by default
/// 4) Cluster that move (very roughly) together are group into MegaClusters, 8 by default
#[derive(Serialize, Deserialize)]
pub enum CentroidType {
    SubCentroid,
    Centroid,
    MegaCentroid,
}

/// The Centroid struct represents a (Sub/Mega)Centroid in the Sub/Cluster/Megacluster aggregating scheme.
pub struct Centroid {
    /// A unique symbol used to represent this centroid, similar to a ticker 
    pub symbol: String,
    /// The type of centroid 
    pub ctype: CentroidType,
    /// The id for the abstraction context in which the clustering was performed
    pub abst_ctx_id: String,
    /// One abstraction context can have more than one clustering context (with differeing counts of subclusters / clusters / megaclusters)
    pub clst_ctx_id: String,
    /// input tickers or symbols assigned to this centroid 
    pub members: Vec<SyntheticTickerInput>,
}


