//! Data engineering module
//!
//! Enhanced data capabilities:
//! - Data validation and cleaning
//! - Multi-source aggregation (Polymarket + Binance + others)
//! - Rate limiting and caching
//!
//! Note: WebSocket is now in `client::polymarket_ws`

pub mod aggregator;
pub mod cleaning;

pub use aggregator::{DataAggregator, AggregatedPrice, DataSource};
pub use cleaning::{DataCleaner, CleaningConfig, ValidationResult, Anomaly};
