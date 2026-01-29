//! Multi-source data aggregation
//!
//! Aggregates data from multiple sources:
//! - Polymarket (prediction markets)
//! - Binance (crypto prices for correlation)
//! - Other exchanges (future support)

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Data source identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataSource {
    Polymarket,
    Binance,
    Coinbase,
    Custom(u32),
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Polymarket => write!(f, "polymarket"),
            DataSource::Binance => write!(f, "binance"),
            DataSource::Coinbase => write!(f, "coinbase"),
            DataSource::Custom(id) => write!(f, "custom_{}", id),
        }
    }
}

/// Aggregated price from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPrice {
    /// Symbol/token ID
    pub symbol: String,
    /// Aggregated mid price
    pub price: Decimal,
    /// Confidence score (0-1)
    pub confidence: Decimal,
    /// Number of sources used
    pub source_count: usize,
    /// Individual source prices
    pub sources: Vec<SourcePrice>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Price from a single source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePrice {
    pub source: DataSource,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub last: Option<Decimal>,
    pub volume_24h: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    /// Weight for aggregation (0-1)
    pub weight: Decimal,
}

impl SourcePrice {
    /// Get mid price
    pub fn mid(&self) -> Option<Decimal> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) => Some((b + a) / dec!(2)),
            _ => self.last,
        }
    }
}

/// Source configuration
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub source: DataSource,
    /// Base weight for this source
    pub weight: Decimal,
    /// Maximum age before data is stale (seconds)
    pub max_age_secs: i64,
    /// Whether this source is required
    pub required: bool,
}

/// Data aggregator
pub struct DataAggregator {
    /// Source configurations
    sources: HashMap<DataSource, SourceConfig>,
    /// Latest data per source per symbol
    data: Arc<RwLock<HashMap<String, HashMap<DataSource, SourcePrice>>>>,
    /// Correlation data (crypto prices for market correlation)
    correlations: Arc<RwLock<HashMap<String, Decimal>>>,
}

impl Default for DataAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl DataAggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            data: Arc::new(RwLock::new(HashMap::new())),
            correlations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with common sources pre-configured
    pub fn with_defaults() -> Self {
        let mut agg = Self::new();
        
        // Polymarket - primary source for prediction markets
        agg.add_source(SourceConfig {
            source: DataSource::Polymarket,
            weight: dec!(1.0),
            max_age_secs: 30,
            required: true,
        });
        
        // Binance - for crypto correlation
        agg.add_source(SourceConfig {
            source: DataSource::Binance,
            weight: dec!(0.8),
            max_age_secs: 10,
            required: false,
        });
        
        agg
    }

    /// Add a data source
    pub fn add_source(&mut self, config: SourceConfig) {
        self.sources.insert(config.source, config);
    }

    /// Update price from a source
    pub fn update(&self, symbol: &str, price: SourcePrice) {
        let mut data = self.data.write();
        let symbol_data = data.entry(symbol.to_string()).or_insert_with(HashMap::new);
        symbol_data.insert(price.source, price);
    }

    /// Update correlation price (e.g., BTC, ETH)
    pub fn update_correlation(&self, symbol: &str, price: Decimal) {
        let mut correlations = self.correlations.write();
        correlations.insert(symbol.to_string(), price);
    }

    /// Get correlation price
    pub fn get_correlation(&self, symbol: &str) -> Option<Decimal> {
        let correlations = self.correlations.read();
        correlations.get(symbol).copied()
    }

    /// Get aggregated price for a symbol
    pub fn aggregate(&self, symbol: &str) -> Option<AggregatedPrice> {
        let data = self.data.read();
        let symbol_data = data.get(symbol)?;

        let now = Utc::now();
        let mut valid_sources: Vec<SourcePrice> = Vec::new();
        let mut total_weight = Decimal::ZERO;

        for (source, price) in symbol_data {
            if let Some(config) = self.sources.get(source) {
                // Check if data is fresh
                let age = (now - price.timestamp).num_seconds();
                if age <= config.max_age_secs {
                    // Adjust weight based on freshness
                    let freshness = dec!(1) - Decimal::from(age) / Decimal::from(config.max_age_secs);
                    let adjusted_weight = config.weight * freshness;
                    
                    let mut source_price = price.clone();
                    source_price.weight = adjusted_weight;
                    valid_sources.push(source_price);
                    total_weight += adjusted_weight;
                } else {
                    debug!("Stale data from {:?} for {}: {}s old", source, symbol, age);
                }
            }
        }

        // Check required sources
        for (source, config) in &self.sources {
            if config.required && !valid_sources.iter().any(|p| p.source == *source) {
                warn!("Required source {:?} missing for {}", source, symbol);
                return None;
            }
        }

        if valid_sources.is_empty() {
            return None;
        }

        // Weighted average
        let weighted_sum: Decimal = valid_sources
            .iter()
            .filter_map(|p| p.mid().map(|m| m * p.weight))
            .sum();

        let price = if total_weight > Decimal::ZERO {
            weighted_sum / total_weight
        } else {
            return None;
        };

        // Calculate confidence based on source agreement
        let confidence = self.calculate_confidence(&valid_sources, price);

        Some(AggregatedPrice {
            symbol: symbol.to_string(),
            price,
            confidence,
            source_count: valid_sources.len(),
            sources: valid_sources,
            timestamp: now,
        })
    }

    /// Calculate confidence based on source agreement
    fn calculate_confidence(&self, sources: &[SourcePrice], avg_price: Decimal) -> Decimal {
        if sources.is_empty() || avg_price == Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Calculate variance from average
        let variance: Decimal = sources
            .iter()
            .filter_map(|p| {
                p.mid().map(|m| {
                    let diff = (m - avg_price) / avg_price;
                    diff * diff * p.weight
                })
            })
            .sum();

        let total_weight: Decimal = sources.iter().map(|p| p.weight).sum();
        let normalized_variance = if total_weight > Decimal::ZERO {
            variance / total_weight
        } else {
            Decimal::ZERO
        };

        // Convert variance to confidence (lower variance = higher confidence)
        // Using exponential decay: confidence = exp(-k * variance)
        let k = dec!(10); // Sensitivity parameter
        let confidence = dec!(1) / (dec!(1) + k * normalized_variance);

        // Boost confidence for multiple sources
        let source_boost = Decimal::from(sources.len().min(4) as u32) / dec!(4);
        
        (confidence * dec!(0.7) + source_boost * dec!(0.3)).min(dec!(1))
    }

    /// Get all symbols with data
    pub fn symbols(&self) -> Vec<String> {
        let data = self.data.read();
        data.keys().cloned().collect()
    }

    /// Clear stale data
    pub fn cleanup(&self) {
        let now = Utc::now();
        let mut data = self.data.write();
        
        for symbol_data in data.values_mut() {
            symbol_data.retain(|source, price| {
                if let Some(config) = self.sources.get(source) {
                    let age = (now - price.timestamp).num_seconds();
                    age <= config.max_age_secs * 2 // Keep 2x max age for recovery
                } else {
                    false
                }
            });
        }

        // Remove empty symbols
        data.retain(|_, v| !v.is_empty());
    }

    /// Get raw data for a symbol from a specific source
    pub fn get_source_price(&self, symbol: &str, source: DataSource) -> Option<SourcePrice> {
        let data = self.data.read();
        data.get(symbol)?.get(&source).cloned()
    }

    /// Start periodic cleanup task
    pub fn start_cleanup_task(self: Arc<Self>, interval_secs: u64) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                interval.tick().await;
                self.cleanup();
                debug!("Aggregator cleanup completed");
            }
        });
    }
}

/// Binance price feed adapter
pub struct BinanceFeed {
    symbols: Vec<String>,
}

impl BinanceFeed {
    pub fn new(symbols: Vec<String>) -> Self {
        Self { symbols }
    }

    /// Parse Binance WebSocket message
    pub fn parse_ticker(msg: &str) -> Option<(String, SourcePrice)> {
        let json: serde_json::Value = serde_json::from_str(msg).ok()?;
        
        let symbol = json.get("s")?.as_str()?.to_string();
        let bid = json.get("b")?.as_str()?.parse::<Decimal>().ok();
        let ask = json.get("a")?.as_str()?.parse::<Decimal>().ok();
        let last = json.get("c")?.as_str()?.parse::<Decimal>().ok();
        let volume = json.get("v")?.as_str()?.parse::<Decimal>().ok();

        Some((
            symbol,
            SourcePrice {
                source: DataSource::Binance,
                bid,
                ask,
                last,
                volume_24h: volume,
                timestamp: Utc::now(),
                weight: dec!(1.0),
            },
        ))
    }

    /// Generate WebSocket subscription message
    pub fn subscribe_message(&self) -> String {
        let streams: Vec<String> = self
            .symbols
            .iter()
            .map(|s| format!("{}@ticker", s.to_lowercase()))
            .collect();

        serde_json::json!({
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        })
        .to_string()
    }

    /// Get WebSocket URL
    pub fn ws_url(&self) -> String {
        "wss://stream.binance.com:9443/ws".to_string()
    }
}

/// Polymarket price adapter
pub struct PolymarketFeed;

impl PolymarketFeed {
    /// Parse Polymarket WebSocket message
    pub fn parse_price(msg: &str) -> Option<(String, SourcePrice)> {
        let json: serde_json::Value = serde_json::from_str(msg).ok()?;
        
        let msg_type = json.get("type")?.as_str()?;
        if msg_type != "price_change" && msg_type != "book_snapshot" {
            return None;
        }

        let data = json.get("data")?;
        let token_id = data.get("asset_id")?.as_str()?.to_string();
        
        let bid = data.get("best_bid")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok());
        let ask = data.get("best_ask")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok());
        let last = data.get("price")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok());

        Some((
            token_id,
            SourcePrice {
                source: DataSource::Polymarket,
                bid,
                ask,
                last,
                volume_24h: None,
                timestamp: Utc::now(),
                weight: dec!(1.0),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_source_display() {
        assert_eq!(DataSource::Polymarket.to_string(), "polymarket");
        assert_eq!(DataSource::Binance.to_string(), "binance");
        assert_eq!(DataSource::Custom(42).to_string(), "custom_42");
    }

    #[test]
    fn test_source_price_mid() {
        let price = SourcePrice {
            source: DataSource::Polymarket,
            bid: Some(dec!(0.45)),
            ask: Some(dec!(0.55)),
            last: Some(dec!(0.50)),
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        };
        
        assert_eq!(price.mid(), Some(dec!(0.50)));
    }

    #[test]
    fn test_source_price_mid_fallback() {
        let price = SourcePrice {
            source: DataSource::Polymarket,
            bid: None,
            ask: None,
            last: Some(dec!(0.50)),
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        };
        
        assert_eq!(price.mid(), Some(dec!(0.50)));
    }

    #[test]
    fn test_aggregator_update_and_get() {
        let agg = DataAggregator::with_defaults();
        
        agg.update("TOKEN1", SourcePrice {
            source: DataSource::Polymarket,
            bid: Some(dec!(0.45)),
            ask: Some(dec!(0.55)),
            last: None,
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        let price = agg.get_source_price("TOKEN1", DataSource::Polymarket);
        assert!(price.is_some());
        assert_eq!(price.unwrap().bid, Some(dec!(0.45)));
    }

    #[test]
    fn test_aggregator_aggregate() {
        let mut agg = DataAggregator::new();
        
        // Only add non-required source
        agg.add_source(SourceConfig {
            source: DataSource::Binance,
            weight: dec!(1.0),
            max_age_secs: 60,
            required: false,
        });
        
        agg.update("BTCUSDT", SourcePrice {
            source: DataSource::Binance,
            bid: Some(dec!(50000)),
            ask: Some(dec!(50010)),
            last: None,
            volume_24h: Some(dec!(1000)),
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        let result = agg.aggregate("BTCUSDT");
        assert!(result.is_some());
        let agg_price = result.unwrap();
        assert_eq!(agg_price.source_count, 1);
        assert!(agg_price.price > dec!(50000));
    }

    #[test]
    fn test_aggregator_multi_source() {
        let mut agg = DataAggregator::new();
        
        agg.add_source(SourceConfig {
            source: DataSource::Binance,
            weight: dec!(1.0),
            max_age_secs: 60,
            required: false,
        });
        agg.add_source(SourceConfig {
            source: DataSource::Coinbase,
            weight: dec!(1.0),
            max_age_secs: 60,
            required: false,
        });
        
        agg.update("BTCUSDT", SourcePrice {
            source: DataSource::Binance,
            bid: Some(dec!(50000)),
            ask: Some(dec!(50010)),
            last: None,
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        agg.update("BTCUSDT", SourcePrice {
            source: DataSource::Coinbase,
            bid: Some(dec!(50020)),
            ask: Some(dec!(50030)),
            last: None,
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        let result = agg.aggregate("BTCUSDT");
        assert!(result.is_some());
        let agg_price = result.unwrap();
        assert_eq!(agg_price.source_count, 2);
        // Average of (50005, 50025) = 50015
        assert!(agg_price.price > dec!(50000) && agg_price.price < dec!(50030));
    }

    #[test]
    fn test_aggregator_confidence() {
        let agg = DataAggregator::new();
        
        // Sources with same price = high confidence
        let sources = vec![
            SourcePrice {
                source: DataSource::Binance,
                bid: Some(dec!(100)),
                ask: Some(dec!(100)),
                last: None,
                volume_24h: None,
                timestamp: Utc::now(),
                weight: dec!(1.0),
            },
            SourcePrice {
                source: DataSource::Coinbase,
                bid: Some(dec!(100)),
                ask: Some(dec!(100)),
                last: None,
                volume_24h: None,
                timestamp: Utc::now(),
                weight: dec!(1.0),
            },
        ];
        
        let confidence = agg.calculate_confidence(&sources, dec!(100));
        assert!(confidence > dec!(0.8));
    }

    #[test]
    fn test_aggregator_correlation() {
        let agg = DataAggregator::new();
        
        agg.update_correlation("BTC", dec!(50000));
        agg.update_correlation("ETH", dec!(3000));
        
        assert_eq!(agg.get_correlation("BTC"), Some(dec!(50000)));
        assert_eq!(agg.get_correlation("ETH"), Some(dec!(3000)));
        assert_eq!(agg.get_correlation("XRP"), None);
    }

    #[test]
    fn test_binance_feed() {
        let feed = BinanceFeed::new(vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()]);
        
        let sub_msg = feed.subscribe_message();
        assert!(sub_msg.contains("btcusdt@ticker"));
        assert!(sub_msg.contains("ethusdt@ticker"));
        
        assert_eq!(feed.ws_url(), "wss://stream.binance.com:9443/ws");
    }

    #[test]
    fn test_binance_parse_ticker() {
        let msg = r#"{"s":"BTCUSDT","b":"50000.00","a":"50010.00","c":"50005.00","v":"1000.5"}"#;
        
        let result = BinanceFeed::parse_ticker(msg);
        assert!(result.is_some());
        let (symbol, price) = result.unwrap();
        assert_eq!(symbol, "BTCUSDT");
        assert_eq!(price.source, DataSource::Binance);
        assert_eq!(price.bid, Some(dec!(50000)));
        assert_eq!(price.ask, Some(dec!(50010)));
    }

    #[test]
    fn test_symbols() {
        let agg = DataAggregator::new();
        
        agg.update("TOKEN1", SourcePrice {
            source: DataSource::Polymarket,
            bid: None,
            ask: None,
            last: Some(dec!(0.5)),
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        agg.update("TOKEN2", SourcePrice {
            source: DataSource::Polymarket,
            bid: None,
            ask: None,
            last: Some(dec!(0.6)),
            volume_24h: None,
            timestamp: Utc::now(),
            weight: dec!(1.0),
        });
        
        let symbols = agg.symbols();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"TOKEN1".to_string()));
        assert!(symbols.contains(&"TOKEN2".to_string()));
    }
}
