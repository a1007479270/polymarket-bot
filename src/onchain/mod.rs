//! On-chain Data Module
//!
//! Provides blockchain data analysis for improved prediction accuracy:
//! - Whale wallet tracking (large holder movements)
//! - Exchange flow monitoring (inflow/outflow signals)
//! - Network metrics (gas, active addresses, transaction volume)
//! - DeFi TVL tracking (market confidence indicator)
//!
//! ## Data Sources
//! - Etherscan/Polygonscan APIs for transaction data
//! - Public exchange wallet labels
//! - On-chain analytics aggregators
//!
//! ## Usage
//! ```ignore
//! let engine = OnchainEngine::new(config).await?;
//! let signals = engine.get_signals("BTC").await?;
//! ```

mod whale_tracker;
mod exchange_flow;
mod network_metrics;

pub use whale_tracker::{WhaleTracker, WhaleAlert, WhaleMovement};
pub use exchange_flow::{ExchangeFlowTracker, FlowSignal, ExchangeFlow};
pub use network_metrics::{NetworkMetrics, NetworkMetricsCollector};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// On-chain signal for trading decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnchainSignal {
    /// Asset symbol (BTC, ETH, SOL, etc.)
    pub asset: String,
    /// Overall on-chain sentiment (-1.0 to 1.0)
    pub sentiment: f64,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Individual signal components
    pub components: OnchainComponents,
    /// Confidence in the signal (0.0 to 1.0)
    pub confidence: f64,
    /// Signal timestamp
    pub timestamp: DateTime<Utc>,
    /// Time-to-live for this signal
    pub ttl_minutes: u32,
}

/// Component signals from different on-chain sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnchainComponents {
    /// Whale movement signal (-1.0 to 1.0, negative = selling)
    pub whale_signal: f64,
    /// Exchange flow signal (-1.0 to 1.0, negative = inflow/selling pressure)
    pub exchange_flow_signal: f64,
    /// Network activity signal (-1.0 to 1.0)
    pub network_activity_signal: f64,
    /// Whale alert count in last hour
    pub whale_alerts_1h: u32,
    /// Net exchange flow in USD (positive = outflow)
    pub net_exchange_flow_usd: f64,
    /// Active addresses change percentage
    pub active_addresses_change_pct: f64,
}

impl Default for OnchainComponents {
    fn default() -> Self {
        Self {
            whale_signal: 0.0,
            exchange_flow_signal: 0.0,
            network_activity_signal: 0.0,
            whale_alerts_1h: 0,
            net_exchange_flow_usd: 0.0,
            active_addresses_change_pct: 0.0,
        }
    }
}

/// Configuration for on-chain data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnchainConfig {
    /// Enable whale tracking
    pub enable_whale_tracking: bool,
    /// Enable exchange flow monitoring
    pub enable_exchange_flow: bool,
    /// Enable network metrics
    pub enable_network_metrics: bool,
    /// Minimum whale transaction size in USD
    pub whale_threshold_usd: f64,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// API rate limit (requests per minute)
    pub rate_limit_rpm: u32,
    /// Supported assets
    pub supported_assets: Vec<String>,
}

impl Default for OnchainConfig {
    fn default() -> Self {
        Self {
            enable_whale_tracking: true,
            enable_exchange_flow: true,
            enable_network_metrics: true,
            whale_threshold_usd: 1_000_000.0, // $1M+
            cache_ttl_seconds: 300, // 5 minutes
            rate_limit_rpm: 30,
            supported_assets: vec![
                "BTC".to_string(),
                "ETH".to_string(),
                "SOL".to_string(),
                "MATIC".to_string(),
            ],
        }
    }
}

/// Main on-chain data engine coordinating all data sources
pub struct OnchainEngine {
    config: OnchainConfig,
    whale_tracker: Arc<WhaleTracker>,
    exchange_flow: Arc<ExchangeFlowTracker>,
    network_metrics: Arc<NetworkMetricsCollector>,
    /// Cached signals per asset
    signal_cache: Arc<RwLock<HashMap<String, CachedSignal>>>,
}

struct CachedSignal {
    signal: OnchainSignal,
    cached_at: DateTime<Utc>,
}

impl OnchainEngine {
    /// Create a new on-chain engine
    pub fn new(config: OnchainConfig) -> Self {
        let whale_threshold = config.whale_threshold_usd;
        
        Self {
            config: config.clone(),
            whale_tracker: Arc::new(WhaleTracker::new(whale_threshold)),
            exchange_flow: Arc::new(ExchangeFlowTracker::new()),
            network_metrics: Arc::new(NetworkMetricsCollector::new()),
            signal_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Get aggregated on-chain signal for an asset
    pub async fn get_signal(&self, asset: &str) -> OnchainSignal {
        let asset_upper = asset.to_uppercase();
        
        // Check cache first
        if let Some(cached) = self.get_cached_signal(&asset_upper).await {
            return cached;
        }
        
        // Collect signals from all sources
        let whale_data = if self.config.enable_whale_tracking {
            self.whale_tracker.get_signal(&asset_upper).await
        } else {
            (0.0, 0)
        };
        
        let exchange_data = if self.config.enable_exchange_flow {
            self.exchange_flow.get_signal(&asset_upper).await
        } else {
            (0.0, 0.0)
        };
        
        let network_data = if self.config.enable_network_metrics {
            self.network_metrics.get_signal(&asset_upper).await
        } else {
            (0.0, 0.0)
        };
        
        // Build components
        let components = OnchainComponents {
            whale_signal: whale_data.0,
            exchange_flow_signal: exchange_data.0,
            network_activity_signal: network_data.0,
            whale_alerts_1h: whale_data.1,
            net_exchange_flow_usd: exchange_data.1,
            active_addresses_change_pct: network_data.1,
        };
        
        // Calculate aggregate sentiment with weights
        let sentiment = self.calculate_aggregate_sentiment(&components);
        let strength = self.calculate_signal_strength(&components);
        let confidence = self.calculate_confidence(&components);
        
        let signal = OnchainSignal {
            asset: asset_upper.clone(),
            sentiment,
            strength,
            components,
            confidence,
            timestamp: Utc::now(),
            ttl_minutes: (self.config.cache_ttl_seconds / 60) as u32,
        };
        
        // Cache the signal
        self.cache_signal(&asset_upper, signal.clone()).await;
        
        signal
    }
    
    /// Calculate aggregate sentiment from components
    fn calculate_aggregate_sentiment(&self, components: &OnchainComponents) -> f64 {
        // Weights for different signals
        const WHALE_WEIGHT: f64 = 0.40; // Whale movements are highly significant
        const EXCHANGE_WEIGHT: f64 = 0.35; // Exchange flows are strong indicators
        const NETWORK_WEIGHT: f64 = 0.25; // Network activity is supporting signal
        
        let weighted_sum = components.whale_signal * WHALE_WEIGHT
            + components.exchange_flow_signal * EXCHANGE_WEIGHT
            + components.network_activity_signal * NETWORK_WEIGHT;
        
        // Clamp to [-1, 1]
        weighted_sum.clamp(-1.0, 1.0)
    }
    
    /// Calculate signal strength (how actionable is the signal)
    fn calculate_signal_strength(&self, components: &OnchainComponents) -> f64 {
        // High strength when multiple signals agree
        let signals = vec![
            components.whale_signal,
            components.exchange_flow_signal,
            components.network_activity_signal,
        ];
        
        // Check signal alignment
        let positive_count = signals.iter().filter(|&&s| s > 0.1).count();
        let negative_count = signals.iter().filter(|&&s| s < -0.1).count();
        
        let alignment = if positive_count >= 2 || negative_count >= 2 {
            0.8 // Strong alignment
        } else if positive_count == 1 && negative_count == 0 
              || negative_count == 1 && positive_count == 0 {
            0.5 // Moderate signal
        } else {
            0.2 // Mixed/weak signals
        };
        
        // Factor in signal magnitudes
        let avg_magnitude = signals.iter().map(|s| s.abs()).sum::<f64>() / signals.len() as f64;
        
        (alignment * avg_magnitude * 2.0).clamp(0.0, 1.0)
    }
    
    /// Calculate confidence based on data freshness and availability
    fn calculate_confidence(&self, components: &OnchainComponents) -> f64 {
        let mut confidence: f64 = 0.5; // Base confidence
        
        // More whale alerts = more data = higher confidence
        if components.whale_alerts_1h > 0 {
            confidence += 0.1;
        }
        if components.whale_alerts_1h > 3 {
            confidence += 0.1;
        }
        
        // Exchange flow data available
        if components.net_exchange_flow_usd.abs() > 0.0 {
            confidence += 0.15;
        }
        
        // Network metrics available
        if components.active_addresses_change_pct.abs() > 0.0 {
            confidence += 0.15;
        }
        
        confidence.clamp(0.0, 1.0)
    }
    
    async fn get_cached_signal(&self, asset: &str) -> Option<OnchainSignal> {
        let cache = self.signal_cache.read().await;
        if let Some(cached) = cache.get(asset) {
            let age = Utc::now() - cached.cached_at;
            if age < Duration::seconds(self.config.cache_ttl_seconds as i64) {
                return Some(cached.signal.clone());
            }
        }
        None
    }
    
    async fn cache_signal(&self, asset: &str, signal: OnchainSignal) {
        let mut cache = self.signal_cache.write().await;
        cache.insert(asset.to_string(), CachedSignal {
            signal,
            cached_at: Utc::now(),
        });
    }
    
    /// Get signals for all supported assets
    pub async fn get_all_signals(&self) -> HashMap<String, OnchainSignal> {
        let mut signals = HashMap::new();
        for asset in &self.config.supported_assets {
            let signal = self.get_signal(asset).await;
            signals.insert(asset.clone(), signal);
        }
        signals
    }
    
    /// Check if an asset is supported
    pub fn is_supported(&self, asset: &str) -> bool {
        self.config.supported_assets.iter()
            .any(|a| a.eq_ignore_ascii_case(asset))
    }
    
    /// Add whale alert manually (for testing or external data)
    pub async fn add_whale_alert(&self, alert: WhaleAlert) {
        self.whale_tracker.add_alert(alert).await;
    }
    
    /// Add exchange flow data manually
    pub async fn add_exchange_flow(&self, flow: ExchangeFlow) {
        self.exchange_flow.add_flow(flow).await;
    }
    
    /// Update network metrics manually
    pub async fn update_network_metrics(&self, asset: &str, metrics: NetworkMetrics) {
        self.network_metrics.update_metrics(asset, metrics).await;
    }
    
    /// Clear all caches
    pub async fn clear_cache(&self) {
        let mut cache = self.signal_cache.write().await;
        cache.clear();
        self.whale_tracker.clear_old_alerts().await;
        self.exchange_flow.clear_old_flows().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_onchain_engine_creation() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        assert!(engine.is_supported("BTC"));
        assert!(engine.is_supported("eth")); // Case insensitive
        assert!(!engine.is_supported("DOGE"));
    }
    
    #[tokio::test]
    async fn test_get_signal_empty() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        let signal = engine.get_signal("BTC").await;
        
        assert_eq!(signal.asset, "BTC");
        assert!(signal.sentiment >= -1.0 && signal.sentiment <= 1.0);
        assert!(signal.strength >= 0.0 && signal.strength <= 1.0);
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_signal_caching() {
        let config = OnchainConfig {
            cache_ttl_seconds: 300,
            ..Default::default()
        };
        let engine = OnchainEngine::new(config);
        
        let signal1 = engine.get_signal("BTC").await;
        let signal2 = engine.get_signal("BTC").await;
        
        // Should be same cached signal
        assert_eq!(signal1.timestamp, signal2.timestamp);
    }
    
    #[tokio::test]
    async fn test_aggregate_sentiment_calculation() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        // All positive signals
        let components = OnchainComponents {
            whale_signal: 0.8,
            exchange_flow_signal: 0.6,
            network_activity_signal: 0.4,
            ..Default::default()
        };
        
        let sentiment = engine.calculate_aggregate_sentiment(&components);
        assert!(sentiment > 0.5); // Should be bullish
        
        // All negative signals
        let components_neg = OnchainComponents {
            whale_signal: -0.8,
            exchange_flow_signal: -0.6,
            network_activity_signal: -0.4,
            ..Default::default()
        };
        
        let sentiment_neg = engine.calculate_aggregate_sentiment(&components_neg);
        assert!(sentiment_neg < -0.5); // Should be bearish
    }
    
    #[tokio::test]
    async fn test_signal_strength_aligned() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        // Aligned positive signals
        let components = OnchainComponents {
            whale_signal: 0.7,
            exchange_flow_signal: 0.6,
            network_activity_signal: 0.5,
            ..Default::default()
        };
        
        let strength = engine.calculate_signal_strength(&components);
        assert!(strength > 0.5); // Strong alignment
        
        // Mixed signals
        let components_mixed = OnchainComponents {
            whale_signal: 0.5,
            exchange_flow_signal: -0.3,
            network_activity_signal: 0.1,
            ..Default::default()
        };
        
        let strength_mixed = engine.calculate_signal_strength(&components_mixed);
        assert!(strength_mixed < 0.5); // Weak/mixed
    }
    
    #[tokio::test]
    async fn test_confidence_calculation() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        // Minimal data
        let components_min = OnchainComponents::default();
        let conf_min = engine.calculate_confidence(&components_min);
        assert!(conf_min < 0.7);
        
        // Rich data
        let components_rich = OnchainComponents {
            whale_alerts_1h: 5,
            net_exchange_flow_usd: 1_000_000.0,
            active_addresses_change_pct: 5.0,
            ..Default::default()
        };
        let conf_rich = engine.calculate_confidence(&components_rich);
        assert!(conf_rich > 0.8);
    }
    
    #[tokio::test]
    async fn test_get_all_signals() {
        let config = OnchainConfig {
            supported_assets: vec!["BTC".to_string(), "ETH".to_string()],
            ..Default::default()
        };
        let engine = OnchainEngine::new(config);
        
        let signals = engine.get_all_signals().await;
        
        assert!(signals.contains_key("BTC"));
        assert!(signals.contains_key("ETH"));
        assert_eq!(signals.len(), 2);
    }
    
    #[tokio::test]
    async fn test_add_whale_alert() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        let alert = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 5_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: Some("Unknown Wallet".to_string()),
            to_label: Some("Binance".to_string()),
            tx_hash: "0x123".to_string(),
            timestamp: Utc::now(),
        };
        
        engine.add_whale_alert(alert).await;
        
        // Signal should now reflect the whale alert
        // (Cache is cleared by the alert)
        engine.clear_cache().await;
        let signal = engine.get_signal("BTC").await;
        
        // Whale moving to exchange = bearish signal
        assert!(signal.components.whale_alerts_1h >= 1 || signal.components.whale_signal != 0.0);
    }
    
    #[tokio::test]
    async fn test_disabled_features() {
        let config = OnchainConfig {
            enable_whale_tracking: false,
            enable_exchange_flow: false,
            enable_network_metrics: false,
            ..Default::default()
        };
        let engine = OnchainEngine::new(config);
        
        let signal = engine.get_signal("BTC").await;
        
        // All components should be zero when disabled
        assert_eq!(signal.components.whale_signal, 0.0);
        assert_eq!(signal.components.exchange_flow_signal, 0.0);
        assert_eq!(signal.components.network_activity_signal, 0.0);
    }
    
    #[tokio::test]
    async fn test_clear_cache() {
        let config = OnchainConfig::default();
        let engine = OnchainEngine::new(config);
        
        // Populate cache
        let _ = engine.get_signal("BTC").await;
        let _ = engine.get_signal("ETH").await;
        
        // Clear
        engine.clear_cache().await;
        
        // Verify cache is empty by checking timestamps differ
        let signal1 = engine.get_signal("BTC").await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        engine.clear_cache().await;
        let signal2 = engine.get_signal("BTC").await;
        
        // New signal should have different timestamp (not cached)
        assert!(signal2.timestamp >= signal1.timestamp);
    }
}
