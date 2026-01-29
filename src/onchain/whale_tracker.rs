//! Whale Wallet Tracking Module
//!
//! Monitors large wallet movements and generates trading signals.
//! Whale movements to exchanges often precede selling pressure,
//! while movements from exchanges suggest accumulation.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// Type of whale movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhaleMovement {
    /// Moving to exchange (potential sell)
    ToExchange,
    /// Moving from exchange (accumulation)
    FromExchange,
    /// Wallet to wallet transfer
    WalletToWallet,
    /// Unknown movement type
    Unknown,
}

impl WhaleMovement {
    /// Get sentiment impact of this movement type
    pub fn sentiment_impact(&self) -> f64 {
        match self {
            WhaleMovement::ToExchange => -0.6,     // Bearish - likely selling
            WhaleMovement::FromExchange => 0.7,    // Bullish - accumulation
            WhaleMovement::WalletToWallet => 0.0,  // Neutral
            WhaleMovement::Unknown => 0.0,         // Neutral
        }
    }
}

/// A whale alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlert {
    /// Asset being moved (BTC, ETH, etc.)
    pub asset: String,
    /// Value in USD
    pub amount_usd: f64,
    /// Type of movement
    pub movement: WhaleMovement,
    /// Source wallet label (if known)
    pub from_label: Option<String>,
    /// Destination wallet label (if known)
    pub to_label: Option<String>,
    /// Transaction hash
    pub tx_hash: String,
    /// When the transaction occurred
    pub timestamp: DateTime<Utc>,
}

impl WhaleAlert {
    /// Calculate the signal impact of this alert
    pub fn calculate_impact(&self, threshold_usd: f64) -> f64 {
        // Base impact from movement type
        let base_impact = self.movement.sentiment_impact();
        
        // Scale by size relative to threshold
        let size_multiplier = (self.amount_usd / threshold_usd).min(3.0);
        
        // Larger transactions have more impact, up to 3x threshold
        base_impact * (0.5 + 0.5 * size_multiplier.sqrt())
    }
    
    /// Check if alert is from a known entity
    pub fn is_labeled(&self) -> bool {
        self.from_label.is_some() || self.to_label.is_some()
    }
}

/// Known exchange wallet labels
#[derive(Debug, Clone)]
pub struct ExchangeLabels {
    labels: HashMap<String, String>,
}

impl Default for ExchangeLabels {
    fn default() -> Self {
        let mut labels = HashMap::new();
        
        // Major centralized exchanges
        labels.insert("binance".to_lowercase(), "Binance".to_string());
        labels.insert("coinbase".to_lowercase(), "Coinbase".to_string());
        labels.insert("kraken".to_lowercase(), "Kraken".to_string());
        labels.insert("okx".to_lowercase(), "OKX".to_string());
        labels.insert("bybit".to_lowercase(), "Bybit".to_string());
        labels.insert("bitfinex".to_lowercase(), "Bitfinex".to_string());
        labels.insert("huobi".to_lowercase(), "Huobi".to_string());
        labels.insert("kucoin".to_lowercase(), "KuCoin".to_string());
        labels.insert("gate.io".to_lowercase(), "Gate.io".to_string());
        labels.insert("gemini".to_lowercase(), "Gemini".to_string());
        
        Self { labels }
    }
}

impl ExchangeLabels {
    /// Check if a label matches a known exchange
    pub fn is_exchange(&self, label: &str) -> bool {
        let lower = label.to_lowercase();
        self.labels.keys().any(|k| lower.contains(k))
    }
    
    /// Get normalized exchange name
    pub fn get_exchange_name(&self, label: &str) -> Option<&str> {
        let lower = label.to_lowercase();
        for (key, name) in &self.labels {
            if lower.contains(key) {
                return Some(name);
            }
        }
        None
    }
}

/// Whale tracking service
pub struct WhaleTracker {
    /// Minimum transaction size to track (USD)
    threshold_usd: f64,
    /// Recent alerts by asset
    alerts: Arc<RwLock<HashMap<String, Vec<WhaleAlert>>>>,
    /// Exchange labels for classification
    exchange_labels: ExchangeLabels,
    /// Alert retention period
    retention_hours: u32,
}

impl WhaleTracker {
    /// Create a new whale tracker
    pub fn new(threshold_usd: f64) -> Self {
        Self {
            threshold_usd,
            alerts: Arc::new(RwLock::new(HashMap::new())),
            exchange_labels: ExchangeLabels::default(),
            retention_hours: 24,
        }
    }
    
    /// Add a whale alert
    pub async fn add_alert(&self, alert: WhaleAlert) {
        let mut alerts = self.alerts.write().await;
        alerts
            .entry(alert.asset.to_uppercase())
            .or_insert_with(Vec::new)
            .push(alert);
    }
    
    /// Get aggregated signal for an asset
    /// Returns (sentiment_signal, alert_count_1h)
    pub async fn get_signal(&self, asset: &str) -> (f64, u32) {
        let alerts = self.alerts.read().await;
        let asset_upper = asset.to_uppercase();
        
        let asset_alerts = match alerts.get(&asset_upper) {
            Some(a) => a,
            None => return (0.0, 0),
        };
        
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);
        let four_hours_ago = now - Duration::hours(4);
        
        // Count alerts in last hour
        let alerts_1h: Vec<_> = asset_alerts.iter()
            .filter(|a| a.timestamp > one_hour_ago)
            .collect();
        
        // Calculate weighted signal from last 4 hours
        let mut total_impact = 0.0;
        let mut total_weight = 0.0;
        
        for alert in asset_alerts.iter().filter(|a| a.timestamp > four_hours_ago) {
            // More recent alerts get higher weight
            let age_hours = (now - alert.timestamp).num_minutes() as f64 / 60.0;
            let time_weight = 1.0 / (1.0 + age_hours * 0.25); // Decay over time
            
            let impact = alert.calculate_impact(self.threshold_usd);
            total_impact += impact * time_weight;
            total_weight += time_weight;
        }
        
        let signal = if total_weight > 0.0 {
            (total_impact / total_weight).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        
        (signal, alerts_1h.len() as u32)
    }
    
    /// Get recent alerts for an asset
    pub async fn get_recent_alerts(&self, asset: &str, hours: u32) -> Vec<WhaleAlert> {
        let alerts = self.alerts.read().await;
        let asset_upper = asset.to_uppercase();
        
        let cutoff = Utc::now() - Duration::hours(hours as i64);
        
        alerts.get(&asset_upper)
            .map(|a| a.iter()
                .filter(|alert| alert.timestamp > cutoff)
                .cloned()
                .collect())
            .unwrap_or_default()
    }
    
    /// Classify a transaction based on labels
    pub fn classify_movement(&self, from_label: Option<&str>, to_label: Option<&str>) -> WhaleMovement {
        let from_is_exchange = from_label
            .map(|l| self.exchange_labels.is_exchange(l))
            .unwrap_or(false);
        let to_is_exchange = to_label
            .map(|l| self.exchange_labels.is_exchange(l))
            .unwrap_or(false);
        
        match (from_is_exchange, to_is_exchange) {
            (false, true) => WhaleMovement::ToExchange,
            (true, false) => WhaleMovement::FromExchange,
            (false, false) => WhaleMovement::WalletToWallet,
            (true, true) => WhaleMovement::Unknown, // Exchange to exchange
        }
    }
    
    /// Clear alerts older than retention period
    pub async fn clear_old_alerts(&self) {
        let mut alerts = self.alerts.write().await;
        let cutoff = Utc::now() - Duration::hours(self.retention_hours as i64);
        
        for asset_alerts in alerts.values_mut() {
            asset_alerts.retain(|a| a.timestamp > cutoff);
        }
    }
    
    /// Get summary statistics
    pub async fn get_stats(&self) -> WhaleStats {
        let alerts = self.alerts.read().await;
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);
        let twenty_four_hours_ago = now - Duration::hours(24);
        
        let mut total_1h = 0;
        let mut total_24h = 0;
        let mut volume_usd_24h = 0.0;
        let mut to_exchange_count = 0;
        let mut from_exchange_count = 0;
        
        for asset_alerts in alerts.values() {
            for alert in asset_alerts {
                if alert.timestamp > twenty_four_hours_ago {
                    total_24h += 1;
                    volume_usd_24h += alert.amount_usd;
                    
                    match alert.movement {
                        WhaleMovement::ToExchange => to_exchange_count += 1,
                        WhaleMovement::FromExchange => from_exchange_count += 1,
                        _ => {}
                    }
                }
                if alert.timestamp > one_hour_ago {
                    total_1h += 1;
                }
            }
        }
        
        WhaleStats {
            alerts_1h: total_1h,
            alerts_24h: total_24h,
            volume_usd_24h,
            to_exchange_count,
            from_exchange_count,
        }
    }
}

/// Whale tracking statistics
#[derive(Debug, Clone, Serialize)]
pub struct WhaleStats {
    pub alerts_1h: u32,
    pub alerts_24h: u32,
    pub volume_usd_24h: f64,
    pub to_exchange_count: u32,
    pub from_exchange_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_whale_movement_sentiment() {
        assert!(WhaleMovement::ToExchange.sentiment_impact() < 0.0);
        assert!(WhaleMovement::FromExchange.sentiment_impact() > 0.0);
        assert_eq!(WhaleMovement::WalletToWallet.sentiment_impact(), 0.0);
    }
    
    #[test]
    fn test_whale_alert_impact() {
        let threshold = 1_000_000.0;
        
        // Small alert to exchange
        let small_alert = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Binance".to_string()),
            tx_hash: "0x123".to_string(),
            timestamp: Utc::now(),
        };
        
        let small_impact = small_alert.calculate_impact(threshold);
        assert!(small_impact < 0.0); // Bearish
        
        // Large alert from exchange
        let large_alert = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 5_000_000.0,
            movement: WhaleMovement::FromExchange,
            from_label: Some("Coinbase".to_string()),
            to_label: None,
            tx_hash: "0x456".to_string(),
            timestamp: Utc::now(),
        };
        
        let large_impact = large_alert.calculate_impact(threshold);
        assert!(large_impact > 0.0); // Bullish
        assert!(large_impact.abs() > small_impact.abs()); // Larger = more impact
    }
    
    #[test]
    fn test_exchange_labels() {
        let labels = ExchangeLabels::default();
        
        assert!(labels.is_exchange("Binance Hot Wallet"));
        assert!(labels.is_exchange("coinbase-1"));
        assert!(!labels.is_exchange("Unknown: 0x123")); // Contains nothing known
        assert!(!labels.is_exchange("Unknown Wallet"));
        
        assert_eq!(labels.get_exchange_name("Binance Hot Wallet 1"), Some("Binance"));
        assert_eq!(labels.get_exchange_name("Random Wallet"), None);
    }
    
    #[tokio::test]
    async fn test_whale_tracker_add_get() {
        let tracker = WhaleTracker::new(1_000_000.0);
        
        let alert = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 5_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Binance".to_string()),
            tx_hash: "0x123".to_string(),
            timestamp: Utc::now(),
        };
        
        tracker.add_alert(alert.clone()).await;
        
        let (signal, count) = tracker.get_signal("BTC").await;
        assert!(signal < 0.0); // Bearish
        assert_eq!(count, 1);
        
        // Case insensitive
        let (signal2, _) = tracker.get_signal("btc").await;
        assert_eq!(signal, signal2);
    }
    
    #[tokio::test]
    async fn test_whale_tracker_multiple_alerts() {
        let tracker = WhaleTracker::new(1_000_000.0);
        
        // Add bearish alert
        tracker.add_alert(WhaleAlert {
            asset: "ETH".to_string(),
            amount_usd: 2_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Kraken".to_string()),
            tx_hash: "0x1".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        // Add bullish alert
        tracker.add_alert(WhaleAlert {
            asset: "ETH".to_string(),
            amount_usd: 3_000_000.0,
            movement: WhaleMovement::FromExchange,
            from_label: Some("Coinbase".to_string()),
            to_label: None,
            tx_hash: "0x2".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        let (signal, count) = tracker.get_signal("ETH").await;
        assert_eq!(count, 2);
        // Signal should be mixed but slightly bullish (larger bullish tx)
        assert!(signal.abs() < 0.5); // Mixed signals
    }
    
    #[tokio::test]
    async fn test_classify_movement() {
        let tracker = WhaleTracker::new(1_000_000.0);
        
        assert_eq!(
            tracker.classify_movement(Some("Unknown Wallet"), Some("Binance Hot Wallet")),
            WhaleMovement::ToExchange
        );
        
        assert_eq!(
            tracker.classify_movement(Some("Coinbase Cold Storage"), Some("Random: 0x123")),
            WhaleMovement::FromExchange
        );
        
        assert_eq!(
            tracker.classify_movement(Some("Wallet A"), Some("Wallet B")),
            WhaleMovement::WalletToWallet
        );
        
        assert_eq!(
            tracker.classify_movement(Some("Binance"), Some("Kraken")),
            WhaleMovement::Unknown
        );
    }
    
    #[tokio::test]
    async fn test_get_recent_alerts() {
        let tracker = WhaleTracker::new(1_000_000.0);
        
        // Add alert from 2 hours ago
        tracker.add_alert(WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_500_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Binance".to_string()),
            tx_hash: "0x1".to_string(),
            timestamp: Utc::now() - Duration::hours(2),
        }).await;
        
        // Add recent alert
        tracker.add_alert(WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 2_000_000.0,
            movement: WhaleMovement::FromExchange,
            from_label: Some("Coinbase".to_string()),
            to_label: None,
            tx_hash: "0x2".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        let alerts_1h = tracker.get_recent_alerts("BTC", 1).await;
        assert_eq!(alerts_1h.len(), 1);
        
        let alerts_4h = tracker.get_recent_alerts("BTC", 4).await;
        assert_eq!(alerts_4h.len(), 2);
    }
    
    #[tokio::test]
    async fn test_clear_old_alerts() {
        let mut tracker = WhaleTracker::new(1_000_000.0);
        tracker.retention_hours = 1; // Short retention for test
        
        // Add old alert
        tracker.add_alert(WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Binance".to_string()),
            tx_hash: "0x1".to_string(),
            timestamp: Utc::now() - Duration::hours(2),
        }).await;
        
        // Add recent alert
        tracker.add_alert(WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_000_000.0,
            movement: WhaleMovement::FromExchange,
            from_label: Some("Coinbase".to_string()),
            to_label: None,
            tx_hash: "0x2".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        tracker.clear_old_alerts().await;
        
        let alerts = tracker.get_recent_alerts("BTC", 24).await;
        assert_eq!(alerts.len(), 1);
    }
    
    #[tokio::test]
    async fn test_get_stats() {
        let tracker = WhaleTracker::new(1_000_000.0);
        
        tracker.add_alert(WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 2_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: None,
            to_label: Some("Binance".to_string()),
            tx_hash: "0x1".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        tracker.add_alert(WhaleAlert {
            asset: "ETH".to_string(),
            amount_usd: 1_500_000.0,
            movement: WhaleMovement::FromExchange,
            from_label: Some("Coinbase".to_string()),
            to_label: None,
            tx_hash: "0x2".to_string(),
            timestamp: Utc::now(),
        }).await;
        
        let stats = tracker.get_stats().await;
        
        assert_eq!(stats.alerts_1h, 2);
        assert_eq!(stats.alerts_24h, 2);
        assert_eq!(stats.volume_usd_24h, 3_500_000.0);
        assert_eq!(stats.to_exchange_count, 1);
        assert_eq!(stats.from_exchange_count, 1);
    }
    
    #[test]
    fn test_alert_is_labeled() {
        let labeled = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_000_000.0,
            movement: WhaleMovement::ToExchange,
            from_label: Some("Whale".to_string()),
            to_label: None,
            tx_hash: "0x1".to_string(),
            timestamp: Utc::now(),
        };
        assert!(labeled.is_labeled());
        
        let unlabeled = WhaleAlert {
            asset: "BTC".to_string(),
            amount_usd: 1_000_000.0,
            movement: WhaleMovement::Unknown,
            from_label: None,
            to_label: None,
            tx_hash: "0x2".to_string(),
            timestamp: Utc::now(),
        };
        assert!(!unlabeled.is_labeled());
    }
}
