//! Network Metrics Module
//!
//! Tracks blockchain network health and activity metrics:
//! - Active addresses
//! - Transaction volume
//! - Gas fees (for EVM chains)
//! - Hash rate (for PoW chains)
//!
//! These metrics provide context for on-chain activity patterns.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// Network metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Asset/chain identifier
    pub asset: String,
    /// Number of active addresses (24h)
    pub active_addresses_24h: u64,
    /// Previous period active addresses (for comparison)
    pub active_addresses_prev_24h: u64,
    /// Transaction count (24h)
    pub transaction_count_24h: u64,
    /// Transaction volume in USD (24h)
    pub transaction_volume_usd_24h: f64,
    /// Average gas price (wei for ETH, lamports for SOL)
    pub avg_gas_price: Option<f64>,
    /// Network hash rate (for PoW chains)
    pub hash_rate: Option<f64>,
    /// Metric timestamp
    pub timestamp: DateTime<Utc>,
}

impl NetworkMetrics {
    /// Calculate active address change percentage
    pub fn active_address_change_pct(&self) -> f64 {
        if self.active_addresses_prev_24h == 0 {
            return 0.0;
        }
        
        let current = self.active_addresses_24h as f64;
        let prev = self.active_addresses_prev_24h as f64;
        
        ((current - prev) / prev) * 100.0
    }
    
    /// Calculate network activity score (0.0 to 1.0)
    pub fn activity_score(&self) -> f64 {
        let address_change = self.active_address_change_pct();
        
        // Positive change = higher activity = bullish
        // Clamp to reasonable range
        let normalized = (address_change / 20.0).clamp(-1.0, 1.0);
        
        // Convert to 0-1 scale
        (normalized + 1.0) / 2.0
    }
    
    /// Check if network is congested (high gas)
    pub fn is_congested(&self, threshold_gwei: f64) -> bool {
        self.avg_gas_price
            .map(|gas| gas > threshold_gwei * 1e9)
            .unwrap_or(false)
    }
}

/// Network metrics collector
pub struct NetworkMetricsCollector {
    /// Latest metrics by asset
    metrics: Arc<RwLock<HashMap<String, Vec<TimestampedMetrics>>>>,
    /// Retention period
    retention_hours: u32,
}

#[derive(Debug, Clone)]
struct TimestampedMetrics {
    metrics: NetworkMetrics,
    collected_at: DateTime<Utc>,
}

impl NetworkMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            retention_hours: 48,
        }
    }
    
    /// Update metrics for an asset
    pub async fn update_metrics(&self, asset: &str, metrics: NetworkMetrics) {
        let mut all_metrics = self.metrics.write().await;
        all_metrics
            .entry(asset.to_uppercase())
            .or_insert_with(Vec::new)
            .push(TimestampedMetrics {
                metrics,
                collected_at: Utc::now(),
            });
    }
    
    /// Get aggregated signal for an asset
    /// Returns (signal, active_address_change_pct)
    pub async fn get_signal(&self, asset: &str) -> (f64, f64) {
        let metrics = self.metrics.read().await;
        let asset_upper = asset.to_uppercase();
        
        let asset_metrics = match metrics.get(&asset_upper) {
            Some(m) if !m.is_empty() => m,
            _ => return (0.0, 0.0),
        };
        
        // Get most recent metrics
        let latest = &asset_metrics.last().unwrap().metrics;
        
        let address_change = latest.active_address_change_pct();
        
        // Convert to signal
        // Positive address change = bullish
        // Negative address change = bearish
        let signal = (address_change / 20.0).clamp(-1.0, 1.0);
        
        (signal, address_change)
    }
    
    /// Get latest metrics for an asset
    pub async fn get_latest(&self, asset: &str) -> Option<NetworkMetrics> {
        let metrics = self.metrics.read().await;
        let asset_upper = asset.to_uppercase();
        
        metrics.get(&asset_upper)
            .and_then(|m| m.last())
            .map(|tm| tm.metrics.clone())
    }
    
    /// Get metrics history for an asset
    pub async fn get_history(&self, asset: &str, hours: u32) -> Vec<NetworkMetrics> {
        let metrics = self.metrics.read().await;
        let asset_upper = asset.to_uppercase();
        let cutoff = Utc::now() - Duration::hours(hours as i64);
        
        metrics.get(&asset_upper)
            .map(|m| m.iter()
                .filter(|tm| tm.collected_at > cutoff)
                .map(|tm| tm.metrics.clone())
                .collect())
            .unwrap_or_default()
    }
    
    /// Calculate trend over time window
    pub async fn calculate_trend(&self, asset: &str, hours: u32) -> NetworkTrend {
        let history = self.get_history(asset, hours).await;
        
        if history.len() < 2 {
            return NetworkTrend::default();
        }
        
        let first = &history[0];
        let last = history.last().unwrap();
        
        // Address trend
        let address_trend = if first.active_addresses_24h == 0 {
            0.0
        } else {
            (last.active_addresses_24h as f64 - first.active_addresses_24h as f64)
                / first.active_addresses_24h as f64 * 100.0
        };
        
        // Volume trend
        let volume_trend = if first.transaction_volume_usd_24h == 0.0 {
            0.0
        } else {
            (last.transaction_volume_usd_24h - first.transaction_volume_usd_24h)
                / first.transaction_volume_usd_24h * 100.0
        };
        
        // Gas trend (if available)
        let gas_trend = match (first.avg_gas_price, last.avg_gas_price) {
            (Some(first_gas), Some(last_gas)) if first_gas > 0.0 => {
                Some((last_gas - first_gas) / first_gas * 100.0)
            }
            _ => None,
        };
        
        NetworkTrend {
            asset: asset.to_uppercase(),
            address_trend_pct: address_trend,
            volume_trend_pct: volume_trend,
            gas_trend_pct: gas_trend,
            window_hours: hours,
            data_points: history.len() as u32,
        }
    }
    
    /// Clear old metrics
    pub async fn clear_old_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        let cutoff = Utc::now() - Duration::hours(self.retention_hours as i64);
        
        for asset_metrics in metrics.values_mut() {
            asset_metrics.retain(|tm| tm.collected_at > cutoff);
        }
    }
    
    /// Get summary across all tracked assets
    pub async fn get_summary(&self) -> NetworkSummary {
        let metrics = self.metrics.read().await;
        
        let mut assets_tracked = 0;
        let mut total_active_addresses = 0u64;
        let mut total_tx_volume_usd = 0.0;
        let mut bullish_count = 0;
        let mut bearish_count = 0;
        
        for asset_metrics in metrics.values() {
            if let Some(latest) = asset_metrics.last() {
                assets_tracked += 1;
                total_active_addresses += latest.metrics.active_addresses_24h;
                total_tx_volume_usd += latest.metrics.transaction_volume_usd_24h;
                
                let change = latest.metrics.active_address_change_pct();
                if change > 5.0 {
                    bullish_count += 1;
                } else if change < -5.0 {
                    bearish_count += 1;
                }
            }
        }
        
        NetworkSummary {
            assets_tracked,
            total_active_addresses,
            total_tx_volume_usd,
            bullish_assets: bullish_count,
            bearish_assets: bearish_count,
        }
    }
}

impl Default for NetworkMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Network trend analysis
#[derive(Debug, Clone, Serialize, Default)]
pub struct NetworkTrend {
    pub asset: String,
    /// Active address change percentage
    pub address_trend_pct: f64,
    /// Transaction volume change percentage
    pub volume_trend_pct: f64,
    /// Gas price change percentage (if available)
    pub gas_trend_pct: Option<f64>,
    /// Time window analyzed
    pub window_hours: u32,
    /// Number of data points
    pub data_points: u32,
}

/// Summary across all networks
#[derive(Debug, Clone, Serialize)]
pub struct NetworkSummary {
    pub assets_tracked: u32,
    pub total_active_addresses: u64,
    pub total_tx_volume_usd: f64,
    pub bullish_assets: u32,
    pub bearish_assets: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_metrics(asset: &str, active: u64, prev_active: u64) -> NetworkMetrics {
        NetworkMetrics {
            asset: asset.to_string(),
            active_addresses_24h: active,
            active_addresses_prev_24h: prev_active,
            transaction_count_24h: 1_000_000,
            transaction_volume_usd_24h: 10_000_000_000.0,
            avg_gas_price: Some(30.0 * 1e9), // 30 gwei
            hash_rate: None,
            timestamp: Utc::now(),
        }
    }
    
    #[test]
    fn test_active_address_change_pct() {
        let metrics = create_test_metrics("BTC", 1_100_000, 1_000_000);
        assert!((metrics.active_address_change_pct() - 10.0).abs() < 0.01);
        
        let metrics_down = create_test_metrics("ETH", 900_000, 1_000_000);
        assert!((metrics_down.active_address_change_pct() - (-10.0)).abs() < 0.01);
        
        // Zero previous
        let metrics_zero = NetworkMetrics {
            active_addresses_prev_24h: 0,
            ..create_test_metrics("SOL", 100_000, 0)
        };
        assert_eq!(metrics_zero.active_address_change_pct(), 0.0);
    }
    
    #[test]
    fn test_activity_score() {
        // Positive change
        let metrics = create_test_metrics("BTC", 1_200_000, 1_000_000);
        let score = metrics.activity_score();
        assert!(score > 0.5); // Above neutral
        
        // Negative change
        let metrics_down = create_test_metrics("ETH", 800_000, 1_000_000);
        let score_down = metrics_down.activity_score();
        assert!(score_down < 0.5); // Below neutral
        
        // No change
        let metrics_flat = create_test_metrics("SOL", 1_000_000, 1_000_000);
        let score_flat = metrics_flat.activity_score();
        assert!((score_flat - 0.5).abs() < 0.01); // Near neutral
    }
    
    #[test]
    fn test_is_congested() {
        let metrics = NetworkMetrics {
            avg_gas_price: Some(50.0 * 1e9), // 50 gwei
            ..create_test_metrics("ETH", 1_000_000, 1_000_000)
        };
        
        assert!(metrics.is_congested(30.0)); // Threshold 30 gwei
        assert!(!metrics.is_congested(100.0)); // Threshold 100 gwei
        
        // No gas data
        let metrics_no_gas = NetworkMetrics {
            avg_gas_price: None,
            ..create_test_metrics("BTC", 1_000_000, 1_000_000)
        };
        assert!(!metrics_no_gas.is_congested(30.0));
    }
    
    #[tokio::test]
    async fn test_update_and_get_metrics() {
        let collector = NetworkMetricsCollector::new();
        
        let metrics = create_test_metrics("BTC", 1_100_000, 1_000_000);
        collector.update_metrics("BTC", metrics.clone()).await;
        
        let latest = collector.get_latest("BTC").await;
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().active_addresses_24h, 1_100_000);
        
        // Case insensitive
        let latest_lower = collector.get_latest("btc").await;
        assert!(latest_lower.is_some());
    }
    
    #[tokio::test]
    async fn test_get_signal() {
        let collector = NetworkMetricsCollector::new();
        
        // 10% increase in active addresses
        let metrics = create_test_metrics("ETH", 1_100_000, 1_000_000);
        collector.update_metrics("ETH", metrics).await;
        
        let (signal, change) = collector.get_signal("ETH").await;
        
        assert!(signal > 0.0); // Bullish
        assert!((change - 10.0).abs() < 0.01);
    }
    
    #[tokio::test]
    async fn test_get_signal_bearish() {
        let collector = NetworkMetricsCollector::new();
        
        // 20% decrease
        let metrics = create_test_metrics("SOL", 800_000, 1_000_000);
        collector.update_metrics("SOL", metrics).await;
        
        let (signal, change) = collector.get_signal("SOL").await;
        
        assert!(signal < 0.0); // Bearish
        assert!(change < -10.0);
    }
    
    #[tokio::test]
    async fn test_get_history() {
        let collector = NetworkMetricsCollector::new();
        
        // Add multiple metrics
        for i in 0..3 {
            let metrics = create_test_metrics("BTC", 1_000_000 + i * 10_000, 1_000_000);
            collector.update_metrics("BTC", metrics).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        
        let history = collector.get_history("BTC", 24).await;
        assert_eq!(history.len(), 3);
    }
    
    #[tokio::test]
    async fn test_calculate_trend() {
        let collector = NetworkMetricsCollector::new();
        
        // First data point
        let metrics1 = NetworkMetrics {
            active_addresses_24h: 1_000_000,
            active_addresses_prev_24h: 950_000,
            transaction_volume_usd_24h: 10_000_000_000.0,
            avg_gas_price: Some(20.0 * 1e9),
            ..create_test_metrics("ETH", 1_000_000, 950_000)
        };
        collector.update_metrics("ETH", metrics1).await;
        
        // Later data point
        let metrics2 = NetworkMetrics {
            active_addresses_24h: 1_200_000,
            active_addresses_prev_24h: 1_000_000,
            transaction_volume_usd_24h: 12_000_000_000.0,
            avg_gas_price: Some(30.0 * 1e9),
            ..create_test_metrics("ETH", 1_200_000, 1_000_000)
        };
        collector.update_metrics("ETH", metrics2).await;
        
        let trend = collector.calculate_trend("ETH", 24).await;
        
        assert_eq!(trend.asset, "ETH");
        assert!(trend.address_trend_pct > 0.0); // Addresses increased
        assert!(trend.volume_trend_pct > 0.0); // Volume increased
        assert!(trend.gas_trend_pct.unwrap() > 0.0); // Gas increased
        assert_eq!(trend.data_points, 2);
    }
    
    #[tokio::test]
    async fn test_empty_asset() {
        let collector = NetworkMetricsCollector::new();
        
        let (signal, change) = collector.get_signal("UNKNOWN").await;
        assert_eq!(signal, 0.0);
        assert_eq!(change, 0.0);
        
        let latest = collector.get_latest("UNKNOWN").await;
        assert!(latest.is_none());
    }
    
    #[tokio::test]
    async fn test_get_summary() {
        let collector = NetworkMetricsCollector::new();
        
        // BTC - bullish (10% up)
        collector.update_metrics("BTC", create_test_metrics("BTC", 1_100_000, 1_000_000)).await;
        
        // ETH - bearish (10% down)
        collector.update_metrics("ETH", create_test_metrics("ETH", 900_000, 1_000_000)).await;
        
        // SOL - neutral
        collector.update_metrics("SOL", create_test_metrics("SOL", 500_000, 500_000)).await;
        
        let summary = collector.get_summary().await;
        
        assert_eq!(summary.assets_tracked, 3);
        assert_eq!(summary.bullish_assets, 1);
        assert_eq!(summary.bearish_assets, 1);
        assert!(summary.total_active_addresses > 0);
    }
    
    #[tokio::test]
    async fn test_clear_old_metrics() {
        let mut collector = NetworkMetricsCollector::new();
        collector.retention_hours = 1;
        
        // This would need time manipulation to properly test
        // For now just verify it doesn't panic
        collector.clear_old_metrics().await;
    }
    
    #[test]
    fn test_network_trend_default() {
        let trend = NetworkTrend::default();
        
        assert_eq!(trend.address_trend_pct, 0.0);
        assert_eq!(trend.volume_trend_pct, 0.0);
        assert!(trend.gas_trend_pct.is_none());
    }
}
