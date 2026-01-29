//! Exchange Flow Monitoring Module
//!
//! Tracks cryptocurrency flows into and out of exchanges.
//! - Inflow (deposits to exchanges): Potential selling pressure
//! - Outflow (withdrawals from exchanges): Accumulation signal
//!
//! Net flow is a key indicator for short-term price direction.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// Flow direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowDirection {
    /// Deposit to exchange
    Inflow,
    /// Withdrawal from exchange
    Outflow,
}

/// Exchange flow record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFlow {
    /// Asset symbol
    pub asset: String,
    /// Exchange name
    pub exchange: String,
    /// Flow direction
    pub direction: FlowDirection,
    /// Amount in USD
    pub amount_usd: f64,
    /// Amount in native units
    pub amount_native: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl ExchangeFlow {
    /// Get signed amount (negative for inflow, positive for outflow)
    pub fn signed_amount(&self) -> f64 {
        match self.direction {
            FlowDirection::Inflow => -self.amount_usd,
            FlowDirection::Outflow => self.amount_usd,
        }
    }
}

/// Aggregated flow signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowSignal {
    /// Asset symbol
    pub asset: String,
    /// Net flow in USD (positive = outflow dominant)
    pub net_flow_usd: f64,
    /// Total inflow in USD
    pub total_inflow_usd: f64,
    /// Total outflow in USD
    pub total_outflow_usd: f64,
    /// Signal strength (-1.0 to 1.0)
    pub signal: f64,
    /// Time window in hours
    pub window_hours: u32,
    /// Number of flow records
    pub record_count: u32,
}

/// Exchange flow tracker
pub struct ExchangeFlowTracker {
    /// Flow records by asset
    flows: Arc<RwLock<HashMap<String, Vec<ExchangeFlow>>>>,
    /// Retention period in hours
    retention_hours: u32,
    /// Signal thresholds
    thresholds: FlowThresholds,
}

/// Thresholds for signal generation
#[derive(Debug, Clone)]
pub struct FlowThresholds {
    /// Minimum net flow to generate signal (USD)
    pub min_net_flow_usd: f64,
    /// Strong signal threshold (USD)
    pub strong_signal_threshold_usd: f64,
    /// Time decay half-life in hours
    pub decay_half_life_hours: f64,
}

impl Default for FlowThresholds {
    fn default() -> Self {
        Self {
            min_net_flow_usd: 100_000.0,         // $100k minimum
            strong_signal_threshold_usd: 10_000_000.0, // $10M for strong signal
            decay_half_life_hours: 2.0,
        }
    }
}

impl ExchangeFlowTracker {
    /// Create a new exchange flow tracker
    pub fn new() -> Self {
        Self {
            flows: Arc::new(RwLock::new(HashMap::new())),
            retention_hours: 24,
            thresholds: FlowThresholds::default(),
        }
    }
    
    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: FlowThresholds) -> Self {
        Self {
            flows: Arc::new(RwLock::new(HashMap::new())),
            retention_hours: 24,
            thresholds,
        }
    }
    
    /// Add a flow record
    pub async fn add_flow(&self, flow: ExchangeFlow) {
        let mut flows = self.flows.write().await;
        flows
            .entry(flow.asset.to_uppercase())
            .or_insert_with(Vec::new)
            .push(flow);
    }
    
    /// Get aggregated signal for an asset
    /// Returns (signal, net_flow_usd)
    pub async fn get_signal(&self, asset: &str) -> (f64, f64) {
        let flows = self.flows.read().await;
        let asset_upper = asset.to_uppercase();
        
        let asset_flows = match flows.get(&asset_upper) {
            Some(f) => f,
            None => return (0.0, 0.0),
        };
        
        let now = Utc::now();
        let window_start = now - Duration::hours(4);
        
        // Calculate time-weighted net flow
        let mut weighted_net_flow = 0.0;
        let mut total_weight = 0.0;
        
        for flow in asset_flows.iter().filter(|f| f.timestamp > window_start) {
            let age_hours = (now - flow.timestamp).num_minutes() as f64 / 60.0;
            let weight = 0.5_f64.powf(age_hours / self.thresholds.decay_half_life_hours);
            
            weighted_net_flow += flow.signed_amount() * weight;
            total_weight += weight;
        }
        
        // Calculate raw net flow
        let net_flow: f64 = asset_flows.iter()
            .filter(|f| f.timestamp > window_start)
            .map(|f| f.signed_amount())
            .sum();
        
        // Convert to signal
        let signal = self.calculate_signal(weighted_net_flow, total_weight);
        
        (signal, net_flow)
    }
    
    /// Calculate signal from weighted net flow
    fn calculate_signal(&self, weighted_net_flow: f64, _total_weight: f64) -> f64 {
        if weighted_net_flow.abs() < self.thresholds.min_net_flow_usd {
            return 0.0;
        }
        
        // Normalize to [-1, 1] using threshold
        let normalized = weighted_net_flow / self.thresholds.strong_signal_threshold_usd;
        
        // Apply sigmoid-like scaling for smooth transition
        let signal = normalized.tanh();
        
        signal.clamp(-1.0, 1.0)
    }
    
    /// Get detailed flow signal
    pub async fn get_flow_signal(&self, asset: &str, window_hours: u32) -> FlowSignal {
        let flows = self.flows.read().await;
        let asset_upper = asset.to_uppercase();
        
        let window_start = Utc::now() - Duration::hours(window_hours as i64);
        
        let asset_flows = flows.get(&asset_upper);
        
        let (total_inflow, total_outflow, record_count) = asset_flows
            .map(|flows| {
                let filtered: Vec<_> = flows.iter()
                    .filter(|f| f.timestamp > window_start)
                    .collect();
                
                let inflow: f64 = filtered.iter()
                    .filter(|f| f.direction == FlowDirection::Inflow)
                    .map(|f| f.amount_usd)
                    .sum();
                
                let outflow: f64 = filtered.iter()
                    .filter(|f| f.direction == FlowDirection::Outflow)
                    .map(|f| f.amount_usd)
                    .sum();
                
                (inflow, outflow, filtered.len() as u32)
            })
            .unwrap_or((0.0, 0.0, 0));
        
        let net_flow = total_outflow - total_inflow;
        let signal = self.calculate_signal(net_flow, 1.0);
        
        FlowSignal {
            asset: asset_upper,
            net_flow_usd: net_flow,
            total_inflow_usd: total_inflow,
            total_outflow_usd: total_outflow,
            signal,
            window_hours,
            record_count,
        }
    }
    
    /// Get flows by exchange
    pub async fn get_flows_by_exchange(&self, asset: &str, hours: u32) -> HashMap<String, (f64, f64)> {
        let flows = self.flows.read().await;
        let asset_upper = asset.to_uppercase();
        let cutoff = Utc::now() - Duration::hours(hours as i64);
        
        let mut by_exchange: HashMap<String, (f64, f64)> = HashMap::new();
        
        if let Some(asset_flows) = flows.get(&asset_upper) {
            for flow in asset_flows.iter().filter(|f| f.timestamp > cutoff) {
                let entry = by_exchange.entry(flow.exchange.clone()).or_insert((0.0, 0.0));
                match flow.direction {
                    FlowDirection::Inflow => entry.0 += flow.amount_usd,
                    FlowDirection::Outflow => entry.1 += flow.amount_usd,
                }
            }
        }
        
        by_exchange
    }
    
    /// Clear old flow records
    pub async fn clear_old_flows(&self) {
        let mut flows = self.flows.write().await;
        let cutoff = Utc::now() - Duration::hours(self.retention_hours as i64);
        
        for asset_flows in flows.values_mut() {
            asset_flows.retain(|f| f.timestamp > cutoff);
        }
    }
    
    /// Get summary statistics
    pub async fn get_stats(&self) -> FlowStats {
        let flows = self.flows.read().await;
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);
        let twenty_four_hours_ago = now - Duration::hours(24);
        
        let mut total_inflow_24h = 0.0;
        let mut total_outflow_24h = 0.0;
        let mut records_1h = 0;
        let mut records_24h = 0;
        
        for asset_flows in flows.values() {
            for flow in asset_flows {
                if flow.timestamp > twenty_four_hours_ago {
                    records_24h += 1;
                    match flow.direction {
                        FlowDirection::Inflow => total_inflow_24h += flow.amount_usd,
                        FlowDirection::Outflow => total_outflow_24h += flow.amount_usd,
                    }
                }
                if flow.timestamp > one_hour_ago {
                    records_1h += 1;
                }
            }
        }
        
        FlowStats {
            total_inflow_24h_usd: total_inflow_24h,
            total_outflow_24h_usd: total_outflow_24h,
            net_flow_24h_usd: total_outflow_24h - total_inflow_24h,
            records_1h,
            records_24h,
        }
    }
}

impl Default for ExchangeFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow statistics
#[derive(Debug, Clone, Serialize)]
pub struct FlowStats {
    pub total_inflow_24h_usd: f64,
    pub total_outflow_24h_usd: f64,
    pub net_flow_24h_usd: f64,
    pub records_1h: u32,
    pub records_24h: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flow_signed_amount() {
        let inflow = ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 1_000_000.0,
            amount_native: 10.0,
            timestamp: Utc::now(),
        };
        assert!(inflow.signed_amount() < 0.0);
        
        let outflow = ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 1_000_000.0,
            amount_native: 10.0,
            timestamp: Utc::now(),
        };
        assert!(outflow.signed_amount() > 0.0);
    }
    
    #[tokio::test]
    async fn test_add_and_get_signal() {
        let tracker = ExchangeFlowTracker::new();
        
        // Add large outflow (bullish)
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 15_000_000.0,
            amount_native: 150.0,
            timestamp: Utc::now(),
        }).await;
        
        let (signal, net_flow) = tracker.get_signal("BTC").await;
        
        assert!(signal > 0.0); // Bullish
        assert!(net_flow > 0.0); // Positive net flow
    }
    
    #[tokio::test]
    async fn test_mixed_flows() {
        let tracker = ExchangeFlowTracker::new();
        
        // Add inflow
        tracker.add_flow(ExchangeFlow {
            asset: "ETH".to_string(),
            exchange: "Coinbase".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 5_000_000.0,
            amount_native: 2000.0,
            timestamp: Utc::now(),
        }).await;
        
        // Add outflow
        tracker.add_flow(ExchangeFlow {
            asset: "ETH".to_string(),
            exchange: "Kraken".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 3_000_000.0,
            amount_native: 1200.0,
            timestamp: Utc::now(),
        }).await;
        
        let (signal, net_flow) = tracker.get_signal("ETH").await;
        
        // Net inflow = bearish
        assert!(signal < 0.0);
        assert!(net_flow < 0.0);
    }
    
    #[tokio::test]
    async fn test_flow_signal_detailed() {
        let tracker = ExchangeFlowTracker::new();
        
        tracker.add_flow(ExchangeFlow {
            asset: "SOL".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 2_000_000.0,
            amount_native: 10000.0,
            timestamp: Utc::now(),
        }).await;
        
        tracker.add_flow(ExchangeFlow {
            asset: "SOL".to_string(),
            exchange: "OKX".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 1_000_000.0,
            amount_native: 5000.0,
            timestamp: Utc::now(),
        }).await;
        
        let signal = tracker.get_flow_signal("SOL", 4).await;
        
        assert_eq!(signal.total_inflow_usd, 2_000_000.0);
        assert_eq!(signal.total_outflow_usd, 1_000_000.0);
        assert_eq!(signal.net_flow_usd, -1_000_000.0);
        assert_eq!(signal.record_count, 2);
        assert!(signal.signal < 0.0);
    }
    
    #[tokio::test]
    async fn test_flows_by_exchange() {
        let tracker = ExchangeFlowTracker::new();
        
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 1_000_000.0,
            amount_native: 10.0,
            timestamp: Utc::now(),
        }).await;
        
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 500_000.0,
            amount_native: 5.0,
            timestamp: Utc::now(),
        }).await;
        
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Coinbase".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 2_000_000.0,
            amount_native: 20.0,
            timestamp: Utc::now(),
        }).await;
        
        let by_exchange = tracker.get_flows_by_exchange("BTC", 4).await;
        
        assert_eq!(by_exchange.get("Binance"), Some(&(1_000_000.0, 500_000.0)));
        assert_eq!(by_exchange.get("Coinbase"), Some(&(0.0, 2_000_000.0)));
    }
    
    #[tokio::test]
    async fn test_clear_old_flows() {
        let mut tracker = ExchangeFlowTracker::new();
        tracker.retention_hours = 1;
        
        // Old flow
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 1_000_000.0,
            amount_native: 10.0,
            timestamp: Utc::now() - Duration::hours(2),
        }).await;
        
        // Recent flow
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 500_000.0,
            amount_native: 5.0,
            timestamp: Utc::now(),
        }).await;
        
        tracker.clear_old_flows().await;
        
        let signal = tracker.get_flow_signal("BTC", 24).await;
        assert_eq!(signal.record_count, 1);
    }
    
    #[tokio::test]
    async fn test_get_stats() {
        let tracker = ExchangeFlowTracker::new();
        
        tracker.add_flow(ExchangeFlow {
            asset: "BTC".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Inflow,
            amount_usd: 5_000_000.0,
            amount_native: 50.0,
            timestamp: Utc::now(),
        }).await;
        
        tracker.add_flow(ExchangeFlow {
            asset: "ETH".to_string(),
            exchange: "Coinbase".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 3_000_000.0,
            amount_native: 1000.0,
            timestamp: Utc::now(),
        }).await;
        
        let stats = tracker.get_stats().await;
        
        assert_eq!(stats.total_inflow_24h_usd, 5_000_000.0);
        assert_eq!(stats.total_outflow_24h_usd, 3_000_000.0);
        assert_eq!(stats.net_flow_24h_usd, -2_000_000.0);
        assert_eq!(stats.records_1h, 2);
        assert_eq!(stats.records_24h, 2);
    }
    
    #[tokio::test]
    async fn test_case_insensitive_asset() {
        let tracker = ExchangeFlowTracker::new();
        
        tracker.add_flow(ExchangeFlow {
            asset: "btc".to_string(),
            exchange: "Binance".to_string(),
            direction: FlowDirection::Outflow,
            amount_usd: 5_000_000.0,
            amount_native: 50.0,
            timestamp: Utc::now(),
        }).await;
        
        let (signal1, _) = tracker.get_signal("BTC").await;
        let (signal2, _) = tracker.get_signal("btc").await;
        let (signal3, _) = tracker.get_signal("Btc").await;
        
        assert_eq!(signal1, signal2);
        assert_eq!(signal2, signal3);
    }
    
    #[tokio::test]
    async fn test_empty_asset() {
        let tracker = ExchangeFlowTracker::new();
        
        let (signal, net_flow) = tracker.get_signal("UNKNOWN").await;
        
        assert_eq!(signal, 0.0);
        assert_eq!(net_flow, 0.0);
    }
    
    #[test]
    fn test_calculate_signal_below_threshold() {
        let tracker = ExchangeFlowTracker::new();
        
        // Below minimum threshold
        let signal = tracker.calculate_signal(50_000.0, 1.0);
        assert_eq!(signal, 0.0);
    }
    
    #[test]
    fn test_calculate_signal_strong() {
        let tracker = ExchangeFlowTracker::new();
        
        // Strong outflow signal
        let signal = tracker.calculate_signal(20_000_000.0, 1.0);
        assert!(signal > 0.9);
        
        // Strong inflow signal
        let signal_neg = tracker.calculate_signal(-20_000_000.0, 1.0);
        assert!(signal_neg < -0.9);
    }
}
