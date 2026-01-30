//! Crypto 15-minute market monitor
//!
//! Complete monitoring system for BTC/ETH/XRP 15-minute Up/Down markets.
//! Includes:
//! - Dynamic market discovery (timestamp-based slugs)
//! - Real-time WebSocket price streaming
//! - Technical indicators (RSI, Stoch RSI)
//! - Spike detection
//!
//! Ported from Go cmd/crypto15m/main.go

use super::crypto_market::{CryptoMarket, CryptoMarketDiscovery, MarketInterval};
use super::indicators::{RSI, StochRSI, StochRSIResult, SignalType, SpikeDetector, SpikeConfig, analyze_signal};
use crate::client::websocket::{MarketStream, MarketUpdate};
use crate::error::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Market indicators for a single symbol
#[derive(Debug)]
pub struct MarketIndicators {
    /// Symbol name
    pub symbol: String,
    /// RSI(6)
    rsi: RSI,
    /// Stoch RSI(14,14,3,3)
    stoch_rsi: StochRSI,
    /// Spike detector for UP token
    spike_up: SpikeDetector,
    /// Spike detector for DOWN token
    spike_down: SpikeDetector,
    
    /// Latest RSI value
    pub last_rsi: f64,
    /// Latest Stoch K
    pub last_stoch_k: f64,
    /// Latest Stoch D
    pub last_stoch_d: f64,
    /// Previous Stoch K (for crossover detection)
    prev_stoch_k: f64,
    /// Previous Stoch D
    prev_stoch_d: f64,
    /// Current signal
    pub signal: SignalType,
    /// Price update count
    pub price_count: u64,
    /// Recent alerts
    alerts: Vec<String>,
}

impl MarketIndicators {
    /// Create new indicators for a symbol
    pub fn new(symbol: &str) -> Self {
        let spike_config = SpikeConfig {
            window_size: 30,
            spike_threshold: 0.02,  // 2%
            recovery_pct: 0.30,     // 30%
            cooldown_ms: 3000,      // 3 seconds
        };

        Self {
            symbol: symbol.to_string(),
            rsi: RSI::new(6),
            stoch_rsi: StochRSI::new(14, 14, 3, 3),
            spike_up: SpikeDetector::new(spike_config.clone()),
            spike_down: SpikeDetector::new(spike_config),
            last_rsi: 50.0,
            last_stoch_k: 50.0,
            last_stoch_d: 50.0,
            prev_stoch_k: 50.0,
            prev_stoch_d: 50.0,
            signal: SignalType::None,
            price_count: 0,
            alerts: Vec::new(),
        }
    }

    /// Update with UP token price
    pub fn update_up(&mut self, price: f64, bid: f64, ask: f64) {
        self.price_count += 1;

        // Save previous values for crossover detection
        self.prev_stoch_k = self.last_stoch_k;
        self.prev_stoch_d = self.last_stoch_d;

        // Update RSI and Stoch RSI with UP price
        self.last_rsi = self.rsi.update(price);
        let stoch = self.stoch_rsi.update(price);
        self.last_stoch_k = stoch.k;
        self.last_stoch_d = stoch.d;

        // Analyze signal
        if self.rsi.is_ready() && self.stoch_rsi.is_ready() {
            self.signal = analyze_signal(
                self.last_rsi,
                self.last_stoch_k,
                self.last_stoch_d,
                self.prev_stoch_k,
                self.prev_stoch_d,
            );
        }

        // Spike detection
        if let Some(spike) = self.spike_up.update(price, bid, ask) {
            let alert = format!(
                "[{}] {} UP {:?}: {:.2}% (recovery {:.0}%)",
                Utc::now().format("%H:%M:%S"),
                self.symbol,
                spike.spike_type,
                spike.spike_percent,
                spike.recovery_pct
            );
            self.add_alert(alert);
        }
    }

    /// Update with DOWN token price
    pub fn update_down(&mut self, price: f64, bid: f64, ask: f64) {
        // Spike detection for DOWN token
        if let Some(spike) = self.spike_down.update(price, bid, ask) {
            let alert = format!(
                "[{}] {} DOWN {:?}: {:.2}% (recovery {:.0}%)",
                Utc::now().format("%H:%M:%S"),
                self.symbol,
                spike.spike_type,
                spike.spike_percent,
                spike.recovery_pct
            );
            self.add_alert(alert);
        }
    }

    fn add_alert(&mut self, alert: String) {
        self.alerts.push(alert);
        // Keep only last 10 alerts
        if self.alerts.len() > 10 {
            self.alerts.remove(0);
        }
    }

    /// Get status
    pub fn get_status(&self) -> IndicatorStatus {
        IndicatorStatus {
            rsi: self.last_rsi,
            stoch_k: self.last_stoch_k,
            stoch_d: self.last_stoch_d,
            signal: self.signal,
            ready: self.rsi.is_ready() && self.stoch_rsi.is_ready(),
        }
    }

    /// Get spike statistics
    pub fn get_spike_stats(&self) -> SpikeStats {
        let (up_total, up_spikes) = self.spike_up.get_stats();
        let (down_total, down_spikes) = self.spike_down.get_stats();

        SpikeStats {
            up_total,
            up_spikes,
            down_total,
            down_spikes,
            volatility_up: self.spike_up.get_volatility(),
            volatility_down: self.spike_down.get_volatility(),
        }
    }

    /// Get recent alerts
    pub fn get_alerts(&self) -> Vec<String> {
        self.alerts.clone()
    }

    /// Reset indicators (for new window)
    pub fn reset(&mut self) {
        self.rsi.reset();
        self.stoch_rsi.reset();
        self.spike_up.reset();
        self.spike_down.reset();
        self.last_rsi = 50.0;
        self.last_stoch_k = 50.0;
        self.last_stoch_d = 50.0;
        self.prev_stoch_k = 50.0;
        self.prev_stoch_d = 50.0;
        self.signal = SignalType::None;
        self.price_count = 0;
        self.alerts.clear();
    }
}

/// Indicator status
#[derive(Debug, Clone)]
pub struct IndicatorStatus {
    pub rsi: f64,
    pub stoch_k: f64,
    pub stoch_d: f64,
    pub signal: SignalType,
    pub ready: bool,
}

/// Spike statistics
#[derive(Debug, Clone)]
pub struct SpikeStats {
    pub up_total: u64,
    pub up_spikes: u64,
    pub down_total: u64,
    pub down_spikes: u64,
    pub volatility_up: f64,
    pub volatility_down: f64,
}

/// Token info for mapping
struct TokenInfo {
    symbol: String,
    is_up: bool,
}

/// Complete Crypto 15m monitor
pub struct Crypto15mMonitor {
    discovery: CryptoMarketDiscovery,
    interval: MarketInterval,
    /// Symbol -> Indicators
    indicators: Arc<RwLock<HashMap<String, MarketIndicators>>>,
    /// Token ID -> Symbol info
    token_map: Arc<RwLock<HashMap<String, TokenInfo>>>,
    /// Current markets
    markets: Arc<RwLock<HashMap<String, CryptoMarket>>>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl Crypto15mMonitor {
    /// Create a new monitor
    pub fn new(gamma_url: &str, interval: MarketInterval) -> Self {
        Self {
            discovery: CryptoMarketDiscovery::new(gamma_url),
            interval,
            indicators: Arc::new(RwLock::new(HashMap::new())),
            token_map: Arc::new(RwLock::new(HashMap::new())),
            markets: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Discover and load current markets
    pub async fn load_markets(&self) -> Result<Vec<String>> {
        let markets = self.discovery.get_all_current_markets(self.interval).await;

        let mut token_ids = Vec::new();
        let mut indicators = self.indicators.write().await;
        let mut token_map = self.token_map.write().await;
        let mut market_map = self.markets.write().await;

        for market in markets {
            info!(
                "[Crypto15m] Loaded {} market: UP={:.3} DOWN={:.3} SUM={:.4}",
                market.symbol, market.up_price, market.down_price, market.sum
            );

            // Create indicators if not exists
            if !indicators.contains_key(&market.symbol) {
                indicators.insert(market.symbol.clone(), MarketIndicators::new(&market.symbol));
            }

            // Map tokens to symbols
            token_map.insert(
                market.up_token_id.clone(),
                TokenInfo {
                    symbol: market.symbol.clone(),
                    is_up: true,
                },
            );
            token_map.insert(
                market.down_token_id.clone(),
                TokenInfo {
                    symbol: market.symbol.clone(),
                    is_up: false,
                },
            );

            token_ids.push(market.up_token_id.clone());
            token_ids.push(market.down_token_id.clone());

            market_map.insert(market.symbol.clone(), market);
        }

        Ok(token_ids)
    }

    /// Process a price update from WebSocket
    pub async fn handle_update(&self, update: MarketUpdate) {
        let token_map = self.token_map.read().await;
        let info = match token_map.get(&update.token_id) {
            Some(i) => i,
            None => return,
        };

        let symbol = info.symbol.clone();
        let is_up = info.is_up;
        drop(token_map);

        // Get price (use ask as reference price)
        let price = match update.best_ask {
            Some(p) => p.to_string().parse::<f64>().unwrap_or(0.0),
            None => return,
        };
        let bid = update.best_bid
            .map(|p| p.to_string().parse::<f64>().unwrap_or(0.0))
            .unwrap_or(price);

        // Update indicators
        let mut indicators = self.indicators.write().await;
        if let Some(ind) = indicators.get_mut(&symbol) {
            if is_up {
                ind.update_up(price, bid, price);
            } else {
                ind.update_down(price, bid, price);
            }
        }

        // Update market prices
        let mut markets = self.markets.write().await;
        if let Some(market) = markets.get_mut(&symbol) {
            if is_up {
                market.up_price = rust_decimal::Decimal::try_from(price).unwrap_or_default();
            } else {
                market.down_price = rust_decimal::Decimal::try_from(price).unwrap_or_default();
            }
            market.sum = market.up_price + market.down_price;
            market.spread = rust_decimal::Decimal::ONE - market.sum;
        }
    }

    /// Get current status for all markets
    pub async fn get_status(&self) -> Vec<MarketStatus> {
        let markets = self.markets.read().await;
        let indicators = self.indicators.read().await;

        let mut result = Vec::new();

        for (symbol, market) in markets.iter() {
            let ind_status = indicators
                .get(symbol)
                .map(|i| i.get_status())
                .unwrap_or(IndicatorStatus {
                    rsi: 50.0,
                    stoch_k: 50.0,
                    stoch_d: 50.0,
                    signal: SignalType::None,
                    ready: false,
                });

            let spike_stats = indicators
                .get(symbol)
                .map(|i| i.get_spike_stats());

            result.push(MarketStatus {
                symbol: symbol.clone(),
                up_price: market.up_price.to_string().parse().unwrap_or(0.0),
                down_price: market.down_price.to_string().parse().unwrap_or(0.0),
                sum: market.sum.to_string().parse().unwrap_or(1.0),
                spread: market.spread.to_string().parse().unwrap_or(0.0),
                remaining: market.remaining(),
                indicators: ind_status,
                spikes: spike_stats,
            });
        }

        result
    }

    /// Get all recent alerts
    pub async fn get_all_alerts(&self) -> Vec<String> {
        let indicators = self.indicators.read().await;
        let mut all_alerts = Vec::new();

        for ind in indicators.values() {
            all_alerts.extend(ind.get_alerts());
        }

        all_alerts
    }

    /// Reset all indicators (call when switching to new window)
    pub async fn reset_indicators(&self) {
        let mut indicators = self.indicators.write().await;
        for ind in indicators.values_mut() {
            ind.reset();
        }
        info!("[Crypto15m] Indicators reset for new window");
    }

    /// Check if any market has arbitrage opportunity
    pub async fn check_arbitrage(&self, min_spread: f64) -> Vec<(String, f64)> {
        let markets = self.markets.read().await;
        let mut opportunities = Vec::new();

        for (symbol, market) in markets.iter() {
            let spread: f64 = market.spread.to_string().parse().unwrap_or(0.0);
            if spread > min_spread {
                opportunities.push((symbol.clone(), spread));
            }
        }

        opportunities
    }
}

/// Market status for display
#[derive(Debug, Clone)]
pub struct MarketStatus {
    pub symbol: String,
    pub up_price: f64,
    pub down_price: f64,
    pub sum: f64,
    pub spread: f64,
    pub remaining: Duration,
    pub indicators: IndicatorStatus,
    pub spikes: Option<SpikeStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_indicators() {
        let mut ind = MarketIndicators::new("BTC");

        // Simulate some price updates
        for i in 0..20 {
            let price = 0.5 + (i as f64 * 0.01).sin() * 0.1;
            ind.update_up(price, price - 0.01, price + 0.01);
        }

        assert!(ind.price_count > 0);
    }

    #[test]
    fn test_indicator_reset() {
        let mut ind = MarketIndicators::new("ETH");
        
        for i in 0..10 {
            ind.update_up(0.5 + i as f64 * 0.01, 0.49, 0.51);
        }

        assert!(ind.price_count > 0);
        
        ind.reset();
        assert_eq!(ind.price_count, 0);
        assert_eq!(ind.last_rsi, 50.0);
    }
}
