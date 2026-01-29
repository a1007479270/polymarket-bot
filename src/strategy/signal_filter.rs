//! Signal filtering and deduplication
//!
//! Ensures high-quality trades by:
//! 1. Deduplicating - only one trade per market per time window
//! 2. Signal fusion - requiring multiple indicators to agree
//! 3. Time filtering - only trading near market close

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::strategy::trend_detector::{Trend, TrendSignal};

/// Tracks which markets have been traded recently
pub struct TradeDeduplicator {
    /// Market ID -> last trade timestamp
    traded_markets: RwLock<HashMap<String, DateTime<Utc>>>,
    /// Cooldown period before trading same market again
    cooldown_minutes: i64,
}

impl TradeDeduplicator {
    pub fn new(cooldown_minutes: i64) -> Self {
        Self {
            traded_markets: RwLock::new(HashMap::new()),
            cooldown_minutes,
        }
    }

    /// Check if we can trade this market (not traded recently)
    /// For crypto markets (detected by question content), use shorter cooldown
    pub fn can_trade(&self, market_id: &str) -> bool {
        let markets = self.traded_markets.read().unwrap();
        match markets.get(market_id) {
            Some(last_trade) => {
                let elapsed = Utc::now() - *last_trade;
                elapsed.num_minutes() >= self.cooldown_minutes
            }
            None => true,
        }
    }
    
    /// Check with dynamic cooldown based on market type
    pub fn can_trade_dynamic(&self, market_id: &str, is_crypto: bool) -> bool {
        let markets = self.traded_markets.read().unwrap();
        match markets.get(market_id) {
            Some(last_trade) => {
                let elapsed = Utc::now() - *last_trade;
                // Crypto: 2 min cooldown (allow fast trading)
                // Politics: 15 min cooldown (slower markets)
                let cooldown = if is_crypto { 2 } else { self.cooldown_minutes };
                elapsed.num_minutes() >= cooldown
            }
            None => true,
        }
    }

    /// Mark a market as traded
    pub fn mark_traded(&self, market_id: &str) {
        let mut markets = self.traded_markets.write().unwrap();
        markets.insert(market_id.to_string(), Utc::now());
    }

    /// Clean up old entries (call periodically)
    pub fn cleanup(&self) {
        let mut markets = self.traded_markets.write().unwrap();
        let cutoff = Utc::now() - chrono::Duration::minutes(self.cooldown_minutes * 2);
        markets.retain(|_, v| *v > cutoff);
    }

    /// Get number of markets traded in current session
    pub fn traded_count(&self) -> usize {
        self.traded_markets.read().unwrap().len()
    }
}

/// Signal fusion - combines multiple signal sources
pub struct SignalFusion {
    /// Minimum trend confidence to consider
    min_trend_confidence: Decimal,
    /// Minimum realtime momentum to consider
    min_momentum: Decimal,
    /// Require both signals to agree on direction
    require_agreement: bool,
}

impl Default for SignalFusion {
    fn default() -> Self {
        Self {
            min_trend_confidence: dec!(0.60),  // 60% confidence minimum
            min_momentum: dec!(0.001),          // 0.1% momentum minimum
            require_agreement: true,
        }
    }
}

impl SignalFusion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure fusion parameters
    pub fn with_thresholds(
        min_trend_confidence: Decimal,
        min_momentum: Decimal,
        require_agreement: bool,
    ) -> Self {
        Self {
            min_trend_confidence,
            min_momentum,
            require_agreement,
        }
    }

    /// Evaluate if signals agree and are strong enough
    pub fn evaluate(
        &self,
        trend_signal: Option<&TrendSignal>,
        realtime_momentum: Option<Decimal>,
    ) -> FusionResult {
        let trend_direction = trend_signal.and_then(|t| {
            if t.confidence >= self.min_trend_confidence {
                match t.trend {
                    Trend::StrongUp | Trend::WeakUp => Some(Direction::Up),
                    Trend::StrongDown | Trend::WeakDown => Some(Direction::Down),
                    Trend::Neutral => None,
                }
            } else {
                None
            }
        });

        let momentum_direction = realtime_momentum.and_then(|m| {
            if m.abs() >= self.min_momentum {
                if m > Decimal::ZERO {
                    Some(Direction::Up)
                } else {
                    Some(Direction::Down)
                }
            } else {
                None
            }
        });

        // Calculate combined confidence
        let trend_conf = trend_signal.map(|t| t.confidence).unwrap_or(dec!(0));
        let momentum_conf = realtime_momentum
            .map(|m| (m.abs() * dec!(100)).min(dec!(1)))
            .unwrap_or(dec!(0));

        let combined_confidence = (trend_conf + momentum_conf) / dec!(2);

        // Check agreement
        match (trend_direction, momentum_direction) {
            (Some(t), Some(m)) if t == m => {
                // Both agree!
                FusionResult {
                    should_trade: true,
                    direction: Some(t),
                    confidence: combined_confidence,
                    reason: format!("Signals agree: {:?} (trend {:.0}%, momentum {:.2}%)", 
                        t, trend_conf * dec!(100), momentum_conf * dec!(100)),
                }
            }
            (Some(t), Some(m)) => {
                // Disagreement
                FusionResult {
                    should_trade: false,
                    direction: None,
                    confidence: dec!(0),
                    reason: format!("Signal conflict: trend={:?}, momentum={:?}", t, m),
                }
            }
            (Some(t), None) if !self.require_agreement => {
                // Only trend signal, allowed if not requiring agreement
                FusionResult {
                    should_trade: trend_conf >= dec!(0.70), // Higher bar for single signal
                    direction: Some(t),
                    confidence: trend_conf,
                    reason: format!("Trend only: {:?} @ {:.0}%", t, trend_conf * dec!(100)),
                }
            }
            (None, Some(m)) if !self.require_agreement => {
                // Only momentum signal
                FusionResult {
                    should_trade: momentum_conf >= dec!(0.50),
                    direction: Some(m),
                    confidence: momentum_conf,
                    reason: format!("Momentum only: {:?} @ {:.2}%", m, momentum_conf * dec!(100)),
                }
            }
            _ => {
                // Not enough signals
                FusionResult {
                    should_trade: false,
                    direction: None,
                    confidence: dec!(0),
                    reason: "Insufficient signals".to_string(),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Down,
}

#[derive(Debug)]
pub struct FusionResult {
    pub should_trade: bool,
    pub direction: Option<Direction>,
    pub confidence: Decimal,
    pub reason: String,
}

/// Time-based filter - only trade near market close
pub struct TimeFilter {
    /// Only trade within X minutes of market close
    max_minutes_before_close: u32,
    /// Minimum minutes before close (don't trade too late)
    min_minutes_before_close: u32,
}

impl Default for TimeFilter {
    fn default() -> Self {
        Self {
            max_minutes_before_close: 10, // Enter up to 10 min before close
            min_minutes_before_close: 1,  // But not in last minute
        }
    }
}

impl TimeFilter {
    pub fn new(max_before: u32, min_before: u32) -> Self {
        Self {
            max_minutes_before_close: max_before,
            min_minutes_before_close: min_before,
        }
    }

    /// Check if current time is within trading window
    pub fn is_trading_window(&self, market_close: DateTime<Utc>) -> bool {
        let now = Utc::now();
        let time_to_close = market_close - now;
        let minutes = time_to_close.num_minutes();

        minutes >= self.min_minutes_before_close as i64
            && minutes <= self.max_minutes_before_close as i64
    }
}

/// Combined filter that applies all rules
pub struct SignalFilter {
    pub deduplicator: TradeDeduplicator,
    pub fusion: SignalFusion,
    pub time_filter: TimeFilter,
}

impl SignalFilter {
    pub fn new() -> Self {
        Self {
            deduplicator: TradeDeduplicator::new(15), // 15 min cooldown per market
            fusion: SignalFusion::new(),
            time_filter: TimeFilter::default(),
        }
    }

    /// Apply all filters to determine if we should trade
    pub fn should_trade(
        &self,
        market_id: &str,
        trend_signal: Option<&TrendSignal>,
        realtime_momentum: Option<Decimal>,
        market_close: Option<DateTime<Utc>>,
    ) -> FilterResult {
        // Check deduplication
        if !self.deduplicator.can_trade(market_id) {
            return FilterResult {
                should_trade: false,
                reason: "Market recently traded (cooldown)".to_string(),
                fusion_result: None,
            };
        }

        // Check time window if market close is known
        if let Some(close) = market_close {
            if !self.time_filter.is_trading_window(close) {
                return FilterResult {
                    should_trade: false,
                    reason: "Outside trading window".to_string(),
                    fusion_result: None,
                };
            }
        }

        // Check signal fusion
        let fusion_result = self.fusion.evaluate(trend_signal, realtime_momentum);
        
        if fusion_result.should_trade {
            self.deduplicator.mark_traded(market_id);
        }

        FilterResult {
            should_trade: fusion_result.should_trade,
            reason: fusion_result.reason.clone(),
            fusion_result: Some(fusion_result),
        }
    }

    /// Get stats
    pub fn stats(&self) -> FilterStats {
        FilterStats {
            markets_traded: self.deduplicator.traded_count(),
        }
    }
}

#[derive(Debug)]
pub struct FilterResult {
    pub should_trade: bool,
    pub reason: String,
    pub fusion_result: Option<FusionResult>,
}

pub struct FilterStats {
    pub markets_traded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplicator_basic() {
        let dedup = TradeDeduplicator::new(15);
        
        assert!(dedup.can_trade("market1"));
        dedup.mark_traded("market1");
        assert!(!dedup.can_trade("market1"));
        assert!(dedup.can_trade("market2"));
    }

    #[test]
    fn test_signal_fusion_agreement() {
        let fusion = SignalFusion::new();
        
        let trend = TrendSignal {
            trend: Trend::StrongUp,
            confidence: dec!(0.75),
            momentum: dec!(0.01),
            rsi: dec!(60),
            macd_signal: dec!(0.1),
            volume_ratio: dec!(1.5),
            reason: "test".to_string(),
        };
        
        let result = fusion.evaluate(Some(&trend), Some(dec!(0.002)));
        
        assert!(result.should_trade);
        assert_eq!(result.direction, Some(Direction::Up));
    }

    #[test]
    fn test_signal_fusion_conflict() {
        let fusion = SignalFusion::new();
        
        let trend = TrendSignal {
            trend: Trend::StrongUp,
            confidence: dec!(0.75),
            momentum: dec!(0.01),
            rsi: dec!(60),
            macd_signal: dec!(0.1),
            volume_ratio: dec!(1.5),
            reason: "test".to_string(),
        };
        
        // Trend says up, momentum says down
        let result = fusion.evaluate(Some(&trend), Some(dec!(-0.005)));
        
        assert!(!result.should_trade);
    }

    #[test]
    fn test_time_filter() {
        let filter = TimeFilter::new(10, 1);
        
        let close_in_5_min = Utc::now() + chrono::Duration::minutes(5);
        assert!(filter.is_trading_window(close_in_5_min));
        
        let close_in_30_sec = Utc::now() + chrono::Duration::seconds(30);
        assert!(!filter.is_trading_window(close_in_30_sec));
        
        let close_in_20_min = Utc::now() + chrono::Duration::minutes(20);
        assert!(!filter.is_trading_window(close_in_20_min));
    }
}
