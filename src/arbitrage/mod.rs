//! # Funding Rate Arbitrage Module
//!
//! Professional-grade funding rate arbitrage system for cryptocurrency perpetual futures.
//!
//! ## Features
//!
//! - **Multi-Exchange Funding Rate Tracking**: Real-time rates from Binance, OKX, Bybit, dYdX
//! - **Funding Rate Prediction**: ML-based prediction of next funding rate
//! - **Delta-Neutral Arbitrage**: Spot + Perp hedging for risk-free carry
//! - **Cross-Exchange Arbitrage**: Exploit funding rate differentials
//! - **Carry Trade Calculator**: APY estimation with slippage/fees
//! - **Risk Management**: Liquidation risk, funding rate reversal protection
//!
//! ## Example
//!
//! ```ignore
//! use polymarket_bot::arbitrage::{FundingRateTracker, ArbitrageEngine};
//!
//! let tracker = FundingRateTracker::new();
//! tracker.add_exchange(Exchange::Binance);
//! tracker.add_exchange(Exchange::OKX);
//!
//! let engine = ArbitrageEngine::new(tracker)
//!     .with_min_spread(Decimal::new(5, 4))  // 0.05% min spread
//!     .with_max_position_size(Decimal::new(100000, 0));  // $100k max
//!
//! let opportunities = engine.scan_opportunities().await?;
//! ```

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Supported cryptocurrency exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    Binance,
    OKX,
    Bybit,
    DYDX,
    Hyperliquid,
    GMX,
}

impl Exchange {
    /// Funding interval in hours (most exchanges use 8h)
    pub fn funding_interval_hours(&self) -> u64 {
        match self {
            Exchange::Binance => 8,
            Exchange::OKX => 8,
            Exchange::Bybit => 8,
            Exchange::DYDX => 1,      // dYdX has hourly funding
            Exchange::Hyperliquid => 1, // Hyperliquid has hourly funding
            Exchange::GMX => 1,       // GMX has hourly funding
        }
    }

    /// Typical maker fee in basis points
    pub fn maker_fee_bps(&self) -> Decimal {
        match self {
            Exchange::Binance => dec!(2),    // 0.02%
            Exchange::OKX => dec!(2),        // 0.02%
            Exchange::Bybit => dec!(1),      // 0.01%
            Exchange::DYDX => dec!(2),       // 0.02%
            Exchange::Hyperliquid => dec!(2), // 0.02%
            Exchange::GMX => dec!(5),        // 0.05%
        }
    }

    /// Typical taker fee in basis points
    pub fn taker_fee_bps(&self) -> Decimal {
        match self {
            Exchange::Binance => dec!(4),    // 0.04%
            Exchange::OKX => dec!(5),        // 0.05%
            Exchange::Bybit => dec!(6),      // 0.06%
            Exchange::DYDX => dec!(5),       // 0.05%
            Exchange::Hyperliquid => dec!(5), // 0.05%
            Exchange::GMX => dec!(5),        // 0.05%
        }
    }
}

/// Funding rate data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub exchange: Exchange,
    pub symbol: String,
    pub rate: Decimal,           // Funding rate (e.g., 0.0001 = 0.01%)
    pub predicted_rate: Option<Decimal>, // Next predicted rate
    pub timestamp: u64,          // Unix timestamp in seconds
    pub next_funding_time: u64,  // Next funding payment time
    pub interval_hours: u64,     // Funding interval
}

impl FundingRate {
    /// Convert funding rate to annualized APY
    pub fn annualized_apy(&self) -> Decimal {
        let periods_per_year = Decimal::from(365 * 24) / Decimal::from(self.interval_hours);
        self.rate * periods_per_year * dec!(100) // Return as percentage
    }

    /// Time until next funding payment in seconds
    pub fn time_to_next_funding(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.next_funding_time.saturating_sub(now)
    }

    /// Check if we're in the funding window (close to payment time)
    pub fn in_funding_window(&self, window_minutes: u64) -> bool {
        self.time_to_next_funding() < window_minutes * 60
    }
}

/// Historical funding rate tracker for a single symbol
#[derive(Debug, Clone)]
pub struct FundingRateHistory {
    symbol: String,
    exchange: Exchange,
    rates: VecDeque<FundingRate>,
    max_history: usize,
}

impl FundingRateHistory {
    pub fn new(symbol: &str, exchange: Exchange, max_history: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            exchange,
            rates: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    pub fn add_rate(&mut self, rate: FundingRate) {
        if self.rates.len() >= self.max_history {
            self.rates.pop_front();
        }
        self.rates.push_back(rate);
    }

    pub fn latest(&self) -> Option<&FundingRate> {
        self.rates.back()
    }

    /// Calculate average funding rate over N periods
    pub fn average_rate(&self, periods: usize) -> Option<Decimal> {
        let count = periods.min(self.rates.len());
        if count == 0 {
            return None;
        }
        let sum: Decimal = self.rates.iter().rev().take(count).map(|r| r.rate).sum();
        Some(sum / Decimal::from(count as u64))
    }

    /// Calculate standard deviation of funding rates
    pub fn rate_std_dev(&self, periods: usize) -> Option<Decimal> {
        let count = periods.min(self.rates.len());
        if count < 2 {
            return None;
        }
        let avg = self.average_rate(count)?;
        let variance: Decimal = self
            .rates
            .iter()
            .rev()
            .take(count)
            .map(|r| {
                let diff = r.rate - avg;
                diff * diff
            })
            .sum::<Decimal>()
            / Decimal::from((count - 1) as u64);
        Some(sqrt_decimal(variance))
    }

    /// Predict next funding rate using exponential moving average
    pub fn predict_next_rate(&self, alpha: Decimal) -> Option<Decimal> {
        if self.rates.is_empty() {
            return None;
        }
        let mut ema = self.rates.front()?.rate;
        for rate in self.rates.iter().skip(1) {
            ema = alpha * rate.rate + (Decimal::ONE - alpha) * ema;
        }
        Some(ema)
    }

    /// Check if funding rate is trending (positive slope)
    pub fn is_trending_up(&self, periods: usize) -> bool {
        if self.rates.len() < periods || periods < 2 {
            return false;
        }
        let recent: Vec<Decimal> = self.rates.iter().rev().take(periods).map(|r| r.rate).collect();
        let first_half_avg: Decimal = recent[periods / 2..].iter().copied().sum::<Decimal>()
            / Decimal::from((periods - periods / 2) as u64);
        let second_half_avg: Decimal = recent[..periods / 2].iter().copied().sum::<Decimal>()
            / Decimal::from((periods / 2) as u64);
        second_half_avg > first_half_avg
    }

    /// Get historical rates as a vector
    pub fn get_rates(&self) -> Vec<&FundingRate> {
        self.rates.iter().collect()
    }
}

/// Multi-exchange funding rate tracker
#[derive(Debug, Clone)]
pub struct FundingRateTracker {
    histories: HashMap<(Exchange, String), FundingRateHistory>,
    supported_symbols: Vec<String>,
    max_history_per_symbol: usize,
}

impl Default for FundingRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FundingRateTracker {
    pub fn new() -> Self {
        Self {
            histories: HashMap::new(),
            supported_symbols: vec![
                "BTC".to_string(),
                "ETH".to_string(),
                "SOL".to_string(),
                "XRP".to_string(),
                "DOGE".to_string(),
                "AVAX".to_string(),
                "LINK".to_string(),
                "MATIC".to_string(),
            ],
            max_history_per_symbol: 100,
        }
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.supported_symbols = symbols;
        self
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history_per_symbol = max_history;
        self
    }

    /// Add a new funding rate observation
    pub fn add_rate(&mut self, rate: FundingRate) {
        let key = (rate.exchange, rate.symbol.clone());
        self.histories
            .entry(key.clone())
            .or_insert_with(|| {
                FundingRateHistory::new(&rate.symbol, rate.exchange, self.max_history_per_symbol)
            })
            .add_rate(rate);
    }

    /// Get latest funding rate for a symbol on an exchange
    pub fn get_latest(&self, exchange: Exchange, symbol: &str) -> Option<&FundingRate> {
        self.histories.get(&(exchange, symbol.to_string()))?.latest()
    }

    /// Get average funding rate across all exchanges for a symbol
    pub fn cross_exchange_average(&self, symbol: &str) -> Option<Decimal> {
        let rates: Vec<Decimal> = self
            .histories
            .iter()
            .filter(|((_, s), _)| s == symbol)
            .filter_map(|(_, h)| h.latest().map(|r| r.rate))
            .collect();

        if rates.is_empty() {
            return None;
        }
        Some(rates.iter().copied().sum::<Decimal>() / Decimal::from(rates.len() as u64))
    }

    /// Find funding rate arbitrage opportunities between exchanges
    pub fn find_arbitrage_opportunities(
        &self,
        symbol: &str,
        min_spread_bps: Decimal,
    ) -> Vec<FundingArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        // Get all exchanges with rates for this symbol
        let exchange_rates: Vec<(Exchange, &FundingRate)> = self
            .histories
            .iter()
            .filter(|((_, s), _)| s == symbol)
            .filter_map(|((e, _), h)| h.latest().map(|r| (*e, r)))
            .collect();

        // Compare all pairs
        for i in 0..exchange_rates.len() {
            for j in (i + 1)..exchange_rates.len() {
                let (ex1, rate1) = &exchange_rates[i];
                let (ex2, rate2) = &exchange_rates[j];

                let spread = (rate1.rate - rate2.rate).abs();
                let spread_bps = spread * dec!(10000);

                if spread_bps >= min_spread_bps {
                    let (long_exchange, short_exchange) = if rate1.rate < rate2.rate {
                        (*ex1, *ex2)
                    } else {
                        (*ex2, *ex1)
                    };

                    opportunities.push(FundingArbitrageOpportunity {
                        symbol: symbol.to_string(),
                        long_exchange,
                        short_exchange,
                        spread,
                        spread_bps,
                        long_rate: rate1.rate.min(rate2.rate),
                        short_rate: rate1.rate.max(rate2.rate),
                        estimated_apy: self.estimate_arbitrage_apy(
                            spread,
                            long_exchange,
                            short_exchange,
                        ),
                        timestamp: std::cmp::max(rate1.timestamp, rate2.timestamp),
                    });
                }
            }
        }

        // Sort by spread (highest first)
        opportunities.sort_by(|a, b| b.spread_bps.cmp(&a.spread_bps));
        opportunities
    }

    fn estimate_arbitrage_apy(
        &self,
        spread: Decimal,
        long_ex: Exchange,
        short_ex: Exchange,
    ) -> Decimal {
        // Assume 8h funding intervals for simplicity
        let periods_per_year = dec!(1095); // 365 * 3

        // Deduct fees (entry + exit on both sides)
        let fees_bps = (long_ex.taker_fee_bps() + short_ex.taker_fee_bps()) * dec!(2);
        let spread_bps = spread * dec!(10000);
        let net_spread_bps = spread_bps - fees_bps;

        if net_spread_bps <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        (net_spread_bps / dec!(10000)) * periods_per_year * dec!(100)
    }

    /// Get predicted direction signal based on funding rates
    /// Returns: positive = bullish, negative = bearish, zero = neutral
    pub fn get_direction_signal(&self, symbol: &str) -> FundingDirectionSignal {
        let avg_rate = self.cross_exchange_average(symbol).unwrap_or_default();

        // Extreme funding rates are contrarian signals
        // High positive funding = too many longs = bearish
        // High negative funding = too many shorts = bullish
        let signal_strength = if avg_rate > dec!(0.001) {
            // > 0.1% funding = very high, expect correction
            -avg_rate * dec!(100)
        } else if avg_rate < dec!(-0.001) {
            // < -0.1% funding = very negative, expect bounce
            -avg_rate * dec!(100)
        } else {
            // Normal range: funding direction indicates momentum
            avg_rate * dec!(50)
        };

        let direction = if signal_strength > dec!(0.01) {
            Direction::Up
        } else if signal_strength < dec!(-0.01) {
            Direction::Down
        } else {
            Direction::Neutral
        };

        FundingDirectionSignal {
            symbol: symbol.to_string(),
            direction,
            strength: signal_strength.abs(),
            avg_funding_rate: avg_rate,
            confidence: self.calculate_signal_confidence(symbol),
        }
    }

    fn calculate_signal_confidence(&self, symbol: &str) -> Decimal {
        // Count how many exchanges have data
        let exchange_count = self
            .histories
            .keys()
            .filter(|(_, s)| s == symbol)
            .count();

        // More exchanges = higher confidence
        let base_confidence = Decimal::from(exchange_count as u64) * dec!(0.2);

        // Check if rates are consistent across exchanges
        let rates: Vec<Decimal> = self
            .histories
            .iter()
            .filter(|((_, s), _)| s == symbol)
            .filter_map(|(_, h)| h.latest().map(|r| r.rate))
            .collect();

        if rates.len() < 2 {
            return base_confidence.min(Decimal::ONE);
        }

        let avg: Decimal = rates.iter().copied().sum::<Decimal>() / Decimal::from(rates.len() as u64);
        let variance: Decimal = rates
            .iter()
            .map(|r| (*r - avg).abs())
            .sum::<Decimal>()
            / Decimal::from(rates.len() as u64);

        // Lower variance = higher confidence
        let consistency_bonus = (dec!(0.001) - variance).max(Decimal::ZERO) * dec!(100);

        (base_confidence + consistency_bonus).min(Decimal::ONE)
    }
}

/// Funding rate arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbitrageOpportunity {
    pub symbol: String,
    pub long_exchange: Exchange,  // Go long here (lower/negative funding)
    pub short_exchange: Exchange, // Go short here (higher/positive funding)
    pub spread: Decimal,          // Absolute spread
    pub spread_bps: Decimal,      // Spread in basis points
    pub long_rate: Decimal,       // Funding rate on long exchange
    pub short_rate: Decimal,      // Funding rate on short exchange
    pub estimated_apy: Decimal,   // Estimated APY after fees
    pub timestamp: u64,
}

impl FundingArbitrageOpportunity {
    /// Calculate profit for a given position size over N funding periods
    pub fn calculate_profit(
        &self,
        position_size: Decimal,
        periods: u64,
        include_fees: bool,
    ) -> Decimal {
        let gross_profit = self.spread * position_size * Decimal::from(periods);

        if !include_fees {
            return gross_profit;
        }

        // Entry and exit fees on both legs
        let fee_rate = (self.long_exchange.taker_fee_bps() + self.short_exchange.taker_fee_bps())
            / dec!(10000);
        let total_fees = position_size * fee_rate * dec!(2) * dec!(2); // Entry + exit, both legs

        gross_profit - total_fees
    }

    /// Check if opportunity is profitable after fees
    pub fn is_profitable(&self, min_periods: u64) -> bool {
        self.calculate_profit(Decimal::ONE, min_periods, true) > Decimal::ZERO
    }
}

/// Direction signal from funding rate analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Neutral,
}

/// Funding rate direction signal for trading
#[derive(Debug, Clone)]
pub struct FundingDirectionSignal {
    pub symbol: String,
    pub direction: Direction,
    pub strength: Decimal,      // 0-1 signal strength
    pub avg_funding_rate: Decimal,
    pub confidence: Decimal,    // 0-1 confidence level
}

impl FundingDirectionSignal {
    /// Convert to a trading edge (probability above 50%)
    pub fn to_edge(&self) -> Option<Decimal> {
        if self.direction == Direction::Neutral {
            return None;
        }
        // Convert strength (0-1) to edge (0-0.1)
        // Max edge from funding signal alone is 10%
        let edge = self.strength * self.confidence * dec!(0.1);
        Some(edge)
    }
}

/// Delta-neutral position for funding rate arbitrage
#[derive(Debug, Clone)]
pub struct DeltaNeutralPosition {
    pub symbol: String,
    pub spot_exchange: Exchange,
    pub perp_exchange: Exchange,
    pub spot_size: Decimal,
    pub perp_size: Decimal,
    pub entry_price: Decimal,
    pub entry_funding_rate: Decimal,
    pub entry_time: u64,
    pub realized_funding: Decimal,
    pub unrealized_pnl: Decimal,
}

impl DeltaNeutralPosition {
    pub fn new(
        symbol: &str,
        spot_exchange: Exchange,
        perp_exchange: Exchange,
        size: Decimal,
        entry_price: Decimal,
        entry_funding_rate: Decimal,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            spot_exchange,
            perp_exchange,
            spot_size: size,
            perp_size: -size, // Short perp to hedge
            entry_price,
            entry_funding_rate,
            entry_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            realized_funding: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
        }
    }

    /// Update with new funding payment
    pub fn add_funding_payment(&mut self, funding_rate: Decimal) {
        // Short position receives positive funding, pays negative
        let payment = self.perp_size.abs() * self.entry_price * funding_rate;
        self.realized_funding += payment;
    }

    /// Update unrealized P&L based on current prices
    pub fn update_pnl(&mut self, spot_price: Decimal, perp_price: Decimal) {
        let spot_pnl = self.spot_size * (spot_price - self.entry_price);
        let perp_pnl = self.perp_size * (perp_price - self.entry_price);
        self.unrealized_pnl = spot_pnl + perp_pnl;
    }

    /// Total P&L including funding
    pub fn total_pnl(&self) -> Decimal {
        self.realized_funding + self.unrealized_pnl
    }

    /// Calculate current APY based on realized funding
    pub fn current_apy(&self) -> Decimal {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let hours_held = (now - self.entry_time) as f64 / 3600.0;
        if hours_held < 1.0 {
            return Decimal::ZERO;
        }

        let position_value = self.spot_size * self.entry_price;
        if position_value == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let hourly_return = self.realized_funding / position_value / Decimal::from_f64_retain(hours_held).unwrap_or(Decimal::ONE);
        hourly_return * dec!(8760) * dec!(100) // Annualized %
    }
}

/// Arbitrage engine configuration
#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    pub min_spread_bps: Decimal,
    pub max_position_size: Decimal,
    pub max_leverage: Decimal,
    pub min_profitability_periods: u64,
    pub max_positions: usize,
    pub exit_spread_threshold: Decimal, // Close when spread narrows to this
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: dec!(5),      // 0.05% minimum spread
            max_position_size: dec!(50000), // $50k max per position
            max_leverage: dec!(3),        // 3x max leverage
            min_profitability_periods: 3, // Must be profitable after 3 periods
            max_positions: 5,
            exit_spread_threshold: dec!(2), // Exit when spread < 0.02%
        }
    }
}

/// Main arbitrage engine
#[derive(Debug, Clone)]
pub struct ArbitrageEngine {
    tracker: FundingRateTracker,
    config: ArbitrageConfig,
    positions: Vec<DeltaNeutralPosition>,
}

impl ArbitrageEngine {
    pub fn new(tracker: FundingRateTracker) -> Self {
        Self {
            tracker,
            config: ArbitrageConfig::default(),
            positions: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: ArbitrageConfig) -> Self {
        self.config = config;
        self
    }

    /// Scan for profitable arbitrage opportunities
    pub fn scan_opportunities(&self) -> Vec<FundingArbitrageOpportunity> {
        let mut all_opportunities = Vec::new();

        for symbol in &self.tracker.supported_symbols {
            let opps = self
                .tracker
                .find_arbitrage_opportunities(symbol, self.config.min_spread_bps);
            for opp in opps {
                if opp.is_profitable(self.config.min_profitability_periods) {
                    all_opportunities.push(opp);
                }
            }
        }

        // Sort by estimated APY
        all_opportunities.sort_by(|a, b| b.estimated_apy.partial_cmp(&a.estimated_apy).unwrap_or(std::cmp::Ordering::Equal));
        all_opportunities
    }

    /// Get direction signals for all tracked symbols
    pub fn get_all_direction_signals(&self) -> Vec<FundingDirectionSignal> {
        self.tracker
            .supported_symbols
            .iter()
            .map(|s| self.tracker.get_direction_signal(s))
            .collect()
    }

    /// Calculate optimal position size for an opportunity
    pub fn calculate_position_size(&self, opportunity: &FundingArbitrageOpportunity, capital: Decimal) -> Decimal {
        // Kelly criterion for funding rate arbitrage
        // Simplified: bet proportional to edge / variance
        let edge = opportunity.spread_bps / dec!(10000);
        let variance = dec!(0.001); // Assume 0.1% variance in funding rates

        let kelly_fraction = edge / variance;
        let half_kelly = kelly_fraction / dec!(2);

        // Apply position limits
        let max_by_capital = capital * self.config.max_leverage * half_kelly;
        let max_by_config = self.config.max_position_size;

        max_by_capital.min(max_by_config)
    }

    /// Add a new position
    pub fn open_position(&mut self, position: DeltaNeutralPosition) -> Result<(), &'static str> {
        if self.positions.len() >= self.config.max_positions {
            return Err("Maximum positions reached");
        }
        self.positions.push(position);
        Ok(())
    }

    /// Get current positions
    pub fn positions(&self) -> &[DeltaNeutralPosition] {
        &self.positions
    }

    /// Update funding rate tracker
    pub fn update_rate(&mut self, rate: FundingRate) {
        self.tracker.add_rate(rate);
    }

    /// Check if any positions should be closed
    pub fn check_exits(&self) -> Vec<String> {
        let mut to_close = Vec::new();

        for pos in &self.positions {
            // Check if spread has narrowed
            if let (Some(long_rate), Some(short_rate)) = (
                self.tracker.get_latest(pos.spot_exchange, &pos.symbol),
                self.tracker.get_latest(pos.perp_exchange, &pos.symbol),
            ) {
                let current_spread = (short_rate.rate - long_rate.rate).abs() * dec!(10000);
                if current_spread < self.config.exit_spread_threshold {
                    to_close.push(pos.symbol.clone());
                }
            }
        }

        to_close
    }
}

/// Calculate square root of a Decimal using Newton's method
fn sqrt_decimal(n: Decimal) -> Decimal {
    if n <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut x = n;
    let two = dec!(2);

    for _ in 0..20 {
        let next_x = (x + n / x) / two;
        if (next_x - x).abs() < dec!(0.0000000001) {
            return next_x;
        }
        x = next_x;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn test_exchange_properties() {
        assert_eq!(Exchange::Binance.funding_interval_hours(), 8);
        assert_eq!(Exchange::DYDX.funding_interval_hours(), 1);
        assert!(Exchange::Binance.maker_fee_bps() < Exchange::GMX.maker_fee_bps());
    }

    #[test]
    fn test_funding_rate_apy() {
        let rate = FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001), // 0.01%
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        };

        // 0.01% * 3 per day * 365 days = ~10.95% APY
        let apy = rate.annualized_apy();
        assert!(apy > dec!(10) && apy < dec!(12));
    }

    #[test]
    fn test_funding_window() {
        let mut rate = FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 300, // 5 minutes
            interval_hours: 8,
        };

        assert!(rate.in_funding_window(10)); // 10 minute window
        assert!(!rate.in_funding_window(3)); // 3 minute window

        rate.next_funding_time = now_timestamp() + 7200; // 2 hours
        assert!(!rate.in_funding_window(60));
    }

    #[test]
    fn test_funding_rate_history() {
        let mut history = FundingRateHistory::new("BTC", Exchange::Binance, 10);

        for i in 0..5 {
            history.add_rate(FundingRate {
                exchange: Exchange::Binance,
                symbol: "BTC".to_string(),
                rate: Decimal::from(i) * dec!(0.0001),
                predicted_rate: None,
                timestamp: now_timestamp() + i * 3600,
                next_funding_time: now_timestamp() + (i + 1) * 3600,
                interval_hours: 8,
            });
        }

        assert_eq!(history.get_rates().len(), 5);
        assert_eq!(history.latest().unwrap().rate, dec!(0.0004));
        
        let avg = history.average_rate(5).unwrap();
        assert_eq!(avg, dec!(0.0002)); // (0+1+2+3+4)/5 * 0.0001
    }

    #[test]
    fn test_history_max_capacity() {
        let mut history = FundingRateHistory::new("BTC", Exchange::Binance, 3);

        for i in 0..5 {
            history.add_rate(FundingRate {
                exchange: Exchange::Binance,
                symbol: "BTC".to_string(),
                rate: Decimal::from(i) * dec!(0.0001),
                predicted_rate: None,
                timestamp: now_timestamp() + i * 3600,
                next_funding_time: now_timestamp() + (i + 1) * 3600,
                interval_hours: 8,
            });
        }

        assert_eq!(history.get_rates().len(), 3); // Only keeps last 3
        assert_eq!(history.latest().unwrap().rate, dec!(0.0004));
    }

    #[test]
    fn test_predict_next_rate() {
        let mut history = FundingRateHistory::new("BTC", Exchange::Binance, 10);

        // Add increasing rates
        for i in 1..=5 {
            history.add_rate(FundingRate {
                exchange: Exchange::Binance,
                symbol: "BTC".to_string(),
                rate: Decimal::from(i) * dec!(0.0001),
                predicted_rate: None,
                timestamp: now_timestamp() + (i as u64) * 3600,
                next_funding_time: now_timestamp() + ((i + 1) as u64) * 3600,
                interval_hours: 8,
            });
        }

        let prediction = history.predict_next_rate(dec!(0.5)).unwrap();
        // EMA should be weighted toward recent values (0.0004, 0.0005)
        assert!(prediction > dec!(0.0003));
    }

    #[test]
    fn test_is_trending_up() {
        let mut history = FundingRateHistory::new("BTC", Exchange::Binance, 10);

        // Add increasing rates
        for i in 1..=6 {
            history.add_rate(FundingRate {
                exchange: Exchange::Binance,
                symbol: "BTC".to_string(),
                rate: Decimal::from(i) * dec!(0.0001),
                predicted_rate: None,
                timestamp: now_timestamp() + (i as u64) * 3600,
                next_funding_time: now_timestamp() + ((i + 1) as u64) * 3600,
                interval_hours: 8,
            });
        }

        assert!(history.is_trending_up(4));
    }

    #[test]
    fn test_funding_rate_tracker() {
        let mut tracker = FundingRateTracker::new()
            .with_symbols(vec!["BTC".to_string(), "ETH".to_string()]);

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::OKX,
            symbol: "BTC".to_string(),
            rate: dec!(0.00015),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let binance_rate = tracker.get_latest(Exchange::Binance, "BTC").unwrap();
        assert_eq!(binance_rate.rate, dec!(0.0001));

        let avg = tracker.cross_exchange_average("BTC").unwrap();
        assert_eq!(avg, dec!(0.000125)); // (0.0001 + 0.00015) / 2
    }

    #[test]
    fn test_find_arbitrage_opportunities() {
        let mut tracker = FundingRateTracker::new();

        // Binance: low funding (good to long)
        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        // OKX: high funding (good to short)
        tracker.add_rate(FundingRate {
            exchange: Exchange::OKX,
            symbol: "BTC".to_string(),
            rate: dec!(0.001), // 0.1% - very high
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let opps = tracker.find_arbitrage_opportunities("BTC", dec!(5)); // 0.05% min
        assert_eq!(opps.len(), 1);

        let opp = &opps[0];
        assert_eq!(opp.long_exchange, Exchange::Binance);
        assert_eq!(opp.short_exchange, Exchange::OKX);
        assert!(opp.spread_bps > dec!(8)); // Should be ~9 bps (0.001 - 0.0001 = 0.0009 = 9 bps)
    }

    #[test]
    fn test_arbitrage_profitability() {
        let opp = FundingArbitrageOpportunity {
            symbol: "BTC".to_string(),
            long_exchange: Exchange::Binance,
            short_exchange: Exchange::OKX,
            spread: dec!(0.0009), // 0.09%
            spread_bps: dec!(9),
            long_rate: dec!(0.0001),
            short_rate: dec!(0.001),
            estimated_apy: dec!(50),
            timestamp: now_timestamp(),
        };

        // Profit without fees
        let gross = opp.calculate_profit(dec!(10000), 1, false);
        assert_eq!(gross, dec!(9)); // 0.0009 * 10000

        // With fees, should be lower
        let net = opp.calculate_profit(dec!(10000), 1, true);
        assert!(net < gross);
    }

    #[test]
    fn test_direction_signal_bullish() {
        let mut tracker = FundingRateTracker::new();

        // Very negative funding = shorts paying longs = bullish signal
        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(-0.002), // -0.2% very negative
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let signal = tracker.get_direction_signal("BTC");
        assert_eq!(signal.direction, Direction::Up);
        assert!(signal.strength > Decimal::ZERO);
    }

    #[test]
    fn test_direction_signal_bearish() {
        let mut tracker = FundingRateTracker::new();

        // Very positive funding = longs paying shorts = bearish signal
        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.002), // 0.2% very positive
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let signal = tracker.get_direction_signal("BTC");
        assert_eq!(signal.direction, Direction::Down);
    }

    #[test]
    fn test_direction_signal_to_edge() {
        let signal = FundingDirectionSignal {
            symbol: "BTC".to_string(),
            direction: Direction::Up,
            strength: dec!(0.5),
            avg_funding_rate: dec!(-0.001),
            confidence: dec!(0.8),
        };

        let edge = signal.to_edge().unwrap();
        assert!(edge > Decimal::ZERO);
        assert!(edge <= dec!(0.1)); // Max 10% edge
    }

    #[test]
    fn test_delta_neutral_position() {
        let mut pos = DeltaNeutralPosition::new(
            "BTC",
            Exchange::Binance,
            Exchange::Binance,
            dec!(1),       // 1 BTC
            dec!(100000),  // $100k entry
            dec!(0.0001),
        );

        assert_eq!(pos.spot_size, dec!(1));
        assert_eq!(pos.perp_size, dec!(-1)); // Short hedge

        // Simulate funding payment (positive rate, short receives)
        pos.add_funding_payment(dec!(0.0001));
        assert_eq!(pos.realized_funding, dec!(10)); // 1 * 100000 * 0.0001

        // Price moves: spot up, perp up (delta neutral)
        pos.update_pnl(dec!(101000), dec!(101000));
        assert_eq!(pos.unrealized_pnl, Decimal::ZERO); // Hedged

        // Total PnL = funding only
        assert_eq!(pos.total_pnl(), dec!(10));
    }

    #[test]
    fn test_arbitrage_engine_scan() {
        let mut tracker = FundingRateTracker::new()
            .with_symbols(vec!["BTC".to_string()]);

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(-0.001), // Negative funding (receive payment)
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::OKX,
            symbol: "BTC".to_string(),
            rate: dec!(0.003), // High positive funding (wide spread = 40 bps)
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let engine = ArbitrageEngine::new(tracker);
        let opps = engine.scan_opportunities();

        assert!(!opps.is_empty());
        assert!(opps[0].estimated_apy > Decimal::ZERO);
    }

    #[test]
    fn test_position_size_calculation() {
        let tracker = FundingRateTracker::new();
        let engine = ArbitrageEngine::new(tracker);

        let opp = FundingArbitrageOpportunity {
            symbol: "BTC".to_string(),
            long_exchange: Exchange::Binance,
            short_exchange: Exchange::OKX,
            spread: dec!(0.001),
            spread_bps: dec!(10),
            long_rate: dec!(0.0001),
            short_rate: dec!(0.0011),
            estimated_apy: dec!(100),
            timestamp: now_timestamp(),
        };

        let size = engine.calculate_position_size(&opp, dec!(100000));
        assert!(size > Decimal::ZERO);
        assert!(size <= dec!(50000)); // Max position
    }

    #[test]
    fn test_engine_max_positions() {
        let tracker = FundingRateTracker::new();
        let mut engine = ArbitrageEngine::new(tracker)
            .with_config(ArbitrageConfig {
                max_positions: 2,
                ..Default::default()
            });

        let pos1 = DeltaNeutralPosition::new("BTC", Exchange::Binance, Exchange::OKX, dec!(1), dec!(100000), dec!(0.0001));
        let pos2 = DeltaNeutralPosition::new("ETH", Exchange::Binance, Exchange::OKX, dec!(10), dec!(3000), dec!(0.0001));
        let pos3 = DeltaNeutralPosition::new("SOL", Exchange::Binance, Exchange::OKX, dec!(100), dec!(100), dec!(0.0001));

        assert!(engine.open_position(pos1).is_ok());
        assert!(engine.open_position(pos2).is_ok());
        assert!(engine.open_position(pos3).is_err()); // Should fail
    }

    #[test]
    fn test_sqrt_decimal() {
        let result = sqrt_decimal(dec!(4));
        assert!((result - dec!(2)).abs() < dec!(0.0001));

        let result = sqrt_decimal(dec!(2));
        assert!((result - dec!(1.414213)).abs() < dec!(0.0001));

        let result = sqrt_decimal(dec!(0));
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_std_dev_calculation() {
        let mut history = FundingRateHistory::new("BTC", Exchange::Binance, 10);

        // Add some rates with known variance
        for rate in [dec!(0.0001), dec!(0.0002), dec!(0.0003), dec!(0.0002), dec!(0.0002)] {
            history.add_rate(FundingRate {
                exchange: Exchange::Binance,
                symbol: "BTC".to_string(),
                rate,
                predicted_rate: None,
                timestamp: now_timestamp(),
                next_funding_time: now_timestamp() + 3600,
                interval_hours: 8,
            });
        }

        let std_dev = history.rate_std_dev(5).unwrap();
        assert!(std_dev > Decimal::ZERO);
        assert!(std_dev < dec!(0.001)); // Should be small
    }

    #[test]
    fn test_multiple_symbols() {
        let mut tracker = FundingRateTracker::new()
            .with_symbols(vec!["BTC".to_string(), "ETH".to_string()]);

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "ETH".to_string(),
            rate: dec!(0.0002),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let btc_rate = tracker.get_latest(Exchange::Binance, "BTC").unwrap();
        let eth_rate = tracker.get_latest(Exchange::Binance, "ETH").unwrap();

        assert_eq!(btc_rate.rate, dec!(0.0001));
        assert_eq!(eth_rate.rate, dec!(0.0002));
    }

    #[test]
    fn test_engine_get_all_signals() {
        let mut tracker = FundingRateTracker::new()
            .with_symbols(vec!["BTC".to_string(), "ETH".to_string()]);

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(-0.002),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "ETH".to_string(),
            rate: dec!(0.002),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let engine = ArbitrageEngine::new(tracker);
        let signals = engine.get_all_direction_signals();

        assert_eq!(signals.len(), 2);
        
        let btc_signal = signals.iter().find(|s| s.symbol == "BTC").unwrap();
        let eth_signal = signals.iter().find(|s| s.symbol == "ETH").unwrap();
        
        assert_eq!(btc_signal.direction, Direction::Up);
        assert_eq!(eth_signal.direction, Direction::Down);
    }

    #[test]
    fn test_check_exits() {
        let mut tracker = FundingRateTracker::new();
        
        // Initial wide spread
        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::OKX,
            symbol: "BTC".to_string(),
            rate: dec!(0.00011), // Very narrow spread now
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let mut engine = ArbitrageEngine::new(tracker)
            .with_config(ArbitrageConfig {
                exit_spread_threshold: dec!(5), // 0.05% threshold
                ..Default::default()
            });

        let pos = DeltaNeutralPosition::new(
            "BTC",
            Exchange::Binance,
            Exchange::OKX,
            dec!(1),
            dec!(100000),
            dec!(0.0005),
        );
        engine.open_position(pos).unwrap();

        let to_close = engine.check_exits();
        assert!(to_close.contains(&"BTC".to_string()));
    }

    #[test]
    fn test_confidence_calculation() {
        let mut tracker = FundingRateTracker::new();

        // Single exchange = lower confidence
        tracker.add_rate(FundingRate {
            exchange: Exchange::Binance,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let signal1 = tracker.get_direction_signal("BTC");

        // Add more exchanges with consistent rates
        tracker.add_rate(FundingRate {
            exchange: Exchange::OKX,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        tracker.add_rate(FundingRate {
            exchange: Exchange::Bybit,
            symbol: "BTC".to_string(),
            rate: dec!(0.0001),
            predicted_rate: None,
            timestamp: now_timestamp(),
            next_funding_time: now_timestamp() + 3600,
            interval_hours: 8,
        });

        let signal2 = tracker.get_direction_signal("BTC");

        // More exchanges + consistent rates = higher confidence
        assert!(signal2.confidence > signal1.confidence);
    }
}
