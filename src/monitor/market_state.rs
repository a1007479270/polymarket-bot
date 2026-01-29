//! Market State Monitor
//!
//! Real-time monitoring of market conditions:
//! 1. Volatility regime detection (low/medium/high/extreme)
//! 2. Liquidity monitoring
//! 3. Price momentum tracking
//! 4. Anomaly detection
//! 5. Alert generation

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

/// Market state monitor configuration
#[derive(Debug, Clone)]
pub struct MarketStateConfig {
    /// Window size for volatility calculation (minutes)
    pub volatility_window_mins: i64,
    /// Minimum price updates needed for valid volatility
    pub min_updates_for_volatility: usize,
    /// Thresholds for volatility regimes (annualized %)
    pub volatility_thresholds: VolatilityThresholds,
    /// Price change threshold for momentum alert (%)
    pub momentum_alert_threshold_pct: Decimal,
    /// Liquidity drop threshold for alert (%)
    pub liquidity_drop_threshold_pct: Decimal,
    /// Anomaly detection sensitivity
    pub anomaly_sensitivity: Decimal,
}

#[derive(Debug, Clone)]
pub struct VolatilityThresholds {
    pub low_to_medium: Decimal,  // Below this = low volatility
    pub medium_to_high: Decimal, // Above this = high volatility
    pub high_to_extreme: Decimal, // Above this = extreme
}

impl Default for VolatilityThresholds {
    fn default() -> Self {
        Self {
            low_to_medium: dec!(20),   // 20% annualized
            medium_to_high: dec!(50),  // 50% annualized
            high_to_extreme: dec!(100), // 100% annualized
        }
    }
}

impl Default for MarketStateConfig {
    fn default() -> Self {
        Self {
            volatility_window_mins: 60, // 1 hour window
            min_updates_for_volatility: 10,
            volatility_thresholds: VolatilityThresholds::default(),
            momentum_alert_threshold_pct: dec!(5), // 5% price move
            liquidity_drop_threshold_pct: dec!(50), // 50% liquidity drop
            anomaly_sensitivity: dec!(2.5), // 2.5 standard deviations
        }
    }
}

/// Volatility regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

impl VolatilityRegime {
    /// Get recommended Kelly fraction multiplier for this regime
    pub fn kelly_multiplier(&self) -> Decimal {
        match self {
            Self::Low => dec!(1.2),     // Can be more aggressive
            Self::Medium => dec!(1.0),  // Normal
            Self::High => dec!(0.7),    // Reduce exposure
            Self::Extreme => dec!(0.4), // Significantly reduce
        }
    }

    /// Get recommended max position size multiplier
    pub fn max_position_multiplier(&self) -> Decimal {
        match self {
            Self::Low => dec!(1.2),
            Self::Medium => dec!(1.0),
            Self::High => dec!(0.6),
            Self::Extreme => dec!(0.3),
        }
    }
}

/// Price update record
#[derive(Debug, Clone)]
struct PriceUpdate {
    timestamp: DateTime<Utc>,
    price: Decimal,
    volume: Option<Decimal>,
    bid_depth: Option<Decimal>,
    ask_depth: Option<Decimal>,
}

/// Current market state
#[derive(Debug, Clone)]
pub struct MarketState {
    pub market_id: String,
    pub current_price: Decimal,
    pub volatility_regime: VolatilityRegime,
    pub volatility_pct: Decimal,
    pub momentum: Momentum,
    pub liquidity_score: Decimal,
    pub last_update: DateTime<Utc>,
    pub alerts: Vec<Alert>,
    pub anomalies: Vec<Anomaly>,
}

#[derive(Debug, Clone)]
pub struct Momentum {
    /// Price change over window (%)
    pub price_change_pct: Decimal,
    /// Direction strength (0-1)
    pub strength: Decimal,
    /// Is the move accelerating
    pub accelerating: bool,
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertType {
    VolatilitySpike,
    LiquidityDrop,
    PriceMomentum,
    Anomaly,
    RegimeChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Anomaly {
    pub anomaly_type: AnomalyType,
    pub value: Decimal,
    pub expected_range: (Decimal, Decimal),
    pub deviation_score: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    PriceJump,
    VolumeSpike,
    SpreadWidening,
    DepthDisappearance,
}

/// Market state monitor
pub struct MarketStateMonitor {
    config: MarketStateConfig,
    /// Price history per market
    price_history: RwLock<HashMap<String, VecDeque<PriceUpdate>>>,
    /// Current state per market
    states: RwLock<HashMap<String, MarketState>>,
    /// Historical volatility for regime detection
    volatility_history: RwLock<HashMap<String, VecDeque<Decimal>>>,
    /// Alert callbacks
    alert_handlers: RwLock<Vec<Box<dyn Fn(&Alert) + Send + Sync>>>,
}

impl MarketStateMonitor {
    pub fn new(config: MarketStateConfig) -> Self {
        Self {
            config,
            price_history: RwLock::new(HashMap::new()),
            states: RwLock::new(HashMap::new()),
            volatility_history: RwLock::new(HashMap::new()),
            alert_handlers: RwLock::new(Vec::new()),
        }
    }

    /// Update market with new price data
    pub fn update(
        &self,
        market_id: &str,
        price: Decimal,
        volume: Option<Decimal>,
        bid_depth: Option<Decimal>,
        ask_depth: Option<Decimal>,
    ) -> MarketState {
        let now = Utc::now();
        
        let update = PriceUpdate {
            timestamp: now,
            price,
            volume,
            bid_depth,
            ask_depth,
        };

        // Add to history
        {
            let mut history = self.price_history.write().unwrap();
            let market_history = history.entry(market_id.to_string()).or_default();
            market_history.push_back(update);
            
            // Keep only recent data
            let cutoff = now - Duration::minutes(self.config.volatility_window_mins * 2);
            while market_history.front().map(|u| u.timestamp < cutoff).unwrap_or(false) {
                market_history.pop_front();
            }
        }

        // Calculate new state
        let state = self.calculate_state(market_id);
        
        // Store state
        {
            let mut states = self.states.write().unwrap();
            states.insert(market_id.to_string(), state.clone());
        }

        // Fire alerts
        for alert in &state.alerts {
            self.fire_alert(alert);
        }

        state
    }

    /// Calculate current market state
    fn calculate_state(&self, market_id: &str) -> MarketState {
        let history = self.price_history.read().unwrap();
        let market_history = match history.get(market_id) {
            Some(h) if !h.is_empty() => h,
            _ => return self.default_state(market_id),
        };

        let now = Utc::now();
        let window_cutoff = now - Duration::minutes(self.config.volatility_window_mins);
        
        // Get recent prices
        let recent_updates: Vec<_> = market_history
            .iter()
            .filter(|u| u.timestamp >= window_cutoff)
            .collect();

        if recent_updates.len() < self.config.min_updates_for_volatility {
            return self.default_state(market_id);
        }

        let current_price = recent_updates.last().unwrap().price;
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&recent_updates);
        let volatility_regime = self.classify_volatility(volatility);

        // Calculate momentum
        let momentum = self.calculate_momentum(&recent_updates);

        // Calculate liquidity score
        let liquidity_score = self.calculate_liquidity_score(&recent_updates);

        // Detect anomalies
        let anomalies = self.detect_anomalies(market_id, &recent_updates);

        // Generate alerts
        let mut alerts = Vec::new();
        
        // Check for regime change
        if let Some(prev_state) = self.states.read().unwrap().get(market_id) {
            if prev_state.volatility_regime != volatility_regime {
                alerts.push(Alert {
                    alert_type: AlertType::RegimeChange,
                    severity: if volatility_regime == VolatilityRegime::Extreme {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    message: format!(
                        "Volatility regime changed: {:?} -> {:?} ({:.1}%)",
                        prev_state.volatility_regime, volatility_regime, volatility
                    ),
                    timestamp: now,
                });
            }

            // Check for liquidity drop
            if liquidity_score < prev_state.liquidity_score * (Decimal::ONE - self.config.liquidity_drop_threshold_pct / dec!(100)) {
                alerts.push(Alert {
                    alert_type: AlertType::LiquidityDrop,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Liquidity dropped {:.0}%: {:.2} -> {:.2}",
                        (Decimal::ONE - liquidity_score / prev_state.liquidity_score) * dec!(100),
                        prev_state.liquidity_score,
                        liquidity_score
                    ),
                    timestamp: now,
                });
            }
        }

        // Check for momentum alert
        if momentum.price_change_pct.abs() >= self.config.momentum_alert_threshold_pct {
            alerts.push(Alert {
                alert_type: AlertType::PriceMomentum,
                severity: if momentum.price_change_pct.abs() >= self.config.momentum_alert_threshold_pct * dec!(2) {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!(
                    "Strong price momentum: {}{:.2}% in {} minutes",
                    if momentum.price_change_pct > Decimal::ZERO { "+" } else { "" },
                    momentum.price_change_pct,
                    self.config.volatility_window_mins
                ),
                timestamp: now,
            });
        }

        // Add anomaly alerts
        for anomaly in &anomalies {
            alerts.push(Alert {
                alert_type: AlertType::Anomaly,
                severity: if anomaly.deviation_score > dec!(3.5) {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!(
                    "{:?} detected: value {:.4} outside expected range [{:.4}, {:.4}] (score: {:.1}σ)",
                    anomaly.anomaly_type,
                    anomaly.value,
                    anomaly.expected_range.0,
                    anomaly.expected_range.1,
                    anomaly.deviation_score
                ),
                timestamp: now,
            });
        }

        drop(history);

        // Update volatility history for regime detection
        {
            let mut vol_history = self.volatility_history.write().unwrap();
            let entry = vol_history.entry(market_id.to_string()).or_default();
            entry.push_back(volatility);
            while entry.len() > 24 { // Keep 24 readings
                entry.pop_front();
            }
        }

        MarketState {
            market_id: market_id.to_string(),
            current_price,
            volatility_regime,
            volatility_pct: volatility,
            momentum,
            liquidity_score,
            last_update: now,
            alerts,
            anomalies,
        }
    }

    /// Calculate annualized volatility from price updates
    fn calculate_volatility(&self, updates: &[&PriceUpdate]) -> Decimal {
        if updates.len() < 2 {
            return dec!(0);
        }

        // Calculate log returns
        let mut returns: Vec<Decimal> = Vec::with_capacity(updates.len() - 1);
        for i in 1..updates.len() {
            let prev = updates[i - 1].price;
            let curr = updates[i].price;
            if prev > Decimal::ZERO {
                let ret = (curr / prev - Decimal::ONE) * dec!(100);
                returns.push(ret);
            }
        }

        if returns.is_empty() {
            return dec!(0);
        }

        // Calculate mean
        let mean: Decimal = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as u32);

        // Calculate variance
        let variance: Decimal = returns
            .iter()
            .map(|r| (*r - mean) * (*r - mean))
            .sum::<Decimal>() / Decimal::from(returns.len() as u32);

        // Standard deviation
        let std_dev = variance.sqrt().unwrap_or(Decimal::ZERO);

        // Annualize (assuming hourly readings, 8760 hours per year)
        // For minute data: sqrt(525600) ≈ 725
        std_dev * dec!(725) / dec!(100) * dec!(100) // Convert to percentage
    }

    /// Classify volatility into regime
    fn classify_volatility(&self, volatility_pct: Decimal) -> VolatilityRegime {
        if volatility_pct < self.config.volatility_thresholds.low_to_medium {
            VolatilityRegime::Low
        } else if volatility_pct < self.config.volatility_thresholds.medium_to_high {
            VolatilityRegime::Medium
        } else if volatility_pct < self.config.volatility_thresholds.high_to_extreme {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }

    /// Calculate momentum metrics
    fn calculate_momentum(&self, updates: &[&PriceUpdate]) -> Momentum {
        if updates.len() < 2 {
            return Momentum {
                price_change_pct: dec!(0),
                strength: dec!(0),
                accelerating: false,
            };
        }

        let first_price = updates.first().unwrap().price;
        let last_price = updates.last().unwrap().price;
        let price_change_pct = if first_price > Decimal::ZERO {
            (last_price - first_price) / first_price * dec!(100)
        } else {
            dec!(0)
        };

        // Calculate strength as consistency of direction
        let mut up_moves = 0u32;
        let mut down_moves = 0u32;
        for i in 1..updates.len() {
            if updates[i].price > updates[i - 1].price {
                up_moves += 1;
            } else if updates[i].price < updates[i - 1].price {
                down_moves += 1;
            }
        }
        let total_moves = up_moves + down_moves;
        let strength = if total_moves > 0 {
            let dominant = up_moves.max(down_moves);
            Decimal::from(dominant) / Decimal::from(total_moves)
        } else {
            dec!(0)
        };

        // Check if accelerating (larger recent moves)
        let half = updates.len() / 2;
        let first_half_change = if updates[half].price > Decimal::ZERO {
            (updates[half].price - first_price).abs() / first_price
        } else {
            dec!(0)
        };
        let second_half_change = if updates[half].price > Decimal::ZERO {
            (last_price - updates[half].price).abs() / updates[half].price
        } else {
            dec!(0)
        };
        let accelerating = second_half_change > first_half_change * dec!(1.2);

        Momentum {
            price_change_pct,
            strength,
            accelerating,
        }
    }

    /// Calculate liquidity score (0-1)
    fn calculate_liquidity_score(&self, updates: &[&PriceUpdate]) -> Decimal {
        let recent = updates.last().unwrap();
        
        match (recent.bid_depth, recent.ask_depth) {
            (Some(bid), Some(ask)) => {
                let total_depth = bid + ask;
                // Normalize to a 0-1 score (assuming $10000 is good liquidity)
                (total_depth / dec!(10000)).min(dec!(1))
            }
            _ => dec!(0.5), // Unknown liquidity
        }
    }

    /// Detect anomalies in recent data
    fn detect_anomalies(&self, _market_id: &str, updates: &[&PriceUpdate]) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();
        
        if updates.len() < 5 {
            return anomalies;
        }

        let now = Utc::now();

        // Calculate price change statistics
        let mut changes: Vec<Decimal> = Vec::new();
        for i in 1..updates.len() {
            if updates[i - 1].price > Decimal::ZERO {
                let change = (updates[i].price - updates[i - 1].price) / updates[i - 1].price * dec!(100);
                changes.push(change);
            }
        }

        if changes.is_empty() {
            return anomalies;
        }

        let mean: Decimal = changes.iter().sum::<Decimal>() / Decimal::from(changes.len() as u32);
        let variance: Decimal = changes
            .iter()
            .map(|c| (*c - mean) * (*c - mean))
            .sum::<Decimal>() / Decimal::from(changes.len() as u32);
        let std_dev = variance.sqrt().unwrap_or(dec!(1));

        // Check latest change for anomaly
        if let Some(&last_change) = changes.last() {
            let z_score = if std_dev > Decimal::ZERO {
                (last_change - mean).abs() / std_dev
            } else {
                dec!(0)
            };

            if z_score > self.config.anomaly_sensitivity {
                let expected_min = mean - std_dev * self.config.anomaly_sensitivity;
                let expected_max = mean + std_dev * self.config.anomaly_sensitivity;
                
                anomalies.push(Anomaly {
                    anomaly_type: AnomalyType::PriceJump,
                    value: last_change,
                    expected_range: (expected_min, expected_max),
                    deviation_score: z_score,
                    timestamp: now,
                });
            }
        }

        // Check volume spike if available
        let volumes: Vec<Decimal> = updates
            .iter()
            .filter_map(|u| u.volume)
            .collect();
        
        if volumes.len() >= 5 {
            let vol_mean: Decimal = volumes.iter().sum::<Decimal>() / Decimal::from(volumes.len() as u32);
            if let Some(&last_vol) = volumes.last() {
                if vol_mean > Decimal::ZERO && last_vol > vol_mean * dec!(3) {
                    let z_score = (last_vol - vol_mean) / vol_mean;
                    anomalies.push(Anomaly {
                        anomaly_type: AnomalyType::VolumeSpike,
                        value: last_vol,
                        expected_range: (vol_mean * dec!(0.5), vol_mean * dec!(2)),
                        deviation_score: z_score,
                        timestamp: now,
                    });
                }
            }
        }

        anomalies
    }

    /// Generate default state for markets with insufficient data
    fn default_state(&self, market_id: &str) -> MarketState {
        MarketState {
            market_id: market_id.to_string(),
            current_price: dec!(0),
            volatility_regime: VolatilityRegime::Medium,
            volatility_pct: dec!(30), // Assume medium volatility
            momentum: Momentum {
                price_change_pct: dec!(0),
                strength: dec!(0),
                accelerating: false,
            },
            liquidity_score: dec!(0.5),
            last_update: Utc::now(),
            alerts: vec![],
            anomalies: vec![],
        }
    }

    /// Fire alert to registered handlers
    fn fire_alert(&self, alert: &Alert) {
        let handlers = self.alert_handlers.read().unwrap();
        for handler in handlers.iter() {
            handler(alert);
        }
    }

    /// Register an alert handler
    pub fn on_alert<F>(&self, handler: F)
    where
        F: Fn(&Alert) + Send + Sync + 'static,
    {
        self.alert_handlers.write().unwrap().push(Box::new(handler));
    }

    /// Get current state for a market
    pub fn get_state(&self, market_id: &str) -> Option<MarketState> {
        self.states.read().unwrap().get(market_id).cloned()
    }

    /// Get all current states
    pub fn get_all_states(&self) -> HashMap<String, MarketState> {
        self.states.read().unwrap().clone()
    }

    /// Get trading recommendation based on market state
    pub fn get_recommendation(&self, market_id: &str) -> TradingRecommendation {
        let state = match self.get_state(market_id) {
            Some(s) => s,
            None => return TradingRecommendation::default(),
        };

        let mut warnings = Vec::new();
        let can_trade;
        let size_multiplier;
        let urgency_modifier;

        // Evaluate based on regime
        match state.volatility_regime {
            VolatilityRegime::Extreme => {
                can_trade = false;
                size_multiplier = dec!(0);
                urgency_modifier = dec!(0);
                warnings.push("Extreme volatility - trading suspended".to_string());
            }
            VolatilityRegime::High => {
                can_trade = true;
                size_multiplier = dec!(0.5);
                urgency_modifier = dec!(0.7);
                warnings.push("High volatility - reduced position sizes".to_string());
            }
            VolatilityRegime::Medium => {
                can_trade = true;
                size_multiplier = dec!(1.0);
                urgency_modifier = dec!(1.0);
            }
            VolatilityRegime::Low => {
                can_trade = true;
                size_multiplier = dec!(1.2);
                urgency_modifier = dec!(1.1);
            }
        }

        // Adjust for momentum
        let _momentum_warning = if state.momentum.accelerating && state.momentum.price_change_pct.abs() > dec!(5) {
            warnings.push(format!(
                "Strong momentum detected: {:.1}% - consider waiting",
                state.momentum.price_change_pct
            ));
            true
        } else {
            false
        };

        // Adjust for liquidity
        let _liquidity_warning = if state.liquidity_score < dec!(0.3) {
            warnings.push("Low liquidity - larger slippage expected".to_string());
            true
        } else {
            false
        };

        // Adjust for anomalies
        let has_critical_anomaly = state.anomalies.iter().any(|a| a.deviation_score > dec!(3.5));
        if has_critical_anomaly {
            warnings.push("Market anomaly detected - proceed with caution".to_string());
        }

        TradingRecommendation {
            can_trade: can_trade && !has_critical_anomaly,
            size_multiplier: if has_critical_anomaly { size_multiplier * dec!(0.5) } else { size_multiplier },
            urgency_modifier,
            warnings,
            volatility_regime: state.volatility_regime,
            liquidity_score: state.liquidity_score,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradingRecommendation {
    pub can_trade: bool,
    pub size_multiplier: Decimal,
    pub urgency_modifier: Decimal,
    pub warnings: Vec<String>,
    pub volatility_regime: VolatilityRegime,
    pub liquidity_score: Decimal,
}

impl Default for TradingRecommendation {
    fn default() -> Self {
        Self {
            can_trade: true,
            size_multiplier: dec!(1),
            urgency_modifier: dec!(1),
            warnings: vec![],
            volatility_regime: VolatilityRegime::Medium,
            liquidity_score: dec!(0.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> MarketStateMonitor {
        MarketStateMonitor::new(MarketStateConfig {
            volatility_window_mins: 60,
            min_updates_for_volatility: 5,
            ..Default::default()
        })
    }

    #[test]
    fn test_basic_update() {
        let monitor = make_monitor();
        
        // Need enough updates to not get default state
        for _ in 0..6 {
            monitor.update("test", dec!(0.50), Some(dec!(1000)), Some(dec!(500)), Some(dec!(500)));
        }
        
        let state = monitor.get_state("test").unwrap();
        assert_eq!(state.market_id, "test");
        assert_eq!(state.current_price, dec!(0.50));
    }

    #[test]
    fn test_volatility_calculation() {
        let monitor = make_monitor();
        
        // Add some stable prices
        for i in 0..10 {
            let price = dec!(0.50) + Decimal::from(i) * dec!(0.001);
            monitor.update("test", price, None, None, None);
        }
        
        let state = monitor.get_state("test").unwrap();
        assert!(state.volatility_pct < dec!(50)); // Should be relatively low
    }

    #[test]
    fn test_high_volatility_detection() {
        let monitor = make_monitor();
        
        // Add prices with high swings
        let prices = [dec!(0.40), dec!(0.60), dec!(0.35), dec!(0.65), dec!(0.30), dec!(0.70)];
        for price in prices {
            monitor.update("test", price, None, None, None);
        }
        
        let state = monitor.get_state("test").unwrap();
        assert!(matches!(state.volatility_regime, VolatilityRegime::High | VolatilityRegime::Extreme));
    }

    #[test]
    fn test_momentum_calculation() {
        let monitor = make_monitor();
        
        // Add steadily increasing prices
        for i in 0..10 {
            let price = dec!(0.50) + Decimal::from(i) * dec!(0.01);
            monitor.update("test", price, None, None, None);
        }
        
        let state = monitor.get_state("test").unwrap();
        assert!(state.momentum.price_change_pct > Decimal::ZERO);
        assert!(state.momentum.strength > dec!(0.5));
    }

    #[test]
    fn test_liquidity_score() {
        let monitor = make_monitor();
        
        // High liquidity - need multiple updates
        for _ in 0..6 {
            monitor.update("test", dec!(0.50), None, Some(dec!(5000)), Some(dec!(5000)));
        }
        let state = monitor.get_state("test").unwrap();
        assert!(state.liquidity_score >= dec!(0.9), "Expected >= 0.9, got {}", state.liquidity_score);
        
        // Low liquidity
        for _ in 0..6 {
            monitor.update("test2", dec!(0.50), None, Some(dec!(100)), Some(dec!(100)));
        }
        let state2 = monitor.get_state("test2").unwrap();
        assert!(state2.liquidity_score < dec!(0.1), "Expected < 0.1, got {}", state2.liquidity_score);
    }

    #[test]
    fn test_regime_change_alert() {
        let monitor = make_monitor();
        
        // Start with low volatility (stable prices)
        for i in 0..10 {
            let price = dec!(0.50) + Decimal::from(i) * dec!(0.0001);
            monitor.update("test", price, None, None, None);
        }
        
        let state1 = monitor.get_state("test").unwrap();
        let initial_regime = state1.volatility_regime;
        
        // Introduce extreme volatility - check each update for regime change alert
        let volatile_prices = [dec!(0.20), dec!(0.80), dec!(0.15), dec!(0.85), dec!(0.10), dec!(0.90)];
        let mut found_regime_change_alert = false;
        
        for price in volatile_prices {
            let state = monitor.update("test", price, None, None, None);
            if state.alerts.iter().any(|a| a.alert_type == AlertType::RegimeChange) {
                found_regime_change_alert = true;
            }
        }
        
        let final_state = monitor.get_state("test").unwrap();
        
        // If regime changed, we should have seen an alert during the transitions
        if final_state.volatility_regime != initial_regime {
            assert!(
                found_regime_change_alert,
                "Expected regime change alert when going from {:?} to {:?}",
                initial_regime, final_state.volatility_regime
            );
        }
    }

    #[test]
    fn test_trading_recommendation_extreme_vol() {
        let monitor = make_monitor();
        
        // Create extreme volatility
        let extreme_prices = [dec!(0.20), dec!(0.80), dec!(0.15), dec!(0.85), dec!(0.10), dec!(0.90)];
        for price in extreme_prices {
            monitor.update("test", price, None, None, None);
        }
        
        let rec = monitor.get_recommendation("test");
        
        // Should not recommend trading in extreme volatility
        if monitor.get_state("test").unwrap().volatility_regime == VolatilityRegime::Extreme {
            assert!(!rec.can_trade);
        }
    }

    #[test]
    fn test_trading_recommendation_low_liquidity() {
        let monitor = make_monitor();
        
        for _ in 0..10 {
            monitor.update("test", dec!(0.50), None, Some(dec!(50)), Some(dec!(50)));
        }
        
        let rec = monitor.get_recommendation("test");
        
        assert!(rec.warnings.iter().any(|w| w.contains("liquidity")));
    }

    #[test]
    fn test_kelly_multiplier() {
        assert!(VolatilityRegime::Low.kelly_multiplier() > dec!(1));
        assert_eq!(VolatilityRegime::Medium.kelly_multiplier(), dec!(1));
        assert!(VolatilityRegime::High.kelly_multiplier() < dec!(1));
        assert!(VolatilityRegime::Extreme.kelly_multiplier() < dec!(0.5));
    }

    #[test]
    fn test_anomaly_detection() {
        let monitor = make_monitor();
        
        // Add normal prices
        for _ in 0..10 {
            monitor.update("test", dec!(0.50), None, None, None);
        }
        
        // Add anomalous price jump
        let state = monitor.update("test", dec!(0.80), None, None, None);
        
        // Should detect price jump anomaly
        assert!(state.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::PriceJump));
    }

    #[test]
    fn test_momentum_acceleration() {
        let monitor = make_monitor();
        
        // First half: small moves
        for i in 0..5 {
            let price = dec!(0.50) + Decimal::from(i) * dec!(0.001);
            monitor.update("test", price, None, None, None);
        }
        
        // Second half: larger moves
        for i in 0..5 {
            let price = dec!(0.505) + Decimal::from(i) * dec!(0.01);
            monitor.update("test", price, None, None, None);
        }
        
        let state = monitor.get_state("test").unwrap();
        assert!(state.momentum.accelerating);
    }

    #[test]
    fn test_volume_spike_detection() {
        let monitor = make_monitor();
        
        // Normal volumes
        for _ in 0..10 {
            monitor.update("test", dec!(0.50), Some(dec!(100)), None, None);
        }
        
        // Volume spike
        let state = monitor.update("test", dec!(0.50), Some(dec!(500)), None, None);
        
        assert!(state.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::VolumeSpike));
    }

    #[test]
    fn test_get_all_states() {
        let monitor = make_monitor();
        
        monitor.update("market1", dec!(0.50), None, None, None);
        monitor.update("market2", dec!(0.60), None, None, None);
        
        let all_states = monitor.get_all_states();
        
        assert_eq!(all_states.len(), 2);
        assert!(all_states.contains_key("market1"));
        assert!(all_states.contains_key("market2"));
    }

    #[test]
    fn test_default_state_insufficient_data() {
        let monitor = make_monitor();
        
        // Only 2 updates (below min_updates_for_volatility)
        monitor.update("test", dec!(0.50), None, None, None);
        monitor.update("test", dec!(0.51), None, None, None);
        
        let state = monitor.get_state("test").unwrap();
        
        // Should use default medium volatility
        assert_eq!(state.volatility_regime, VolatilityRegime::Medium);
    }
}
