//! Feature engineering for ML prediction
//!
//! Extracts meaningful features from raw market data:
//! - Price-based features (momentum, volatility, trends)
//! - Volume features (VWAP deviation, volume profile)
//! - Order book features (depth imbalance, spread dynamics)
//! - Time features (time to expiry, day of week effects)

use chrono::{DateTime, Utc, Datelike, Timelike};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Window size for short-term features (e.g., 5 minutes)
    pub short_window: usize,
    /// Window size for medium-term features (e.g., 1 hour)
    pub medium_window: usize,
    /// Window size for long-term features (e.g., 24 hours)
    pub long_window: usize,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            short_window: 5,
            medium_window: 60,
            long_window: 1440,
            min_data_points: 10,
            normalize: true,
        }
    }
}

/// Extracted features from market data
#[derive(Debug, Clone, Default)]
pub struct MarketFeatures {
    // Price features
    pub price_momentum_short: Decimal,
    pub price_momentum_medium: Decimal,
    pub price_momentum_long: Decimal,
    pub price_volatility_short: Decimal,
    pub price_volatility_medium: Decimal,
    pub price_acceleration: Decimal,
    pub price_mean_reversion: Decimal,
    
    // Volume features
    pub volume_momentum: Decimal,
    pub vwap_deviation: Decimal,
    pub volume_concentration: Decimal,
    pub buy_sell_ratio: Decimal,
    
    // Order book features
    pub bid_ask_spread: Decimal,
    pub depth_imbalance: Decimal,
    pub spread_volatility: Decimal,
    pub mid_price_stability: Decimal,
    
    // Time features
    pub time_to_expiry_hours: Decimal,
    pub hour_of_day: Decimal,
    pub day_of_week: Decimal,
    pub is_weekend: bool,
    
    // Derived features
    pub trend_strength: Decimal,
    pub regime_indicator: Decimal,
    pub liquidity_score: Decimal,
    
    // Feature quality
    pub data_completeness: Decimal,
    pub feature_timestamp: DateTime<Utc>,
}

impl MarketFeatures {
    /// Convert to feature vector for ML models
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            decimal_to_f64(self.price_momentum_short),
            decimal_to_f64(self.price_momentum_medium),
            decimal_to_f64(self.price_momentum_long),
            decimal_to_f64(self.price_volatility_short),
            decimal_to_f64(self.price_volatility_medium),
            decimal_to_f64(self.price_acceleration),
            decimal_to_f64(self.price_mean_reversion),
            decimal_to_f64(self.volume_momentum),
            decimal_to_f64(self.vwap_deviation),
            decimal_to_f64(self.volume_concentration),
            decimal_to_f64(self.buy_sell_ratio),
            decimal_to_f64(self.bid_ask_spread),
            decimal_to_f64(self.depth_imbalance),
            decimal_to_f64(self.spread_volatility),
            decimal_to_f64(self.mid_price_stability),
            decimal_to_f64(self.time_to_expiry_hours),
            decimal_to_f64(self.hour_of_day),
            decimal_to_f64(self.day_of_week),
            if self.is_weekend { 1.0 } else { 0.0 },
            decimal_to_f64(self.trend_strength),
            decimal_to_f64(self.regime_indicator),
            decimal_to_f64(self.liquidity_score),
        ]
    }
    
    /// Feature names for interpretability
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "price_momentum_short",
            "price_momentum_medium",
            "price_momentum_long",
            "price_volatility_short",
            "price_volatility_medium",
            "price_acceleration",
            "price_mean_reversion",
            "volume_momentum",
            "vwap_deviation",
            "volume_concentration",
            "buy_sell_ratio",
            "bid_ask_spread",
            "depth_imbalance",
            "spread_volatility",
            "mid_price_stability",
            "time_to_expiry_hours",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "trend_strength",
            "regime_indicator",
            "liquidity_score",
        ]
    }
}

/// Price/volume data point for feature calculation
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub volume: Decimal,
    pub bid_price: Option<Decimal>,
    pub ask_price: Option<Decimal>,
    pub bid_size: Option<Decimal>,
    pub ask_size: Option<Decimal>,
}

/// Feature extractor from market data
pub struct FeatureExtractor {
    config: FeatureConfig,
    price_history: VecDeque<(DateTime<Utc>, Decimal)>,
    volume_history: VecDeque<(DateTime<Utc>, Decimal)>,
    spread_history: VecDeque<Decimal>,
    vwap_numerator: Decimal,
    vwap_denominator: Decimal,
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            spread_history: VecDeque::new(),
            vwap_numerator: Decimal::ZERO,
            vwap_denominator: Decimal::ZERO,
        }
    }
    
    pub fn with_defaults() -> Self {
        Self::new(FeatureConfig::default())
    }
    
    /// Add a new data point
    pub fn update(&mut self, point: &DataPoint) {
        // Update price history
        self.price_history.push_back((point.timestamp, point.price));
        if self.price_history.len() > self.config.long_window {
            self.price_history.pop_front();
        }
        
        // Update volume history
        self.volume_history.push_back((point.timestamp, point.volume));
        if self.volume_history.len() > self.config.long_window {
            self.volume_history.pop_front();
        }
        
        // Update spread history
        if let (Some(bid), Some(ask)) = (point.bid_price, point.ask_price) {
            let spread = ask - bid;
            self.spread_history.push_back(spread);
            if self.spread_history.len() > self.config.medium_window {
                self.spread_history.pop_front();
            }
        }
        
        // Update VWAP
        self.vwap_numerator += point.price * point.volume;
        self.vwap_denominator += point.volume;
    }
    
    /// Extract all features from current state
    pub fn extract(&self, expiry: Option<DateTime<Utc>>, current_point: &DataPoint) -> MarketFeatures {
        let now = current_point.timestamp;
        let prices: Vec<Decimal> = self.price_history.iter().map(|(_, p)| *p).collect();
        let volumes: Vec<Decimal> = self.volume_history.iter().map(|(_, v)| *v).collect();
        
        let has_enough_data = prices.len() >= self.config.min_data_points;
        let data_completeness = if has_enough_data {
            Decimal::ONE
        } else {
            Decimal::from(prices.len() as i64) / Decimal::from(self.config.min_data_points as i64)
        };
        
        // Price momentum at different horizons
        let price_momentum_short = self.calculate_momentum(&prices, self.config.short_window);
        let price_momentum_medium = self.calculate_momentum(&prices, self.config.medium_window);
        let price_momentum_long = self.calculate_momentum(&prices, self.config.long_window);
        
        // Price volatility
        let price_volatility_short = self.calculate_volatility(&prices, self.config.short_window);
        let price_volatility_medium = self.calculate_volatility(&prices, self.config.medium_window);
        
        // Price acceleration (momentum of momentum)
        let price_acceleration = price_momentum_short - price_momentum_medium;
        
        // Mean reversion indicator
        let price_mean_reversion = self.calculate_mean_reversion(&prices);
        
        // Volume features
        let volume_momentum = self.calculate_momentum(&volumes, self.config.short_window);
        let vwap_deviation = self.calculate_vwap_deviation(current_point.price);
        let volume_concentration = self.calculate_volume_concentration(&volumes);
        let buy_sell_ratio = self.estimate_buy_sell_ratio(current_point);
        
        // Order book features
        let (bid_ask_spread, depth_imbalance) = self.calculate_book_features(current_point);
        let spread_volatility = self.calculate_volatility(
            &self.spread_history.iter().copied().collect::<Vec<_>>(),
            self.config.short_window
        );
        let mid_price_stability = dec!(1.0) - price_volatility_short.min(dec!(1.0));
        
        // Time features
        let time_to_expiry_hours = expiry
            .map(|e| {
                let duration = e.signed_duration_since(now);
                Decimal::from(duration.num_hours().max(0))
            })
            .unwrap_or(dec!(8760)); // Default 1 year if no expiry
        
        let hour_of_day = Decimal::from(now.hour());
        let day_of_week = Decimal::from(now.weekday().num_days_from_monday());
        let is_weekend = now.weekday().num_days_from_monday() >= 5;
        
        // Derived features
        let trend_strength = self.calculate_trend_strength(&prices);
        let regime_indicator = self.calculate_regime(&prices, &volumes);
        let liquidity_score = self.calculate_liquidity_score(
            bid_ask_spread,
            current_point.bid_size.unwrap_or(Decimal::ZERO),
            current_point.ask_size.unwrap_or(Decimal::ZERO),
        );
        
        MarketFeatures {
            price_momentum_short,
            price_momentum_medium,
            price_momentum_long,
            price_volatility_short,
            price_volatility_medium,
            price_acceleration,
            price_mean_reversion,
            volume_momentum,
            vwap_deviation,
            volume_concentration,
            buy_sell_ratio,
            bid_ask_spread,
            depth_imbalance,
            spread_volatility,
            mid_price_stability,
            time_to_expiry_hours,
            hour_of_day,
            day_of_week,
            is_weekend,
            trend_strength,
            regime_indicator,
            liquidity_score,
            data_completeness,
            feature_timestamp: now,
        }
    }
    
    /// Calculate momentum (rate of change) over a window
    fn calculate_momentum(&self, data: &[Decimal], window: usize) -> Decimal {
        if data.len() < 2 {
            return Decimal::ZERO;
        }
        
        let actual_window = window.min(data.len());
        let start_idx = data.len().saturating_sub(actual_window);
        let start = data[start_idx];
        let end = data[data.len() - 1];
        
        if start == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        (end - start) / start
    }
    
    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self, data: &[Decimal], window: usize) -> Decimal {
        if data.len() < 3 {
            return Decimal::ZERO;
        }
        
        let actual_window = window.min(data.len());
        let start_idx = data.len().saturating_sub(actual_window);
        let slice = &data[start_idx..];
        
        // Calculate returns
        let returns: Vec<Decimal> = slice.windows(2)
            .filter_map(|w| {
                if w[0] != Decimal::ZERO {
                    Some((w[1] - w[0]) / w[0])
                } else {
                    None
                }
            })
            .collect();
        
        if returns.is_empty() {
            return Decimal::ZERO;
        }
        
        // Calculate mean
        let mean = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as i64);
        
        // Calculate variance
        let variance = returns.iter()
            .map(|r| (*r - mean) * (*r - mean))
            .sum::<Decimal>() / Decimal::from(returns.len() as i64);
        
        // Return std dev (approximate sqrt)
        sqrt_decimal(variance)
    }
    
    /// Calculate mean reversion indicator
    fn calculate_mean_reversion(&self, data: &[Decimal]) -> Decimal {
        if data.len() < self.config.medium_window {
            return Decimal::ZERO;
        }
        
        let mean = data.iter().sum::<Decimal>() / Decimal::from(data.len() as i64);
        let current = data[data.len() - 1];
        
        if mean == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        // Negative when price above mean (expect reversion down)
        // Positive when price below mean (expect reversion up)
        (mean - current) / mean
    }
    
    /// Calculate VWAP deviation
    fn calculate_vwap_deviation(&self, current_price: Decimal) -> Decimal {
        if self.vwap_denominator == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        let vwap = self.vwap_numerator / self.vwap_denominator;
        
        if vwap == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        (current_price - vwap) / vwap
    }
    
    /// Calculate volume concentration (how concentrated volume is in recent trades)
    fn calculate_volume_concentration(&self, volumes: &[Decimal]) -> Decimal {
        if volumes.len() < self.config.short_window {
            return Decimal::ZERO;
        }
        
        let total: Decimal = volumes.iter().sum();
        if total == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        let recent_start = volumes.len().saturating_sub(self.config.short_window);
        let recent: Decimal = volumes[recent_start..].iter().sum();
        
        recent / total
    }
    
    /// Estimate buy/sell ratio from order flow
    fn estimate_buy_sell_ratio(&self, point: &DataPoint) -> Decimal {
        match (point.bid_size, point.ask_size) {
            (Some(bid), Some(ask)) if bid + ask > Decimal::ZERO => {
                bid / (bid + ask)
            }
            _ => dec!(0.5), // Neutral if no data
        }
    }
    
    /// Calculate order book features
    fn calculate_book_features(&self, point: &DataPoint) -> (Decimal, Decimal) {
        let spread = match (point.bid_price, point.ask_price) {
            (Some(bid), Some(ask)) if bid > Decimal::ZERO => {
                (ask - bid) / bid
            }
            _ => Decimal::ZERO,
        };
        
        let imbalance = match (point.bid_size, point.ask_size) {
            (Some(bid), Some(ask)) if bid + ask > Decimal::ZERO => {
                (bid - ask) / (bid + ask)
            }
            _ => Decimal::ZERO,
        };
        
        (spread, imbalance)
    }
    
    /// Calculate trend strength using linear regression slope
    fn calculate_trend_strength(&self, data: &[Decimal]) -> Decimal {
        if data.len() < 3 {
            return Decimal::ZERO;
        }
        
        let n = Decimal::from(data.len() as i64);
        
        // Calculate sums for linear regression
        let mut sum_x = Decimal::ZERO;
        let mut sum_y = Decimal::ZERO;
        let mut sum_xy = Decimal::ZERO;
        let mut sum_xx = Decimal::ZERO;
        
        for (i, y) in data.iter().enumerate() {
            let x = Decimal::from(i as i64);
            sum_x += x;
            sum_y += *y;
            sum_xy += x * *y;
            sum_xx += x * x;
        }
        
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        // Slope of linear regression
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        
        // Normalize by average price
        let avg_price = sum_y / n;
        if avg_price == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        slope / avg_price * Decimal::from(data.len() as i64)
    }
    
    /// Calculate market regime indicator
    fn calculate_regime(&self, prices: &[Decimal], volumes: &[Decimal]) -> Decimal {
        let vol_short = self.calculate_volatility(prices, self.config.short_window);
        let vol_medium = self.calculate_volatility(prices, self.config.medium_window);
        
        let vol_ratio = if vol_medium > Decimal::ZERO {
            vol_short / vol_medium
        } else {
            dec!(1.0)
        };
        
        let volume_trend = self.calculate_momentum(volumes, self.config.short_window);
        
        // Regime score: high = volatile/active, low = calm
        (vol_ratio + volume_trend.abs()) / dec!(2.0)
    }
    
    /// Calculate liquidity score
    fn calculate_liquidity_score(
        &self,
        spread: Decimal,
        bid_size: Decimal,
        ask_size: Decimal,
    ) -> Decimal {
        // Lower spread = more liquid
        let spread_score = (dec!(1.0) - spread * dec!(10.0)).max(Decimal::ZERO);
        
        // More depth = more liquid
        let depth = bid_size + ask_size;
        let depth_score = (depth / dec!(1000.0)).min(dec!(1.0));
        
        // Balance between bid and ask
        let balance_score = if depth > Decimal::ZERO {
            dec!(1.0) - ((bid_size - ask_size).abs() / depth)
        } else {
            Decimal::ZERO
        };
        
        // Weighted average
        spread_score * dec!(0.4) + depth_score * dec!(0.4) + balance_score * dec!(0.2)
    }
    
    /// Reset state
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.volume_history.clear();
        self.spread_history.clear();
        self.vwap_numerator = Decimal::ZERO;
        self.vwap_denominator = Decimal::ZERO;
    }
}

/// Approximate square root for Decimal
fn sqrt_decimal(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    // Newton-Raphson method
    let mut guess = x / dec!(2.0);
    for _ in 0..10 {
        if guess == Decimal::ZERO {
            return Decimal::ZERO;
        }
        guess = (guess + x / guess) / dec!(2.0);
    }
    guess
}

fn decimal_to_f64(d: Decimal) -> f64 {
    use std::str::FromStr;
    f64::from_str(&d.to_string()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    
    fn make_point(price: Decimal, volume: Decimal, ts_offset_mins: i64) -> DataPoint {
        DataPoint {
            timestamp: Utc::now() + Duration::minutes(ts_offset_mins),
            price,
            volume,
            bid_price: Some(price - dec!(0.01)),
            ask_price: Some(price + dec!(0.01)),
            bid_size: Some(dec!(100)),
            ask_size: Some(dec!(100)),
        }
    }
    
    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::with_defaults();
        assert_eq!(extractor.config.short_window, 5);
        assert_eq!(extractor.config.medium_window, 60);
    }
    
    #[test]
    fn test_momentum_calculation() {
        let mut extractor = FeatureExtractor::with_defaults();
        
        // Add increasing prices
        for i in 0..10 {
            let price = dec!(0.5) + Decimal::from(i) * dec!(0.01);
            extractor.update(&make_point(price, dec!(100), i));
        }
        
        let features = extractor.extract(None, &make_point(dec!(0.59), dec!(100), 10));
        assert!(features.price_momentum_short > Decimal::ZERO);
    }
    
    #[test]
    fn test_volatility_calculation() {
        let mut extractor = FeatureExtractor::with_defaults();
        
        // Add alternating prices (high volatility)
        for i in 0..20 {
            let price = if i % 2 == 0 { dec!(0.6) } else { dec!(0.4) };
            extractor.update(&make_point(price, dec!(100), i));
        }
        
        let features = extractor.extract(None, &make_point(dec!(0.5), dec!(100), 20));
        assert!(features.price_volatility_short > dec!(0.1));
    }
    
    #[test]
    fn test_spread_calculation() {
        let extractor = FeatureExtractor::with_defaults();
        let point = DataPoint {
            timestamp: Utc::now(),
            price: dec!(0.5),
            volume: dec!(100),
            bid_price: Some(dec!(0.49)),
            ask_price: Some(dec!(0.51)),
            bid_size: Some(dec!(100)),
            ask_size: Some(dec!(100)),
        };
        
        let features = extractor.extract(None, &point);
        assert!(features.bid_ask_spread > Decimal::ZERO);
        assert_eq!(features.depth_imbalance, Decimal::ZERO); // Equal bid/ask
    }
    
    #[test]
    fn test_time_features() {
        let extractor = FeatureExtractor::with_defaults();
        let expiry = Utc::now() + Duration::hours(24);
        let point = make_point(dec!(0.5), dec!(100), 0);
        
        let features = extractor.extract(Some(expiry), &point);
        assert!(features.time_to_expiry_hours >= dec!(23));
        assert!(features.time_to_expiry_hours <= dec!(25));
    }
    
    #[test]
    fn test_feature_vector() {
        let extractor = FeatureExtractor::with_defaults();
        let point = make_point(dec!(0.5), dec!(100), 0);
        let features = extractor.extract(None, &point);
        
        let vector = features.to_vector();
        assert_eq!(vector.len(), MarketFeatures::feature_names().len());
    }
    
    #[test]
    fn test_trend_strength() {
        let mut extractor = FeatureExtractor::with_defaults();
        
        // Strong uptrend
        for i in 0..20 {
            let price = dec!(0.3) + Decimal::from(i) * dec!(0.02);
            extractor.update(&make_point(price, dec!(100), i));
        }
        
        let features = extractor.extract(None, &make_point(dec!(0.68), dec!(100), 20));
        assert!(features.trend_strength > Decimal::ZERO);
    }
    
    #[test]
    fn test_mean_reversion_indicator() {
        let mut extractor = FeatureExtractor::new(FeatureConfig {
            min_data_points: 5,
            medium_window: 10,
            ..Default::default()
        });
        
        // Price recently spiked above mean
        for i in 0..15 {
            let price = if i < 10 { dec!(0.5) } else { dec!(0.7) };
            extractor.update(&make_point(price, dec!(100), i));
        }
        
        let features = extractor.extract(None, &make_point(dec!(0.7), dec!(100), 15));
        // Price above mean, expect negative (reversion down)
        assert!(features.price_mean_reversion < Decimal::ZERO);
    }
    
    #[test]
    fn test_liquidity_score() {
        let extractor = FeatureExtractor::with_defaults();
        
        // Good liquidity: tight spread, balanced depth
        let good = DataPoint {
            timestamp: Utc::now(),
            price: dec!(0.5),
            volume: dec!(100),
            bid_price: Some(dec!(0.499)),
            ask_price: Some(dec!(0.501)),
            bid_size: Some(dec!(500)),
            ask_size: Some(dec!(500)),
        };
        
        // Poor liquidity: wide spread, imbalanced
        let poor = DataPoint {
            timestamp: Utc::now(),
            price: dec!(0.5),
            volume: dec!(100),
            bid_price: Some(dec!(0.45)),
            ask_price: Some(dec!(0.55)),
            bid_size: Some(dec!(10)),
            ask_size: Some(dec!(100)),
        };
        
        let good_features = extractor.extract(None, &good);
        let poor_features = extractor.extract(None, &poor);
        
        assert!(good_features.liquidity_score > poor_features.liquidity_score);
    }
    
    #[test]
    fn test_data_completeness() {
        let extractor = FeatureExtractor::with_defaults();
        let point = make_point(dec!(0.5), dec!(100), 0);
        
        // Not enough data
        let features = extractor.extract(None, &point);
        assert!(features.data_completeness < Decimal::ONE);
    }
}
