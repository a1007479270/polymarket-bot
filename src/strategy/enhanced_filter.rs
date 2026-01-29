//! Enhanced Signal Filtering
//!
//! Advanced filtering that adds:
//! 1. Historical performance tracking per market/category
//! 2. Expected value threshold filtering
//! 3. Probability confidence bands (reject signals near 50%)
//! 4. Market correlation filtering (avoid correlated positions)
//! 5. Signal quality scoring

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::RwLock;

/// Enhanced filter configuration
#[derive(Debug, Clone)]
pub struct EnhancedFilterConfig {
    /// Minimum expected value to trade (e.g., 0.02 = 2%)
    pub min_expected_value: Decimal,
    /// Probability band to reject (signals near 50%)
    /// e.g., if this is 0.10, reject if model_prob is between 0.40 and 0.60
    pub uncertainty_band: Decimal,
    /// Minimum win rate for a category before allowing more trades
    pub min_category_win_rate: Decimal,
    /// Number of trades before applying win rate filter
    pub min_trades_for_stats: u32,
    /// Maximum correlation score for new positions (0-1)
    pub max_correlation: Decimal,
    /// Decay half-life for historical stats (days)
    pub stats_halflife_days: f64,
}

impl Default for EnhancedFilterConfig {
    fn default() -> Self {
        Self {
            min_expected_value: dec!(0.03),     // 3% minimum EV
            uncertainty_band: dec!(0.08),       // Reject 0.42-0.58 range
            min_category_win_rate: dec!(0.40),  // 40% minimum win rate
            min_trades_for_stats: 5,            // Need 5 trades before applying
            max_correlation: dec!(0.70),        // 70% max correlation
            stats_halflife_days: 14.0,          // 2-week half-life
        }
    }
}

/// Historical trade record for a market/category
#[derive(Debug, Clone)]
struct TradeRecord {
    timestamp: DateTime<Utc>,
    pnl: Decimal,
    category: String,
    market_id: String,
    model_confidence: Decimal,
}

/// Aggregated stats for a category
#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    pub total_trades: u32,
    pub wins: u32,
    pub losses: u32,
    pub total_pnl: Decimal,
    pub win_rate: Decimal,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub last_trade: Option<DateTime<Utc>>,
}

/// Market position for correlation checking
#[derive(Debug, Clone)]
pub struct OpenPosition {
    pub market_id: String,
    pub category: String,
    pub keywords: Vec<String>,
    pub direction: PositionDirection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionDirection {
    Long,
    Short,
}

/// Enhanced signal filter
pub struct EnhancedSignalFilter {
    config: EnhancedFilterConfig,
    /// Historical trades
    trade_history: RwLock<Vec<TradeRecord>>,
    /// Cached category stats
    category_stats: RwLock<HashMap<String, CategoryStats>>,
    /// Current open positions
    open_positions: RwLock<Vec<OpenPosition>>,
}

impl EnhancedSignalFilter {
    pub fn new(config: EnhancedFilterConfig) -> Self {
        Self {
            config,
            trade_history: RwLock::new(Vec::new()),
            category_stats: RwLock::new(HashMap::new()),
            open_positions: RwLock::new(Vec::new()),
        }
    }

    /// Main filter function - returns whether to trade and why
    pub fn should_trade(&self, signal: &SignalCandidate) -> FilterDecision {
        let mut checks = Vec::new();
        let mut passed = true;

        // 1. Expected Value Check
        let ev_check = self.check_expected_value(signal);
        if !ev_check.passed {
            passed = false;
        }
        checks.push(ev_check);

        // 2. Uncertainty Band Check
        let uncertainty_check = self.check_uncertainty_band(signal);
        if !uncertainty_check.passed {
            passed = false;
        }
        checks.push(uncertainty_check);

        // 3. Category Performance Check
        let category_check = self.check_category_performance(&signal.category);
        if !category_check.passed {
            passed = false;
        }
        checks.push(category_check);

        // 4. Correlation Check
        let correlation_check = self.check_correlation(signal);
        if !correlation_check.passed {
            passed = false;
        }
        checks.push(correlation_check);

        // 5. Signal Quality Score
        let quality = self.calculate_signal_quality(signal);
        
        // Calculate size multiplier before moving checks
        let size_multiplier = self.calculate_size_multiplier(&checks, quality);

        FilterDecision {
            should_trade: passed,
            checks,
            quality_score: quality,
            recommended_size_multiplier: size_multiplier,
        }
    }

    /// Check expected value threshold
    fn check_expected_value(&self, signal: &SignalCandidate) -> FilterCheck {
        // EV = edge * probability of being right - (1-probability) * potential loss
        // Simplified: EV ≈ edge * confidence
        let ev = signal.edge.abs() * signal.confidence;
        
        FilterCheck {
            name: "expected_value".to_string(),
            passed: ev >= self.config.min_expected_value,
            value: ev,
            threshold: self.config.min_expected_value,
            reason: format!(
                "EV {:.2}% {} min {:.2}%",
                ev * dec!(100),
                if ev >= self.config.min_expected_value { ">=" } else { "<" },
                self.config.min_expected_value * dec!(100)
            ),
        }
    }

    /// Check if probability is too uncertain (near 50%)
    fn check_uncertainty_band(&self, signal: &SignalCandidate) -> FilterCheck {
        let deviation_from_50 = (signal.model_probability - dec!(0.5)).abs();
        let min_deviation = self.config.uncertainty_band;
        
        FilterCheck {
            name: "uncertainty_band".to_string(),
            passed: deviation_from_50 >= min_deviation,
            value: deviation_from_50,
            threshold: min_deviation,
            reason: format!(
                "Prob {:.0}% is {:.0}% from 50% (min: {:.0}%)",
                signal.model_probability * dec!(100),
                deviation_from_50 * dec!(100),
                min_deviation * dec!(100)
            ),
        }
    }

    /// Check category historical performance
    fn check_category_performance(&self, category: &str) -> FilterCheck {
        let stats = self.category_stats.read().unwrap();
        
        match stats.get(category) {
            Some(cat_stats) if cat_stats.total_trades >= self.config.min_trades_for_stats => {
                FilterCheck {
                    name: "category_performance".to_string(),
                    passed: cat_stats.win_rate >= self.config.min_category_win_rate,
                    value: cat_stats.win_rate,
                    threshold: self.config.min_category_win_rate,
                    reason: format!(
                        "{}: {:.0}% win rate ({}/{} trades)",
                        category,
                        cat_stats.win_rate * dec!(100),
                        cat_stats.wins,
                        cat_stats.total_trades
                    ),
                }
            }
            _ => {
                // Not enough data - allow trading
                FilterCheck {
                    name: "category_performance".to_string(),
                    passed: true,
                    value: dec!(1),
                    threshold: self.config.min_category_win_rate,
                    reason: format!("{}: insufficient data (allowed)", category),
                }
            }
        }
    }

    /// Check correlation with existing positions
    fn check_correlation(&self, signal: &SignalCandidate) -> FilterCheck {
        let positions = self.open_positions.read().unwrap();
        
        if positions.is_empty() {
            return FilterCheck {
                name: "correlation".to_string(),
                passed: true,
                value: dec!(0),
                threshold: self.config.max_correlation,
                reason: "No existing positions".to_string(),
            };
        }

        let max_corr = positions
            .iter()
            .map(|p| self.calculate_correlation(signal, p))
            .fold(Decimal::ZERO, |acc, c| acc.max(c));

        FilterCheck {
            name: "correlation".to_string(),
            passed: max_corr < self.config.max_correlation,
            value: max_corr,
            threshold: self.config.max_correlation,
            reason: format!(
                "Max correlation {:.0}% with existing positions",
                max_corr * dec!(100)
            ),
        }
    }

    /// Calculate correlation between a signal and an existing position
    fn calculate_correlation(&self, signal: &SignalCandidate, position: &OpenPosition) -> Decimal {
        // Category match: 30% correlation
        let category_corr = if signal.category == position.category {
            dec!(0.30)
        } else {
            dec!(0)
        };

        // Keyword overlap: up to 40% correlation
        let keyword_overlap = signal
            .keywords
            .iter()
            .filter(|k| position.keywords.contains(k))
            .count();
        let total_keywords = signal.keywords.len().max(1);
        let keyword_corr = Decimal::from(keyword_overlap as u32) 
            / Decimal::from(total_keywords as u32) 
            * dec!(0.40);

        // Same direction in same category: additional 30%
        let direction_corr = if signal.category == position.category 
            && signal.direction == position.direction 
        {
            dec!(0.30)
        } else {
            dec!(0)
        };

        (category_corr + keyword_corr + direction_corr).min(dec!(1))
    }

    /// Calculate overall signal quality score (0-100)
    fn calculate_signal_quality(&self, signal: &SignalCandidate) -> Decimal {
        let mut score = dec!(50); // Base score

        // Edge contribution (up to +20)
        let edge_score = (signal.edge.abs() * dec!(100)).min(dec!(20));
        score += edge_score;

        // Confidence contribution (up to +20)
        let conf_score = signal.confidence * dec!(20);
        score += conf_score;

        // Probability clarity (distance from 50%) (up to +10)
        let clarity_score = (signal.model_probability - dec!(0.5)).abs() * dec!(20);
        score += clarity_score.min(dec!(10));

        // Category track record (up to +/-10)
        let stats = self.category_stats.read().unwrap();
        if let Some(cat_stats) = stats.get(&signal.category) {
            if cat_stats.total_trades >= self.config.min_trades_for_stats {
                let track_record = (cat_stats.win_rate - dec!(0.5)) * dec!(20);
                score += track_record;
            }
        }

        score.max(dec!(0)).min(dec!(100))
    }

    /// Calculate size multiplier based on filter results
    fn calculate_size_multiplier(&self, checks: &[FilterCheck], quality: Decimal) -> Decimal {
        if checks.iter().any(|c| !c.passed) {
            return dec!(0);
        }

        // Quality score affects size
        let quality_mult = quality / dec!(100);

        // How much margin we have on each check
        let margin_mult = checks
            .iter()
            .filter(|c| c.threshold > Decimal::ZERO)
            .map(|c| {
                let margin = (c.value - c.threshold) / c.threshold;
                (Decimal::ONE + margin * dec!(0.5)).min(dec!(1.5)) // Up to 50% boost
            })
            .fold(Decimal::ONE, |acc, m| acc * m);

        (quality_mult * margin_mult).min(dec!(1.5)).max(dec!(0.5))
    }

    /// Record a completed trade
    pub fn record_trade(&self, market_id: &str, category: &str, pnl: Decimal, confidence: Decimal) {
        let record = TradeRecord {
            timestamp: Utc::now(),
            pnl,
            category: category.to_string(),
            market_id: market_id.to_string(),
            model_confidence: confidence,
        };

        self.trade_history.write().unwrap().push(record);
        self.update_category_stats(category, pnl);
    }

    /// Update category statistics
    fn update_category_stats(&self, category: &str, pnl: Decimal) {
        let mut stats = self.category_stats.write().unwrap();
        let entry = stats.entry(category.to_string()).or_default();

        entry.total_trades += 1;
        entry.total_pnl += pnl;
        entry.last_trade = Some(Utc::now());

        if pnl > Decimal::ZERO {
            entry.wins += 1;
            entry.avg_win = (entry.avg_win * Decimal::from(entry.wins - 1) + pnl) 
                / Decimal::from(entry.wins);
        } else {
            entry.losses += 1;
            entry.avg_loss = (entry.avg_loss * Decimal::from(entry.losses - 1) + pnl.abs()) 
                / Decimal::from(entry.losses);
        }

        entry.win_rate = if entry.total_trades > 0 {
            Decimal::from(entry.wins) / Decimal::from(entry.total_trades)
        } else {
            dec!(0)
        };
    }

    /// Add an open position
    pub fn add_position(&self, position: OpenPosition) {
        self.open_positions.write().unwrap().push(position);
    }

    /// Remove a closed position
    pub fn remove_position(&self, market_id: &str) {
        self.open_positions
            .write()
            .unwrap()
            .retain(|p| p.market_id != market_id);
    }

    /// Get category stats
    pub fn get_category_stats(&self, category: &str) -> Option<CategoryStats> {
        self.category_stats.read().unwrap().get(category).cloned()
    }

    /// Get all category stats
    pub fn get_all_stats(&self) -> HashMap<String, CategoryStats> {
        self.category_stats.read().unwrap().clone()
    }

    /// Clean up old history (keep last 30 days)
    pub fn cleanup_history(&self) {
        let cutoff = Utc::now() - Duration::days(30);
        self.trade_history
            .write()
            .unwrap()
            .retain(|r| r.timestamp > cutoff);
    }
}

/// A signal candidate to evaluate
#[derive(Debug, Clone)]
pub struct SignalCandidate {
    pub market_id: String,
    pub category: String,
    pub keywords: Vec<String>,
    pub model_probability: Decimal,
    pub market_probability: Decimal,
    pub edge: Decimal,
    pub confidence: Decimal,
    pub direction: PositionDirection,
}

/// Individual filter check result
#[derive(Debug, Clone)]
pub struct FilterCheck {
    pub name: String,
    pub passed: bool,
    pub value: Decimal,
    pub threshold: Decimal,
    pub reason: String,
}

/// Overall filter decision
#[derive(Debug, Clone)]
pub struct FilterDecision {
    pub should_trade: bool,
    pub checks: Vec<FilterCheck>,
    pub quality_score: Decimal,
    pub recommended_size_multiplier: Decimal,
}

impl FilterDecision {
    pub fn summary(&self) -> String {
        let passed_count = self.checks.iter().filter(|c| c.passed).count();
        let failed: Vec<_> = self.checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        
        if self.should_trade {
            format!(
                "✅ PASS ({}/{}) Quality: {:.0} Size: {:.0}%",
                passed_count,
                self.checks.len(),
                self.quality_score,
                self.recommended_size_multiplier * dec!(100)
            )
        } else {
            format!(
                "❌ FAIL ({}/{}) Failed: {}",
                passed_count,
                self.checks.len(),
                failed.join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter() -> EnhancedSignalFilter {
        EnhancedSignalFilter::new(EnhancedFilterConfig::default())
    }

    fn make_signal(
        model_prob: Decimal,
        market_prob: Decimal,
        confidence: Decimal,
        category: &str,
    ) -> SignalCandidate {
        SignalCandidate {
            market_id: "test_market".to_string(),
            category: category.to_string(),
            keywords: vec!["btc".to_string(), "crypto".to_string()],
            model_probability: model_prob,
            market_probability: market_prob,
            edge: model_prob - market_prob,
            confidence,
            direction: if model_prob > market_prob {
                PositionDirection::Long
            } else {
                PositionDirection::Short
            },
        }
    }

    #[test]
    fn test_good_signal_passes() {
        let filter = make_filter();
        
        // Strong edge (70% vs 50%), high confidence
        let signal = make_signal(dec!(0.70), dec!(0.50), dec!(0.80), "crypto");
        let decision = filter.should_trade(&signal);
        
        assert!(decision.should_trade);
        println!("{}", decision.summary());
    }

    #[test]
    fn test_low_ev_rejected() {
        let filter = make_filter();
        
        // Small edge (52% vs 50%), even high confidence = low EV
        let signal = make_signal(dec!(0.52), dec!(0.50), dec!(0.80), "crypto");
        let decision = filter.should_trade(&signal);
        
        assert!(!decision.should_trade);
        assert!(decision.checks.iter().any(|c| c.name == "expected_value" && !c.passed));
        println!("{}", decision.summary());
    }

    #[test]
    fn test_uncertainty_band_rejected() {
        let filter = make_filter();
        
        // Model thinks 52%, too close to 50%
        let signal = make_signal(dec!(0.52), dec!(0.45), dec!(0.90), "crypto");
        let decision = filter.should_trade(&signal);
        
        assert!(!decision.should_trade);
        assert!(decision.checks.iter().any(|c| c.name == "uncertainty_band" && !c.passed));
        println!("{}", decision.summary());
    }

    #[test]
    fn test_category_performance_filter() {
        let filter = make_filter();
        
        // Record some losing trades in a category
        for _ in 0..5 {
            filter.record_trade("market_1", "politics", dec!(-10), dec!(0.70));
        }
        filter.record_trade("market_2", "politics", dec!(5), dec!(0.70)); // 1 win, 5 losses = 17%
        
        let signal = make_signal(dec!(0.70), dec!(0.50), dec!(0.80), "politics");
        let decision = filter.should_trade(&signal);
        
        assert!(!decision.should_trade);
        assert!(decision.checks.iter().any(|c| c.name == "category_performance" && !c.passed));
        println!("{}", decision.summary());
    }

    #[test]
    fn test_category_performance_insufficient_data() {
        let filter = make_filter();
        
        // Only 2 trades - not enough data
        filter.record_trade("market_1", "sports", dec!(-10), dec!(0.70));
        filter.record_trade("market_2", "sports", dec!(-10), dec!(0.70));
        
        let signal = make_signal(dec!(0.70), dec!(0.50), dec!(0.80), "sports");
        let decision = filter.should_trade(&signal);
        
        // Should pass because insufficient data
        let cat_check = decision.checks.iter().find(|c| c.name == "category_performance").unwrap();
        assert!(cat_check.passed);
    }

    #[test]
    fn test_correlation_filter() {
        let filter = make_filter();
        
        // Add existing position
        filter.add_position(OpenPosition {
            market_id: "btc_price_100k".to_string(),
            category: "crypto".to_string(),
            keywords: vec!["btc".to_string(), "bitcoin".to_string(), "price".to_string()],
            direction: PositionDirection::Long,
        });
        
        // Try to add similar position
        let signal = SignalCandidate {
            market_id: "btc_price_110k".to_string(),
            category: "crypto".to_string(),
            keywords: vec!["btc".to_string(), "bitcoin".to_string()],
            model_probability: dec!(0.70),
            market_probability: dec!(0.50),
            edge: dec!(0.20),
            confidence: dec!(0.80),
            direction: PositionDirection::Long,
        };
        
        let decision = filter.should_trade(&signal);
        
        // Should fail correlation check
        assert!(!decision.should_trade);
        assert!(decision.checks.iter().any(|c| c.name == "correlation" && !c.passed));
        println!("{}", decision.summary());
    }

    #[test]
    fn test_uncorrelated_positions_allowed() {
        let filter = make_filter();
        
        // Add crypto position
        filter.add_position(OpenPosition {
            market_id: "btc_price".to_string(),
            category: "crypto".to_string(),
            keywords: vec!["btc".to_string()],
            direction: PositionDirection::Long,
        });
        
        // Politics position should be uncorrelated
        let signal = make_signal(dec!(0.70), dec!(0.50), dec!(0.80), "politics");
        let decision = filter.should_trade(&signal);
        
        let corr_check = decision.checks.iter().find(|c| c.name == "correlation").unwrap();
        assert!(corr_check.passed);
    }

    #[test]
    fn test_quality_score_calculation() {
        let filter = make_filter();
        
        // High quality signal
        let good_signal = make_signal(dec!(0.75), dec!(0.50), dec!(0.90), "crypto");
        let good_decision = filter.should_trade(&good_signal);
        
        // Lower quality signal
        let ok_signal = make_signal(dec!(0.62), dec!(0.50), dec!(0.65), "crypto");
        let ok_decision = filter.should_trade(&ok_signal);
        
        assert!(good_decision.quality_score > ok_decision.quality_score);
        println!("Good signal quality: {}", good_decision.quality_score);
        println!("OK signal quality: {}", ok_decision.quality_score);
    }

    #[test]
    fn test_size_multiplier_calculation() {
        let filter = make_filter();
        
        // Excellent signal should get size boost
        let signal = make_signal(dec!(0.80), dec!(0.50), dec!(0.90), "crypto");
        let decision = filter.should_trade(&signal);
        
        assert!(decision.should_trade);
        assert!(decision.recommended_size_multiplier > dec!(0.8));
        println!("Size multiplier: {}", decision.recommended_size_multiplier);
    }

    #[test]
    fn test_remove_position() {
        let filter = make_filter();
        
        filter.add_position(OpenPosition {
            market_id: "test_market".to_string(),
            category: "crypto".to_string(),
            keywords: vec!["btc".to_string()],
            direction: PositionDirection::Long,
        });
        
        assert_eq!(filter.open_positions.read().unwrap().len(), 1);
        
        filter.remove_position("test_market");
        
        assert_eq!(filter.open_positions.read().unwrap().len(), 0);
    }

    #[test]
    fn test_category_stats_tracking() {
        let filter = make_filter();
        
        filter.record_trade("m1", "crypto", dec!(100), dec!(0.80));
        filter.record_trade("m2", "crypto", dec!(50), dec!(0.75));
        filter.record_trade("m3", "crypto", dec!(-30), dec!(0.70));
        
        let stats = filter.get_category_stats("crypto").unwrap();
        
        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.wins, 2);
        assert_eq!(stats.losses, 1);
        assert_eq!(stats.total_pnl, dec!(120));
        assert!(stats.win_rate > dec!(0.65)); // 2/3 = 66%
    }

    #[test]
    fn test_decision_summary() {
        let filter = make_filter();
        
        let signal = make_signal(dec!(0.70), dec!(0.50), dec!(0.80), "crypto");
        let decision = filter.should_trade(&signal);
        
        let summary = decision.summary();
        assert!(summary.contains("PASS") || summary.contains("FAIL"));
        println!("Summary: {}", summary);
    }

    #[test]
    fn test_edge_case_extreme_probabilities() {
        let filter = make_filter();
        
        // Very high probability signal
        let signal = make_signal(dec!(0.95), dec!(0.60), dec!(0.90), "crypto");
        let decision = filter.should_trade(&signal);
        
        assert!(decision.should_trade);
        assert!(decision.quality_score > dec!(70));
    }

    #[test]
    fn test_edge_case_negative_edge() {
        let filter = make_filter();
        
        // Model thinks market is overpriced (short signal)
        let signal = SignalCandidate {
            market_id: "test".to_string(),
            category: "crypto".to_string(),
            keywords: vec!["btc".to_string()],
            model_probability: dec!(0.30),
            market_probability: dec!(0.50),
            edge: dec!(-0.20),
            confidence: dec!(0.80),
            direction: PositionDirection::Short,
        };
        
        let decision = filter.should_trade(&signal);
        
        // Should still pass EV check because we use abs(edge)
        let ev_check = decision.checks.iter().find(|c| c.name == "expected_value").unwrap();
        assert!(ev_check.passed);
    }
}
