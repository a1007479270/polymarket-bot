//! Multi-factor fusion module
//!
//! Combines multiple prediction factors with:
//! - Factor orthogonalization to reduce redundancy
//! - Dynamic weight adjustment based on regime
//! - Risk parity weighting
//! - Information coefficient tracking

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Factor definition
#[derive(Debug, Clone)]
pub struct Factor {
    /// Unique factor identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Factor category (fundamental, technical, sentiment, etc.)
    pub category: FactorCategory,
    /// Current factor value (normalized -1 to 1)
    pub value: Decimal,
    /// Factor's prediction of probability direction
    pub signal: Decimal,
    /// Confidence in this factor
    pub confidence: Decimal,
    /// Timestamp of last update
    pub timestamp: DateTime<Utc>,
}

/// Factor category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactorCategory {
    /// Fundamental analysis (news, events)
    Fundamental,
    /// Technical indicators (price patterns)
    Technical,
    /// Market microstructure (order flow, liquidity)
    Microstructure,
    /// Sentiment analysis (social, news sentiment)
    Sentiment,
    /// On-chain data (blockchain metrics)
    OnChain,
    /// Copy trading signals
    CopyTrade,
    /// Model predictions (ML/LLM)
    Model,
}

/// Factor weight configuration
#[derive(Debug, Clone)]
pub struct FactorWeight {
    pub factor_id: String,
    pub base_weight: Decimal,
    pub current_weight: Decimal,
    pub regime_multiplier: Decimal,
    pub ic_adjustment: Decimal,
}

/// Configuration for multi-factor fusion
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Whether to orthogonalize factors
    pub orthogonalize: bool,
    /// Use risk parity weighting
    pub risk_parity: bool,
    /// Minimum weight for any factor
    pub min_weight: Decimal,
    /// Maximum weight for any factor
    pub max_weight: Decimal,
    /// IC decay factor for historical performance
    pub ic_decay: Decimal,
    /// Window for IC calculation
    pub ic_window: usize,
    /// Category weight caps
    pub category_caps: HashMap<FactorCategory, Decimal>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        let mut category_caps = HashMap::new();
        category_caps.insert(FactorCategory::Model, dec!(0.4));
        category_caps.insert(FactorCategory::Technical, dec!(0.3));
        category_caps.insert(FactorCategory::Sentiment, dec!(0.2));
        category_caps.insert(FactorCategory::Microstructure, dec!(0.3));
        category_caps.insert(FactorCategory::Fundamental, dec!(0.3));
        category_caps.insert(FactorCategory::OnChain, dec!(0.2));
        category_caps.insert(FactorCategory::CopyTrade, dec!(0.3));
        
        Self {
            orthogonalize: true,
            risk_parity: true,
            min_weight: dec!(0.05),
            max_weight: dec!(0.4),
            ic_decay: dec!(0.95),
            ic_window: 100,
            category_caps,
        }
    }
}

/// Result of multi-factor fusion
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Combined signal (-1 to 1)
    pub signal: Decimal,
    /// Derived probability (0 to 1)
    pub probability: Decimal,
    /// Overall confidence
    pub confidence: Decimal,
    /// Factor diversity score (higher = more diversified)
    pub diversity: Decimal,
    /// Individual factor contributions
    pub contributions: Vec<FactorContribution>,
    /// Active category weights
    pub category_weights: HashMap<FactorCategory, Decimal>,
}

/// Individual factor's contribution
#[derive(Debug, Clone)]
pub struct FactorContribution {
    pub factor_id: String,
    pub category: FactorCategory,
    pub raw_signal: Decimal,
    pub orthogonalized_signal: Decimal,
    pub weight: Decimal,
    pub contribution: Decimal,
    pub ic: Decimal,
}

/// Factor performance tracking
#[derive(Debug, Clone)]
struct FactorPerformance {
    factor_id: String,
    predictions: Vec<(Decimal, bool)>, // (signal, outcome)
    information_coefficient: Decimal,
    hit_rate: Decimal,
    volatility: Decimal,
}

impl FactorPerformance {
    fn new(factor_id: &str) -> Self {
        Self {
            factor_id: factor_id.to_string(),
            predictions: Vec::new(),
            information_coefficient: dec!(0.0),
            hit_rate: dec!(0.5),
            volatility: dec!(0.1),
        }
    }
    
    fn update(&mut self, window: usize) {
        if self.predictions.is_empty() {
            return;
        }
        
        // Keep only recent predictions
        if self.predictions.len() > window {
            self.predictions = self.predictions[self.predictions.len() - window..].to_vec();
        }
        
        // Calculate IC (correlation between signal and outcome)
        self.information_coefficient = self.calculate_ic();
        
        // Calculate hit rate
        let correct = self.predictions.iter()
            .filter(|(signal, outcome)| {
                (*signal > Decimal::ZERO) == *outcome
            })
            .count();
        self.hit_rate = Decimal::from(correct as i64) / Decimal::from(self.predictions.len() as i64);
        
        // Calculate signal volatility
        self.volatility = self.calculate_volatility();
    }
    
    fn calculate_ic(&self) -> Decimal {
        if self.predictions.len() < 3 {
            return Decimal::ZERO;
        }
        
        let n = Decimal::from(self.predictions.len() as i64);
        
        // Convert outcomes to +1/-1
        let signals: Vec<Decimal> = self.predictions.iter().map(|(s, _)| *s).collect();
        let outcomes: Vec<Decimal> = self.predictions.iter()
            .map(|(_, o)| if *o { dec!(1.0) } else { dec!(-1.0) })
            .collect();
        
        // Pearson correlation
        let mean_signal = signals.iter().sum::<Decimal>() / n;
        let mean_outcome = outcomes.iter().sum::<Decimal>() / n;
        
        let mut cov = Decimal::ZERO;
        let mut var_signal = Decimal::ZERO;
        let mut var_outcome = Decimal::ZERO;
        
        for i in 0..self.predictions.len() {
            let ds = signals[i] - mean_signal;
            let do_ = outcomes[i] - mean_outcome;
            cov += ds * do_;
            var_signal += ds * ds;
            var_outcome += do_ * do_;
        }
        
        if var_signal == Decimal::ZERO || var_outcome == Decimal::ZERO {
            return Decimal::ZERO;
        }
        
        cov / sqrt_decimal(var_signal * var_outcome)
    }
    
    fn calculate_volatility(&self) -> Decimal {
        if self.predictions.len() < 3 {
            return dec!(0.1);
        }
        
        let signals: Vec<Decimal> = self.predictions.iter().map(|(s, _)| *s).collect();
        let n = Decimal::from(signals.len() as i64);
        let mean = signals.iter().sum::<Decimal>() / n;
        
        let variance = signals.iter()
            .map(|s| (*s - mean) * (*s - mean))
            .sum::<Decimal>() / n;
        
        sqrt_decimal(variance)
    }
}

/// Multi-factor fusion engine
pub struct MultiFactorFusion {
    config: FusionConfig,
    factor_performance: HashMap<String, FactorPerformance>,
    base_weights: HashMap<String, Decimal>,
}

impl MultiFactorFusion {
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            factor_performance: HashMap::new(),
            base_weights: HashMap::new(),
        }
    }
    
    pub fn with_defaults() -> Self {
        Self::new(FusionConfig::default())
    }
    
    /// Set base weight for a factor
    pub fn set_factor_weight(&mut self, factor_id: &str, weight: Decimal) {
        self.base_weights.insert(factor_id.to_string(), weight);
    }
    
    /// Fuse multiple factors into a single prediction
    pub fn fuse(&self, factors: &[Factor]) -> FusionResult {
        if factors.is_empty() {
            return FusionResult {
                signal: Decimal::ZERO,
                probability: dec!(0.5),
                confidence: Decimal::ZERO,
                diversity: Decimal::ZERO,
                contributions: Vec::new(),
                category_weights: HashMap::new(),
            };
        }
        
        // Step 1: Orthogonalize factors if configured
        let signals: Vec<Decimal> = if self.config.orthogonalize && factors.len() > 1 {
            self.orthogonalize_factors(factors)
        } else {
            factors.iter().map(|f| f.signal).collect()
        };
        
        // Step 2: Calculate weights
        let weights = self.calculate_weights(factors);
        
        // Step 3: Apply category caps
        let capped_weights = self.apply_category_caps(factors, &weights);
        
        // Step 4: Combine signals
        let mut weighted_signal = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut contributions = Vec::new();
        let mut category_weights: HashMap<FactorCategory, Decimal> = HashMap::new();
        
        for (i, factor) in factors.iter().enumerate() {
            let weight = capped_weights[i];
            let signal = signals[i];
            
            weighted_signal += signal * weight;
            total_weight += weight;
            
            *category_weights.entry(factor.category).or_insert(Decimal::ZERO) += weight;
            
            let ic = self.factor_performance
                .get(&factor.id)
                .map(|p| p.information_coefficient)
                .unwrap_or(Decimal::ZERO);
            
            contributions.push(FactorContribution {
                factor_id: factor.id.clone(),
                category: factor.category,
                raw_signal: factor.signal,
                orthogonalized_signal: signal,
                weight,
                contribution: signal * weight,
                ic,
            });
        }
        
        let signal = if total_weight > Decimal::ZERO {
            (weighted_signal / total_weight).max(dec!(-1.0)).min(dec!(1.0))
        } else {
            Decimal::ZERO
        };
        
        // Convert signal to probability
        let probability = (signal + dec!(1.0)) / dec!(2.0);
        
        // Calculate confidence
        let avg_confidence = factors.iter().map(|f| f.confidence).sum::<Decimal>()
            / Decimal::from(factors.len() as i64);
        let diversity = self.calculate_diversity(factors);
        
        // Confidence adjusted by diversity (more diverse = more confident)
        let confidence = avg_confidence * (dec!(0.5) + diversity * dec!(0.5));
        
        FusionResult {
            signal,
            probability,
            confidence,
            diversity,
            contributions,
            category_weights,
        }
    }
    
    /// Record factor outcome for performance tracking
    pub fn record_outcome(&mut self, factor_id: &str, signal: Decimal, outcome: bool) {
        let perf = self.factor_performance
            .entry(factor_id.to_string())
            .or_insert_with(|| FactorPerformance::new(factor_id));
        
        perf.predictions.push((signal, outcome));
        perf.update(self.config.ic_window);
    }
    
    /// Orthogonalize factors using Gram-Schmidt
    fn orthogonalize_factors(&self, factors: &[Factor]) -> Vec<Decimal> {
        let n = factors.len();
        let mut signals: Vec<Decimal> = factors.iter().map(|f| f.signal).collect();
        
        // Simple Gram-Schmidt orthogonalization
        for i in 1..n {
            for j in 0..i {
                let signal_i = signals[i];
                let signal_j = signals[j];
                let dot = signal_i * signal_j;
                let norm_sq = signal_j * signal_j;
                
                if norm_sq > dec!(0.0001) {
                    signals[i] = signal_i - (dot / norm_sq) * signal_j;
                }
            }
        }
        
        // Normalize
        for signal in &mut signals {
            let magnitude = signal.abs();
            if magnitude > dec!(1.0) {
                *signal /= magnitude;
            }
        }
        
        signals
    }
    
    /// Calculate factor weights
    fn calculate_weights(&self, factors: &[Factor]) -> Vec<Decimal> {
        let mut weights = Vec::new();
        
        for factor in factors {
            // Start with base weight
            let base = self.base_weights
                .get(&factor.id)
                .copied()
                .unwrap_or(dec!(0.2));
            
            // Adjust by IC if available
            let ic_adj = self.factor_performance
                .get(&factor.id)
                .map(|p| {
                    // IC ranges from -1 to 1, we want positive IC to increase weight
                    dec!(1.0) + p.information_coefficient * dec!(0.5)
                })
                .unwrap_or(dec!(1.0));
            
            // Risk parity adjustment
            let risk_adj = if self.config.risk_parity {
                let vol = self.factor_performance
                    .get(&factor.id)
                    .map(|p| p.volatility)
                    .unwrap_or(dec!(0.1));
                
                if vol > dec!(0.01) {
                    dec!(0.1) / vol
                } else {
                    dec!(1.0)
                }
            } else {
                dec!(1.0)
            };
            
            // Combine adjustments
            let weight = (base * ic_adj * risk_adj * factor.confidence)
                .max(self.config.min_weight)
                .min(self.config.max_weight);
            
            weights.push(weight);
        }
        
        // Normalize weights
        let total: Decimal = weights.iter().sum();
        if total > Decimal::ZERO {
            for w in &mut weights {
                *w /= total;
            }
        }
        
        weights
    }
    
    /// Apply category caps to weights
    fn apply_category_caps(&self, factors: &[Factor], weights: &[Decimal]) -> Vec<Decimal> {
        let mut capped = weights.to_vec();
        
        // Calculate category totals
        let mut category_totals: HashMap<FactorCategory, Decimal> = HashMap::new();
        for (i, factor) in factors.iter().enumerate() {
            *category_totals.entry(factor.category).or_insert(Decimal::ZERO) += weights[i];
        }
        
        // Apply caps
        for (i, factor) in factors.iter().enumerate() {
            if let Some(&cap) = self.config.category_caps.get(&factor.category) {
                if let Some(&total) = category_totals.get(&factor.category) {
                    if total > cap {
                        capped[i] = weights[i] * cap / total;
                    }
                }
            }
        }
        
        // Renormalize
        let total: Decimal = capped.iter().sum();
        if total > Decimal::ZERO {
            for w in &mut capped {
                *w /= total;
            }
        }
        
        capped
    }
    
    /// Calculate factor diversity (using category spread and correlation)
    fn calculate_diversity(&self, factors: &[Factor]) -> Decimal {
        if factors.len() < 2 {
            return Decimal::ZERO;
        }
        
        // Category diversity: how many categories are represented
        let mut categories: HashMap<FactorCategory, usize> = HashMap::new();
        for factor in factors {
            *categories.entry(factor.category).or_insert(0) += 1;
        }
        
        let category_count = categories.len();
        let max_categories = 7; // All possible categories
        let category_diversity = Decimal::from(category_count as i64) / Decimal::from(max_categories);
        
        // Signal diversity: how uncorrelated are the signals
        let signals: Vec<f64> = factors.iter()
            .map(|f| decimal_to_f64(f.signal))
            .collect();
        
        let mean: f64 = signals.iter().sum::<f64>() / signals.len() as f64;
        let variance: f64 = signals.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / signals.len() as f64;
        
        // Higher variance = more diversity
        let signal_diversity = f64_to_decimal((variance * 4.0).min(1.0));
        
        // Combined diversity score
        category_diversity * dec!(0.5) + signal_diversity * dec!(0.5)
    }
    
    /// Get factor statistics
    pub fn factor_stats(&self, factor_id: &str) -> Option<FactorStats> {
        self.factor_performance.get(factor_id).map(|p| FactorStats {
            factor_id: factor_id.to_string(),
            predictions_count: p.predictions.len(),
            information_coefficient: p.information_coefficient,
            hit_rate: p.hit_rate,
            volatility: p.volatility,
        })
    }
    
    /// Get all factor weights
    pub fn get_weights(&self, factors: &[Factor]) -> Vec<FactorWeight> {
        let weights = self.calculate_weights(factors);
        let capped = self.apply_category_caps(factors, &weights);
        
        factors.iter().zip(capped.iter()).map(|(f, &w)| {
            let ic_adj = self.factor_performance
                .get(&f.id)
                .map(|p| dec!(1.0) + p.information_coefficient * dec!(0.5))
                .unwrap_or(dec!(1.0));
            
            FactorWeight {
                factor_id: f.id.clone(),
                base_weight: self.base_weights.get(&f.id).copied().unwrap_or(dec!(0.2)),
                current_weight: w,
                regime_multiplier: dec!(1.0), // Could be adjusted by market regime
                ic_adjustment: ic_adj,
            }
        }).collect()
    }
}

/// Factor statistics
#[derive(Debug, Clone)]
pub struct FactorStats {
    pub factor_id: String,
    pub predictions_count: usize,
    pub information_coefficient: Decimal,
    pub hit_rate: Decimal,
    pub volatility: Decimal,
}

fn sqrt_decimal(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }
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

fn f64_to_decimal(f: f64) -> Decimal {
    use std::str::FromStr;
    if f.is_nan() || f.is_infinite() {
        return dec!(0.0);
    }
    Decimal::from_str(&format!("{:.6}", f)).unwrap_or(dec!(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_factor(id: &str, signal: Decimal, category: FactorCategory) -> Factor {
        Factor {
            id: id.to_string(),
            name: id.to_string(),
            category,
            value: signal,
            signal,
            confidence: dec!(0.8),
            timestamp: Utc::now(),
        }
    }
    
    #[test]
    fn test_basic_fusion() {
        let fusion = MultiFactorFusion::with_defaults();
        
        let factors = vec![
            make_factor("f1", dec!(0.5), FactorCategory::Technical),
            make_factor("f2", dec!(0.6), FactorCategory::Sentiment),
        ];
        
        let result = fusion.fuse(&factors);
        assert!(result.signal > Decimal::ZERO);
        assert!(result.probability > dec!(0.5));
    }
    
    #[test]
    fn test_empty_factors() {
        let fusion = MultiFactorFusion::with_defaults();
        let result = fusion.fuse(&[]);
        
        assert_eq!(result.signal, Decimal::ZERO);
        assert_eq!(result.probability, dec!(0.5));
    }
    
    #[test]
    fn test_opposing_signals() {
        let fusion = MultiFactorFusion::with_defaults();
        
        let factors = vec![
            make_factor("bull", dec!(0.8), FactorCategory::Technical),
            make_factor("bear", dec!(-0.8), FactorCategory::Sentiment),
        ];
        
        let result = fusion.fuse(&factors);
        // Should be near zero with opposing signals
        assert!(result.signal.abs() < dec!(0.5));
    }
    
    #[test]
    fn test_category_caps() {
        let mut config = FusionConfig::default();
        // Set a low cap for Technical
        config.category_caps.insert(FactorCategory::Technical, dec!(0.3));
        // Set a high cap for Sentiment so it doesn't get capped
        config.category_caps.insert(FactorCategory::Sentiment, dec!(0.9));
        
        let mut fusion = MultiFactorFusion::new(config);
        fusion.set_factor_weight("t1", dec!(0.5));
        fusion.set_factor_weight("t2", dec!(0.5));
        fusion.set_factor_weight("s1", dec!(0.2));
        
        let factors = vec![
            make_factor("t1", dec!(0.6), FactorCategory::Technical),
            make_factor("t2", dec!(0.7), FactorCategory::Technical),
            make_factor("s1", dec!(0.5), FactorCategory::Sentiment),
        ];
        
        let result = fusion.fuse(&factors);
        
        // After capping and renormalization, Technical category should have reduced share
        // compared to its uncapped proportion
        let tech_weight: Decimal = result.contributions.iter()
            .filter(|c| c.category == FactorCategory::Technical)
            .map(|c| c.weight)
            .sum();
        
        // With caps, technical shouldn't dominate completely
        // (without caps it would be close to 0.8-0.9 given the higher weights)
        assert!(tech_weight <= dec!(0.75));
    }
    
    #[test]
    fn test_diversity_calculation() {
        let fusion = MultiFactorFusion::with_defaults();
        
        // Single category factors
        let homogeneous = vec![
            make_factor("f1", dec!(0.5), FactorCategory::Technical),
            make_factor("f2", dec!(0.6), FactorCategory::Technical),
        ];
        
        // Multi-category factors
        let diverse = vec![
            make_factor("f1", dec!(0.5), FactorCategory::Technical),
            make_factor("f2", dec!(-0.3), FactorCategory::Sentiment),
            make_factor("f3", dec!(0.2), FactorCategory::OnChain),
        ];
        
        let result_homo = fusion.fuse(&homogeneous);
        let result_diverse = fusion.fuse(&diverse);
        
        assert!(result_diverse.diversity > result_homo.diversity);
    }
    
    #[test]
    fn test_ic_weighting() {
        let mut fusion = MultiFactorFusion::with_defaults();
        
        // Good factor: positive signal correlates with positive outcome
        for _ in 0..50 {
            fusion.record_outcome("good", dec!(0.5), true);
            fusion.record_outcome("good", dec!(-0.5), false);
        }
        
        // Bad factor: negative correlation
        for _ in 0..50 {
            fusion.record_outcome("bad", dec!(0.5), false);
            fusion.record_outcome("bad", dec!(-0.5), true);
        }
        
        let factors = vec![
            make_factor("good", dec!(0.6), FactorCategory::Technical),
            make_factor("bad", dec!(0.6), FactorCategory::Sentiment),
        ];
        
        let result = fusion.fuse(&factors);
        
        // Good factor should have higher weight
        let good_contrib = result.contributions.iter()
            .find(|c| c.factor_id == "good")
            .unwrap();
        let bad_contrib = result.contributions.iter()
            .find(|c| c.factor_id == "bad")
            .unwrap();
        
        assert!(good_contrib.weight > bad_contrib.weight);
    }
    
    #[test]
    fn test_orthogonalization() {
        let fusion = MultiFactorFusion::new(FusionConfig {
            orthogonalize: true,
            ..Default::default()
        });
        
        // Highly correlated factors
        let factors = vec![
            make_factor("f1", dec!(0.8), FactorCategory::Technical),
            make_factor("f2", dec!(0.79), FactorCategory::Technical), // Almost same signal
        ];
        
        let result = fusion.fuse(&factors);
        
        // After orthogonalization, second factor should have reduced contribution
        assert!(result.contributions[1].orthogonalized_signal.abs() < dec!(0.5));
    }
    
    #[test]
    fn test_factor_stats() {
        let mut fusion = MultiFactorFusion::with_defaults();
        
        // Record outcomes with positive correlation between signal and outcome
        // Positive signals -> true outcomes, negative signals -> false outcomes
        for i in 0..20 {
            let signal = if i % 2 == 0 { dec!(0.6) } else { dec!(-0.4) };
            let outcome = i % 2 == 0; // positive signal -> true, negative -> false
            fusion.record_outcome("test", signal, outcome);
        }
        
        let stats = fusion.factor_stats("test").unwrap();
        assert_eq!(stats.predictions_count, 20);
        // With correlated signals and outcomes, IC should be positive
        assert!(stats.information_coefficient > Decimal::ZERO);
    }
    
    #[test]
    fn test_probability_bounds() {
        let fusion = MultiFactorFusion::with_defaults();
        
        // Extreme signals
        let factors = vec![
            make_factor("extreme1", dec!(1.0), FactorCategory::Technical),
            make_factor("extreme2", dec!(1.0), FactorCategory::Sentiment),
        ];
        
        let result = fusion.fuse(&factors);
        assert!(result.probability >= Decimal::ZERO);
        assert!(result.probability <= Decimal::ONE);
    }
    
    #[test]
    fn test_get_weights() {
        let mut fusion = MultiFactorFusion::with_defaults();
        fusion.set_factor_weight("f1", dec!(0.6));
        
        let factors = vec![
            make_factor("f1", dec!(0.5), FactorCategory::Technical),
            make_factor("f2", dec!(0.6), FactorCategory::Sentiment),
        ];
        
        let weights = fusion.get_weights(&factors);
        assert_eq!(weights.len(), 2);
        assert!(weights[0].base_weight > weights[1].base_weight);
    }
    
    #[test]
    fn test_contributions_sum_to_one() {
        let fusion = MultiFactorFusion::with_defaults();
        
        let factors = vec![
            make_factor("f1", dec!(0.3), FactorCategory::Technical),
            make_factor("f2", dec!(0.5), FactorCategory::Sentiment),
            make_factor("f3", dec!(0.7), FactorCategory::Model),
        ];
        
        let result = fusion.fuse(&factors);
        let weight_sum: Decimal = result.contributions.iter().map(|c| c.weight).sum();
        
        assert!((weight_sum - dec!(1.0)).abs() < dec!(0.01));
    }
}
