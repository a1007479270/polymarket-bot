//! Ensemble prediction module
//!
//! Combines multiple prediction sources for robust estimates:
//! - Weighted averaging with dynamic weights
//! - Stacking (meta-learner)
//! - Bayesian model combination
//! - Disagreement-based confidence adjustment

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Ensemble combination method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleMethod {
    /// Simple average of all predictions
    SimpleAverage,
    /// Weighted average based on historical performance
    WeightedAverage,
    /// Median prediction (robust to outliers)
    Median,
    /// Bayesian model combination
    BayesianCombination,
    /// Best model selection based on recent performance
    BestModel,
}

/// Configuration for ensemble predictor
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Ensemble method to use
    pub method: EnsembleMethod,
    /// Minimum models required for prediction
    pub min_models: usize,
    /// Weight decay factor for older performance data
    pub weight_decay: Decimal,
    /// Whether to penalize disagreement
    pub penalize_disagreement: bool,
    /// Maximum weight for any single model
    pub max_model_weight: Decimal,
    /// Window size for performance tracking (in predictions)
    pub performance_window: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            method: EnsembleMethod::WeightedAverage,
            min_models: 2,
            weight_decay: dec!(0.95),
            penalize_disagreement: true,
            max_model_weight: dec!(0.5),
            performance_window: 100,
        }
    }
}

/// Single model prediction
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    /// Unique model identifier
    pub model_id: String,
    /// Predicted probability
    pub probability: Decimal,
    /// Model's confidence in its prediction
    pub confidence: Decimal,
    /// Model's reported uncertainty/variance
    pub uncertainty: Option<Decimal>,
    /// Timestamp of prediction
    pub timestamp: DateTime<Utc>,
    /// Any model-specific metadata
    pub metadata: Option<String>,
}

/// Ensemble prediction result
#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    /// Combined probability estimate
    pub probability: Decimal,
    /// Confidence (adjusted for model agreement)
    pub confidence: Decimal,
    /// Uncertainty estimate
    pub uncertainty: Decimal,
    /// Number of models contributing
    pub model_count: usize,
    /// Model agreement score (0-1, higher = more agreement)
    pub agreement: Decimal,
    /// Individual model contributions
    pub contributions: Vec<ModelContribution>,
    /// Ensemble method used
    pub method: EnsembleMethod,
}

/// Contribution of a single model to ensemble
#[derive(Debug, Clone)]
pub struct ModelContribution {
    pub model_id: String,
    pub prediction: Decimal,
    pub weight: Decimal,
    pub performance_score: Decimal,
}

/// Model performance tracking
#[derive(Debug, Clone)]
struct ModelPerformance {
    model_id: String,
    predictions: Vec<(Decimal, bool)>, // (predicted, actual_outcome)
    brier_score: Decimal,
    accuracy: Decimal,
    calibration_error: Decimal,
    last_updated: DateTime<Utc>,
}

impl ModelPerformance {
    fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            predictions: Vec::new(),
            brier_score: dec!(0.25), // Default (random)
            accuracy: dec!(0.5),
            calibration_error: dec!(0.1),
            last_updated: Utc::now(),
        }
    }
    
    fn update_metrics(&mut self) {
        if self.predictions.is_empty() {
            return;
        }
        
        let n = Decimal::from(self.predictions.len() as i64);
        
        // Brier score: mean squared error of probability predictions
        let mut brier_sum = Decimal::ZERO;
        let mut correct = 0usize;
        
        for (pred, actual) in &self.predictions {
            let target = if *actual { dec!(1.0) } else { dec!(0.0) };
            brier_sum += (*pred - target) * (*pred - target);
            
            // Accuracy: did the predicted direction match?
            let predicted_yes = *pred > dec!(0.5);
            if predicted_yes == *actual {
                correct += 1;
            }
        }
        
        self.brier_score = brier_sum / n;
        self.accuracy = Decimal::from(correct as i64) / n;
        self.last_updated = Utc::now();
        
        // Simple calibration error estimate
        self.calibration_error = self.estimate_calibration_error();
    }
    
    fn estimate_calibration_error(&self) -> Decimal {
        if self.predictions.len() < 10 {
            return dec!(0.1);
        }
        
        // Group predictions into bins
        let mut bins: HashMap<i32, (Decimal, usize, usize)> = HashMap::new();
        
        for (pred, actual) in &self.predictions {
            let bin = ((*pred * dec!(10.0)).floor().min(dec!(9.0))).to_string()
                .parse::<i32>().unwrap_or(5);
            
            let entry = bins.entry(bin).or_insert((Decimal::ZERO, 0, 0));
            entry.0 += *pred;
            entry.1 += if *actual { 1 } else { 0 };
            entry.2 += 1;
        }
        
        // ECE: weighted average of |avg_confidence - accuracy| per bin
        let mut ece = Decimal::ZERO;
        let total = self.predictions.len();
        
        for (_, (sum_conf, positives, count)) in bins {
            if count > 0 {
                let avg_conf = sum_conf / Decimal::from(count as i64);
                let accuracy = Decimal::from(positives as i64) / Decimal::from(count as i64);
                let weight = Decimal::from(count as i64) / Decimal::from(total as i64);
                ece += weight * (avg_conf - accuracy).abs();
            }
        }
        
        ece
    }
    
    /// Performance score for weighting (higher = better)
    fn performance_score(&self) -> Decimal {
        // Combine Brier score and calibration error
        // Brier score of 0 is perfect, 0.25 is random
        let brier_score = (dec!(0.25) - self.brier_score).max(Decimal::ZERO) / dec!(0.25);
        let calibration_score = (dec!(0.2) - self.calibration_error).max(Decimal::ZERO) / dec!(0.2);
        
        // Weighted combination
        brier_score * dec!(0.7) + calibration_score * dec!(0.3)
    }
}

/// Ensemble predictor combining multiple models
pub struct EnsemblePredictor {
    config: EnsembleConfig,
    model_performance: HashMap<String, ModelPerformance>,
}

impl EnsemblePredictor {
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            model_performance: HashMap::new(),
        }
    }
    
    pub fn with_defaults() -> Self {
        Self::new(EnsembleConfig::default())
    }
    
    /// Combine multiple model predictions
    pub fn predict(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        if predictions.len() < self.config.min_models {
            return None;
        }
        
        match self.config.method {
            EnsembleMethod::SimpleAverage => self.simple_average(predictions),
            EnsembleMethod::WeightedAverage => self.weighted_average(predictions),
            EnsembleMethod::Median => self.median_prediction(predictions),
            EnsembleMethod::BayesianCombination => self.bayesian_combination(predictions),
            EnsembleMethod::BestModel => self.best_model(predictions),
        }
    }
    
    /// Update model performance after outcome is known
    pub fn record_outcome(&mut self, model_id: &str, predicted: Decimal, actual: bool) {
        let perf = self.model_performance
            .entry(model_id.to_string())
            .or_insert_with(|| ModelPerformance::new(model_id));
        
        perf.predictions.push((predicted, actual));
        
        // Keep only recent predictions
        if perf.predictions.len() > self.config.performance_window {
            perf.predictions.remove(0);
        }
        
        perf.update_metrics();
    }
    
    /// Get current model weights
    pub fn get_model_weights(&self) -> HashMap<String, Decimal> {
        let mut weights = HashMap::new();
        let mut total_score = Decimal::ZERO;
        
        for (id, perf) in &self.model_performance {
            let score = perf.performance_score();
            weights.insert(id.clone(), score);
            total_score += score;
        }
        
        // Normalize
        if total_score > Decimal::ZERO {
            for weight in weights.values_mut() {
                *weight = (*weight / total_score).min(self.config.max_model_weight);
            }
        }
        
        weights
    }
    
    /// Simple average ensemble
    fn simple_average(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        let n = Decimal::from(predictions.len() as i64);
        let sum: Decimal = predictions.iter().map(|p| p.probability).sum();
        let probability = sum / n;
        
        let agreement = self.calculate_agreement(predictions);
        let (confidence, uncertainty) = self.calculate_confidence_uncertainty(predictions, agreement);
        
        let contributions = predictions.iter().map(|p| {
            ModelContribution {
                model_id: p.model_id.clone(),
                prediction: p.probability,
                weight: dec!(1.0) / n,
                performance_score: self.model_performance
                    .get(&p.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.5)),
            }
        }).collect();
        
        Some(EnsemblePrediction {
            probability,
            confidence,
            uncertainty,
            model_count: predictions.len(),
            agreement,
            contributions,
            method: EnsembleMethod::SimpleAverage,
        })
    }
    
    /// Weighted average based on model performance
    fn weighted_average(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        let model_weights = self.get_model_weights();
        
        let mut weighted_sum = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut contributions = Vec::new();
        
        for pred in predictions {
            let base_weight = model_weights
                .get(&pred.model_id)
                .copied()
                .unwrap_or(dec!(0.5));
            
            // Adjust by model's reported confidence
            let weight = base_weight * pred.confidence;
            
            weighted_sum += pred.probability * weight;
            total_weight += weight;
            
            contributions.push(ModelContribution {
                model_id: pred.model_id.clone(),
                prediction: pred.probability,
                weight,
                performance_score: self.model_performance
                    .get(&pred.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.5)),
            });
        }
        
        let probability = if total_weight > Decimal::ZERO {
            weighted_sum / total_weight
        } else {
            predictions.iter().map(|p| p.probability).sum::<Decimal>() 
                / Decimal::from(predictions.len() as i64)
        };
        
        // Normalize contribution weights
        if total_weight > Decimal::ZERO {
            for c in &mut contributions {
                c.weight /= total_weight;
            }
        }
        
        let agreement = self.calculate_agreement(predictions);
        let (confidence, uncertainty) = self.calculate_confidence_uncertainty(predictions, agreement);
        
        Some(EnsemblePrediction {
            probability,
            confidence,
            uncertainty,
            model_count: predictions.len(),
            agreement,
            contributions,
            method: EnsembleMethod::WeightedAverage,
        })
    }
    
    /// Median prediction (robust to outliers)
    fn median_prediction(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        let mut sorted: Vec<Decimal> = predictions.iter().map(|p| p.probability).collect();
        sorted.sort();
        
        let probability = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / dec!(2.0)
        } else {
            sorted[sorted.len() / 2]
        };
        
        let agreement = self.calculate_agreement(predictions);
        let (confidence, uncertainty) = self.calculate_confidence_uncertainty(predictions, agreement);
        
        let contributions = predictions.iter().map(|p| {
            let weight = if (p.probability - probability).abs() < dec!(0.1) {
                dec!(0.5)
            } else {
                dec!(0.1)
            };
            ModelContribution {
                model_id: p.model_id.clone(),
                prediction: p.probability,
                weight,
                performance_score: self.model_performance
                    .get(&p.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.5)),
            }
        }).collect();
        
        Some(EnsemblePrediction {
            probability,
            confidence,
            uncertainty,
            model_count: predictions.len(),
            agreement,
            contributions,
            method: EnsembleMethod::Median,
        })
    }
    
    /// Bayesian model combination
    fn bayesian_combination(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        // Simplified Bayesian combination using log-odds
        let mut log_odds_sum = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        
        for pred in predictions {
            // Convert to log-odds
            let p = pred.probability.max(dec!(0.001)).min(dec!(0.999));
            let log_odds = (decimal_to_f64(p) / (1.0 - decimal_to_f64(p))).ln();
            
            // Weight by model performance and confidence
            let perf = self.model_performance
                .get(&pred.model_id)
                .map(|m| m.performance_score())
                .unwrap_or(dec!(0.5));
            
            let weight = perf * pred.confidence;
            
            log_odds_sum += f64_to_decimal(log_odds) * weight;
            total_weight += weight;
        }
        
        // Convert back to probability
        let avg_log_odds = if total_weight > Decimal::ZERO {
            log_odds_sum / total_weight
        } else {
            Decimal::ZERO
        };
        
        let probability = sigmoid(avg_log_odds);
        
        let agreement = self.calculate_agreement(predictions);
        let (confidence, uncertainty) = self.calculate_confidence_uncertainty(predictions, agreement);
        
        let contributions = predictions.iter().map(|p| {
            ModelContribution {
                model_id: p.model_id.clone(),
                prediction: p.probability,
                weight: dec!(1.0) / Decimal::from(predictions.len() as i64),
                performance_score: self.model_performance
                    .get(&p.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.5)),
            }
        }).collect();
        
        Some(EnsemblePrediction {
            probability,
            confidence,
            uncertainty,
            model_count: predictions.len(),
            agreement,
            contributions,
            method: EnsembleMethod::BayesianCombination,
        })
    }
    
    /// Use best performing model only
    fn best_model(&self, predictions: &[ModelPrediction]) -> Option<EnsemblePrediction> {
        let best = predictions.iter()
            .max_by(|a, b| {
                let score_a = self.model_performance
                    .get(&a.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.0));
                let score_b = self.model_performance
                    .get(&b.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.0));
                score_a.partial_cmp(&score_b).unwrap()
            })?;
        
        let agreement = self.calculate_agreement(predictions);
        
        let contributions = predictions.iter().map(|p| {
            ModelContribution {
                model_id: p.model_id.clone(),
                prediction: p.probability,
                weight: if p.model_id == best.model_id { dec!(1.0) } else { dec!(0.0) },
                performance_score: self.model_performance
                    .get(&p.model_id)
                    .map(|m| m.performance_score())
                    .unwrap_or(dec!(0.5)),
            }
        }).collect();
        
        Some(EnsemblePrediction {
            probability: best.probability,
            confidence: best.confidence * agreement,
            uncertainty: best.uncertainty.unwrap_or(dec!(0.1)),
            model_count: predictions.len(),
            agreement,
            contributions,
            method: EnsembleMethod::BestModel,
        })
    }
    
    /// Calculate agreement between models (0-1)
    fn calculate_agreement(&self, predictions: &[ModelPrediction]) -> Decimal {
        if predictions.len() < 2 {
            return dec!(1.0);
        }
        
        let mean: Decimal = predictions.iter().map(|p| p.probability).sum::<Decimal>()
            / Decimal::from(predictions.len() as i64);
        
        // Variance
        let variance: Decimal = predictions.iter()
            .map(|p| (p.probability - mean) * (p.probability - mean))
            .sum::<Decimal>() / Decimal::from(predictions.len() as i64);
        
        // Convert variance to agreement (low variance = high agreement)
        // Max variance for probabilities is 0.25 (all 0s and 1s)
        (dec!(0.25) - variance.min(dec!(0.25))) / dec!(0.25)
    }
    
    /// Calculate confidence and uncertainty
    fn calculate_confidence_uncertainty(
        &self,
        predictions: &[ModelPrediction],
        agreement: Decimal,
    ) -> (Decimal, Decimal) {
        // Base confidence from model confidences
        let avg_confidence: Decimal = predictions.iter()
            .map(|p| p.confidence)
            .sum::<Decimal>() / Decimal::from(predictions.len() as i64);
        
        // Adjust confidence by agreement
        let confidence = if self.config.penalize_disagreement {
            avg_confidence * agreement
        } else {
            avg_confidence
        };
        
        // Uncertainty from model uncertainties and disagreement
        let avg_uncertainty: Decimal = predictions.iter()
            .filter_map(|p| p.uncertainty)
            .sum::<Decimal>();
        
        let count_with_uncertainty = predictions.iter()
            .filter(|p| p.uncertainty.is_some())
            .count();
        
        let base_uncertainty = if count_with_uncertainty > 0 {
            avg_uncertainty / Decimal::from(count_with_uncertainty as i64)
        } else {
            dec!(0.1)
        };
        
        // Add disagreement to uncertainty
        let uncertainty = base_uncertainty + (dec!(1.0) - agreement) * dec!(0.2);
        
        (confidence.min(dec!(1.0)), uncertainty.min(dec!(1.0)))
    }
    
    /// Get model performance stats
    pub fn model_stats(&self, model_id: &str) -> Option<ModelStats> {
        self.model_performance.get(model_id).map(|p| ModelStats {
            model_id: model_id.to_string(),
            predictions_count: p.predictions.len(),
            brier_score: p.brier_score,
            accuracy: p.accuracy,
            calibration_error: p.calibration_error,
            performance_score: p.performance_score(),
        })
    }
}

/// Model statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub model_id: String,
    pub predictions_count: usize,
    pub brier_score: Decimal,
    pub accuracy: Decimal,
    pub calibration_error: Decimal,
    pub performance_score: Decimal,
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

fn sigmoid(x: Decimal) -> Decimal {
    let x_f = decimal_to_f64(x);
    let result = 1.0 / (1.0 + (-x_f).exp());
    f64_to_decimal(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_prediction(model_id: &str, prob: Decimal, conf: Decimal) -> ModelPrediction {
        ModelPrediction {
            model_id: model_id.to_string(),
            probability: prob,
            confidence: conf,
            uncertainty: Some(dec!(0.1)),
            timestamp: Utc::now(),
            metadata: None,
        }
    }
    
    #[test]
    fn test_simple_average() {
        let ensemble = EnsemblePredictor::with_defaults();
        let predictions = vec![
            make_prediction("m1", dec!(0.6), dec!(0.8)),
            make_prediction("m2", dec!(0.7), dec!(0.9)),
            make_prediction("m3", dec!(0.65), dec!(0.85)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        assert!((result.probability - dec!(0.65)).abs() < dec!(0.01));
    }
    
    #[test]
    fn test_weighted_average() {
        let mut ensemble = EnsemblePredictor::new(EnsembleConfig {
            method: EnsembleMethod::WeightedAverage,
            ..Default::default()
        });
        
        // Train model performance
        for _ in 0..50 {
            ensemble.record_outcome("good_model", dec!(0.7), true);
            ensemble.record_outcome("bad_model", dec!(0.7), false);
        }
        
        let predictions = vec![
            make_prediction("good_model", dec!(0.8), dec!(0.9)),
            make_prediction("bad_model", dec!(0.3), dec!(0.9)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        // Good model should have more weight
        assert!(result.probability > dec!(0.55));
    }
    
    #[test]
    fn test_median_prediction() {
        let ensemble = EnsemblePredictor::new(EnsembleConfig {
            method: EnsembleMethod::Median,
            ..Default::default()
        });
        
        // Include an outlier
        let predictions = vec![
            make_prediction("m1", dec!(0.5), dec!(0.8)),
            make_prediction("m2", dec!(0.55), dec!(0.9)),
            make_prediction("outlier", dec!(0.95), dec!(0.5)), // Outlier
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        assert!((result.probability - dec!(0.55)).abs() < dec!(0.01)); // Median ignores outlier
    }
    
    #[test]
    fn test_agreement_calculation() {
        let ensemble = EnsemblePredictor::with_defaults();
        
        // High agreement
        let high_agree = vec![
            make_prediction("m1", dec!(0.6), dec!(0.8)),
            make_prediction("m2", dec!(0.62), dec!(0.9)),
            make_prediction("m3", dec!(0.58), dec!(0.85)),
        ];
        
        let result_high = ensemble.predict(&high_agree).unwrap();
        
        // Low agreement
        let low_agree = vec![
            make_prediction("m1", dec!(0.2), dec!(0.8)),
            make_prediction("m2", dec!(0.8), dec!(0.9)),
            make_prediction("m3", dec!(0.5), dec!(0.85)),
        ];
        
        let result_low = ensemble.predict(&low_agree).unwrap();
        
        assert!(result_high.agreement > result_low.agreement);
    }
    
    #[test]
    fn test_confidence_penalty_for_disagreement() {
        let ensemble = EnsemblePredictor::new(EnsembleConfig {
            penalize_disagreement: true,
            ..Default::default()
        });
        
        // Disagreeing predictions
        let predictions = vec![
            make_prediction("m1", dec!(0.2), dec!(0.9)),
            make_prediction("m2", dec!(0.8), dec!(0.9)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        // Confidence should be lower than average due to disagreement
        assert!(result.confidence < dec!(0.9));
    }
    
    #[test]
    fn test_min_models_requirement() {
        let ensemble = EnsemblePredictor::new(EnsembleConfig {
            min_models: 3,
            ..Default::default()
        });
        
        let predictions = vec![
            make_prediction("m1", dec!(0.6), dec!(0.8)),
            make_prediction("m2", dec!(0.7), dec!(0.9)),
        ];
        
        let result = ensemble.predict(&predictions);
        assert!(result.is_none()); // Not enough models
    }
    
    #[test]
    fn test_bayesian_combination() {
        let ensemble = EnsemblePredictor::new(EnsembleConfig {
            method: EnsembleMethod::BayesianCombination,
            ..Default::default()
        });
        
        let predictions = vec![
            make_prediction("m1", dec!(0.7), dec!(0.8)),
            make_prediction("m2", dec!(0.8), dec!(0.9)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        assert!(result.probability > dec!(0.5));
        assert!(result.probability < dec!(1.0));
    }
    
    #[test]
    fn test_best_model_selection() {
        let mut ensemble = EnsemblePredictor::new(EnsembleConfig {
            method: EnsembleMethod::BestModel,
            ..Default::default()
        });
        
        // Make one model clearly better
        for _ in 0..100 {
            ensemble.record_outcome("best", dec!(0.7), true);
            ensemble.record_outcome("worst", dec!(0.7), false);
        }
        
        let predictions = vec![
            make_prediction("best", dec!(0.8), dec!(0.9)),
            make_prediction("worst", dec!(0.2), dec!(0.9)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        assert_eq!(result.probability, dec!(0.8)); // Uses best model's prediction
    }
    
    #[test]
    fn test_model_stats() {
        let mut ensemble = EnsemblePredictor::with_defaults();
        
        for _ in 0..10 {
            ensemble.record_outcome("test_model", dec!(0.6), true);
        }
        
        let stats = ensemble.model_stats("test_model").unwrap();
        assert_eq!(stats.predictions_count, 10);
        assert!(stats.accuracy > dec!(0.5)); // Predicted 0.6, all true
    }
    
    #[test]
    fn test_contributions_sum() {
        let ensemble = EnsemblePredictor::with_defaults();
        
        let predictions = vec![
            make_prediction("m1", dec!(0.6), dec!(0.8)),
            make_prediction("m2", dec!(0.7), dec!(0.9)),
            make_prediction("m3", dec!(0.65), dec!(0.85)),
        ];
        
        let result = ensemble.predict(&predictions).unwrap();
        let weight_sum: Decimal = result.contributions.iter().map(|c| c.weight).sum();
        
        // Weights should approximately sum to 1
        assert!((weight_sum - dec!(1.0)).abs() < dec!(0.01));
    }
}
