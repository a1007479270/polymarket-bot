//! Probability calibration module
//!
//! Calibrates raw model outputs to well-calibrated probabilities:
//! - Platt scaling (sigmoid fitting)
//! - Isotonic regression
//! - Beta calibration
//! - Temperature scaling
//!
//! Well-calibrated probabilities are crucial for Kelly criterion sizing.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// Calibration method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Platt scaling: fits a sigmoid to map scores to probabilities
    PlattScaling,
    /// Isotonic regression: non-parametric monotonic calibration
    IsotonicRegression,
    /// Beta calibration: fits a beta distribution
    BetaCalibration,
    /// Temperature scaling: simple scaling factor
    TemperatureScaling,
    /// No calibration (pass-through)
    None,
}

/// Result of probability calibration
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Raw model output
    pub raw_probability: Decimal,
    /// Calibrated probability
    pub calibrated_probability: Decimal,
    /// Calibration method used
    pub method: CalibrationMethod,
    /// Confidence in calibration (based on training data)
    pub calibration_confidence: Decimal,
    /// Expected calibration error estimate
    pub ece_estimate: Decimal,
}

/// Calibration training sample
#[derive(Debug, Clone)]
struct CalibrationSample {
    predicted: Decimal,
    actual: bool, // Did the event occur?
}

/// Probability calibrator
pub struct ProbabilityCalibrator {
    method: CalibrationMethod,
    // Platt scaling parameters: P(y=1|x) = 1 / (1 + exp(A*x + B))
    platt_a: Decimal,
    platt_b: Decimal,
    // Temperature scaling
    temperature: Decimal,
    // Isotonic regression bins
    isotonic_bins: Vec<(Decimal, Decimal)>, // (threshold, calibrated_prob)
    // Training data for online learning
    training_samples: VecDeque<CalibrationSample>,
    max_samples: usize,
    // Calibration metrics
    calibration_error: Decimal,
    samples_seen: usize,
}

impl ProbabilityCalibrator {
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            platt_a: dec!(-1.0),
            platt_b: Decimal::ZERO,
            temperature: dec!(1.0),
            isotonic_bins: Self::default_isotonic_bins(),
            training_samples: VecDeque::new(),
            max_samples: 1000,
            calibration_error: dec!(0.1),
            samples_seen: 0,
        }
    }
    
    pub fn with_platt_scaling() -> Self {
        Self::new(CalibrationMethod::PlattScaling)
    }
    
    pub fn with_isotonic() -> Self {
        Self::new(CalibrationMethod::IsotonicRegression)
    }
    
    pub fn with_temperature(temp: Decimal) -> Self {
        let mut cal = Self::new(CalibrationMethod::TemperatureScaling);
        cal.temperature = temp;
        cal
    }
    
    /// Calibrate a raw probability
    pub fn calibrate(&self, raw_prob: Decimal) -> CalibrationResult {
        let calibrated = match self.method {
            CalibrationMethod::PlattScaling => self.platt_calibrate(raw_prob),
            CalibrationMethod::IsotonicRegression => self.isotonic_calibrate(raw_prob),
            CalibrationMethod::BetaCalibration => self.beta_calibrate(raw_prob),
            CalibrationMethod::TemperatureScaling => self.temperature_calibrate(raw_prob),
            CalibrationMethod::None => raw_prob,
        };
        
        // Clamp to valid probability range
        let calibrated = calibrated.max(dec!(0.001)).min(dec!(0.999));
        
        CalibrationResult {
            raw_probability: raw_prob,
            calibrated_probability: calibrated,
            method: self.method,
            calibration_confidence: self.calibration_confidence(),
            ece_estimate: self.calibration_error,
        }
    }
    
    /// Add a calibration sample (for online learning)
    pub fn add_sample(&mut self, predicted: Decimal, actual: bool) {
        self.training_samples.push_back(CalibrationSample { predicted, actual });
        if self.training_samples.len() > self.max_samples {
            self.training_samples.pop_front();
        }
        self.samples_seen += 1;
        
        // Re-fit periodically
        if self.samples_seen % 50 == 0 && self.training_samples.len() >= 30 {
            self.refit();
        }
    }
    
    /// Manually trigger refitting
    pub fn refit(&mut self) {
        match self.method {
            CalibrationMethod::PlattScaling => self.fit_platt(),
            CalibrationMethod::IsotonicRegression => self.fit_isotonic(),
            CalibrationMethod::TemperatureScaling => self.fit_temperature(),
            CalibrationMethod::BetaCalibration => self.fit_beta(),
            CalibrationMethod::None => {}
        }
        self.update_calibration_error();
    }
    
    /// Platt scaling calibration
    fn platt_calibrate(&self, raw: Decimal) -> Decimal {
        // Sigmoid: 1 / (1 + exp(A*x + B))
        let logit = self.platt_a * raw + self.platt_b;
        sigmoid(logit)
    }
    
    /// Isotonic regression calibration
    fn isotonic_calibrate(&self, raw: Decimal) -> Decimal {
        // Find the appropriate bin
        for (i, &(threshold, calibrated)) in self.isotonic_bins.iter().enumerate() {
            if raw <= threshold {
                if i == 0 {
                    return calibrated;
                }
                // Linear interpolation between bins
                let (prev_thresh, prev_cal) = self.isotonic_bins[i - 1];
                if threshold == prev_thresh {
                    return calibrated;
                }
                let ratio = (raw - prev_thresh) / (threshold - prev_thresh);
                return prev_cal + ratio * (calibrated - prev_cal);
            }
        }
        // Above all thresholds
        self.isotonic_bins.last().map(|&(_, c)| c).unwrap_or(raw)
    }
    
    /// Beta calibration (simplified)
    fn beta_calibrate(&self, raw: Decimal) -> Decimal {
        // Simplified beta calibration using power transform
        // Full implementation would fit a beta distribution
        let raw_f = decimal_to_f64(raw);
        let adjusted = raw_f.powf(1.0 / decimal_to_f64(self.temperature));
        f64_to_decimal(adjusted)
    }
    
    /// Temperature scaling calibration
    fn temperature_calibrate(&self, raw: Decimal) -> Decimal {
        if self.temperature == Decimal::ZERO {
            return raw;
        }
        
        // Convert probability to logit, scale, convert back
        let logit = prob_to_logit(raw);
        let scaled_logit = logit / self.temperature;
        sigmoid(scaled_logit)
    }
    
    /// Fit Platt scaling parameters
    fn fit_platt(&mut self) {
        if self.training_samples.len() < 10 {
            return;
        }
        
        // Simplified gradient descent for A and B
        let mut a = self.platt_a;
        let mut b = self.platt_b;
        let learning_rate = dec!(0.1);
        
        for _ in 0..100 {
            let mut grad_a = Decimal::ZERO;
            let mut grad_b = Decimal::ZERO;
            
            for sample in &self.training_samples {
                let pred = sigmoid(a * sample.predicted + b);
                let target = if sample.actual { dec!(1.0) } else { dec!(0.0) };
                let error = pred - target;
                
                grad_a += error * sample.predicted;
                grad_b += error;
            }
            
            let n = Decimal::from(self.training_samples.len() as i64);
            a -= learning_rate * grad_a / n;
            b -= learning_rate * grad_b / n;
        }
        
        self.platt_a = a;
        self.platt_b = b;
    }
    
    /// Fit isotonic regression
    fn fit_isotonic(&mut self) {
        if self.training_samples.len() < 10 {
            return;
        }
        
        // Sort samples by predicted probability
        let mut sorted: Vec<_> = self.training_samples.iter().cloned().collect();
        sorted.sort_by(|a, b| a.predicted.partial_cmp(&b.predicted).unwrap());
        
        // Pool Adjacent Violators Algorithm (PAVA)
        let n_bins = 10;
        let bin_size = sorted.len() / n_bins;
        
        self.isotonic_bins.clear();
        
        for i in 0..n_bins {
            let start = i * bin_size;
            let end = if i == n_bins - 1 { sorted.len() } else { (i + 1) * bin_size };
            
            if start >= end {
                continue;
            }
            
            let bin_samples = &sorted[start..end];
            let threshold = bin_samples.last().unwrap().predicted;
            let positive_count = bin_samples.iter().filter(|s| s.actual).count();
            let calibrated = Decimal::from(positive_count as i64) / Decimal::from(bin_samples.len() as i64);
            
            // Enforce monotonicity
            let calibrated = if let Some(&(_, prev_cal)) = self.isotonic_bins.last() {
                calibrated.max(prev_cal)
            } else {
                calibrated
            };
            
            self.isotonic_bins.push((threshold, calibrated));
        }
    }
    
    /// Fit temperature scaling
    fn fit_temperature(&mut self) {
        if self.training_samples.len() < 10 {
            return;
        }
        
        // Grid search for best temperature
        let mut best_temp = self.temperature;
        let mut best_loss = Decimal::MAX;
        
        for t in [dec!(0.5), dec!(0.75), dec!(1.0), dec!(1.25), dec!(1.5), dec!(2.0)] {
            let loss = self.calculate_nll(t);
            if loss < best_loss {
                best_loss = loss;
                best_temp = t;
            }
        }
        
        self.temperature = best_temp;
    }
    
    /// Fit beta calibration (simplified)
    fn fit_beta(&mut self) {
        // Reuse temperature fitting for simplified version
        self.fit_temperature();
    }
    
    /// Calculate negative log-likelihood for temperature
    fn calculate_nll(&self, temperature: Decimal) -> Decimal {
        let mut nll = Decimal::ZERO;
        
        for sample in &self.training_samples {
            let logit = prob_to_logit(sample.predicted);
            let scaled_logit = logit / temperature;
            let pred = sigmoid(scaled_logit);
            
            let pred = pred.max(dec!(0.0001)).min(dec!(0.9999));
            let target = if sample.actual { pred } else { dec!(1.0) - pred };
            
            // -log(p) approximation
            nll += dec!(1.0) - target;
        }
        
        nll / Decimal::from(self.training_samples.len() as i64)
    }
    
    /// Update calibration error estimate (Expected Calibration Error)
    fn update_calibration_error(&mut self) {
        if self.training_samples.len() < 10 {
            return;
        }
        
        let n_bins = 10;
        let mut bins: Vec<(Decimal, usize, usize)> = vec![(Decimal::ZERO, 0, 0); n_bins];
        
        for sample in &self.training_samples {
            let calibrated = self.calibrate(sample.predicted).calibrated_probability;
            let bin_idx = ((calibrated * Decimal::from(n_bins as i64)).floor())
                .min(Decimal::from((n_bins - 1) as i64));
            let bin_idx = decimal_to_usize(bin_idx);
            
            bins[bin_idx].0 += calibrated;
            bins[bin_idx].1 += if sample.actual { 1 } else { 0 };
            bins[bin_idx].2 += 1;
        }
        
        let mut ece = Decimal::ZERO;
        let total = self.training_samples.len();
        
        for (sum_conf, positives, count) in bins {
            if count > 0 {
                let avg_conf = sum_conf / Decimal::from(count as i64);
                let accuracy = Decimal::from(positives as i64) / Decimal::from(count as i64);
                let weight = Decimal::from(count as i64) / Decimal::from(total as i64);
                ece += weight * (avg_conf - accuracy).abs();
            }
        }
        
        self.calibration_error = ece;
    }
    
    /// Confidence in calibration based on sample count
    fn calibration_confidence(&self) -> Decimal {
        let samples = self.training_samples.len();
        if samples < 10 {
            dec!(0.1)
        } else if samples < 50 {
            dec!(0.3)
        } else if samples < 100 {
            dec!(0.5)
        } else if samples < 500 {
            dec!(0.7)
        } else {
            dec!(0.9)
        }
    }
    
    /// Default isotonic bins (before fitting)
    fn default_isotonic_bins() -> Vec<(Decimal, Decimal)> {
        vec![
            (dec!(0.1), dec!(0.1)),
            (dec!(0.2), dec!(0.2)),
            (dec!(0.3), dec!(0.3)),
            (dec!(0.4), dec!(0.4)),
            (dec!(0.5), dec!(0.5)),
            (dec!(0.6), dec!(0.6)),
            (dec!(0.7), dec!(0.7)),
            (dec!(0.8), dec!(0.8)),
            (dec!(0.9), dec!(0.9)),
            (dec!(1.0), dec!(1.0)),
        ]
    }
    
    /// Get current calibration stats
    pub fn stats(&self) -> CalibrationStats {
        CalibrationStats {
            method: self.method,
            samples_seen: self.samples_seen,
            calibration_error: self.calibration_error,
            platt_a: self.platt_a,
            platt_b: self.platt_b,
            temperature: self.temperature,
        }
    }
}

/// Calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub method: CalibrationMethod,
    pub samples_seen: usize,
    pub calibration_error: Decimal,
    pub platt_a: Decimal,
    pub platt_b: Decimal,
    pub temperature: Decimal,
}

/// Sigmoid function
fn sigmoid(x: Decimal) -> Decimal {
    // 1 / (1 + exp(-x))
    let x_f = decimal_to_f64(x);
    let result = 1.0 / (1.0 + (-x_f).exp());
    f64_to_decimal(result)
}

/// Convert probability to logit
fn prob_to_logit(p: Decimal) -> Decimal {
    let p = p.max(dec!(0.0001)).min(dec!(0.9999));
    let p_f = decimal_to_f64(p);
    let logit = (p_f / (1.0 - p_f)).ln();
    f64_to_decimal(logit)
}

fn decimal_to_f64(d: Decimal) -> f64 {
    use std::str::FromStr;
    f64::from_str(&d.to_string()).unwrap_or(0.0)
}

fn f64_to_decimal(f: f64) -> Decimal {
    use std::str::FromStr;
    if f.is_nan() || f.is_infinite() {
        return dec!(0.5);
    }
    Decimal::from_str(&format!("{:.6}", f)).unwrap_or(dec!(0.5))
}

fn decimal_to_usize(d: Decimal) -> usize {
    use std::str::FromStr;
    let s = d.to_string();
    usize::from_str(&s).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platt_calibration() {
        let cal = ProbabilityCalibrator::with_platt_scaling();
        let result = cal.calibrate(dec!(0.5));
        
        assert!(result.calibrated_probability > Decimal::ZERO);
        assert!(result.calibrated_probability < Decimal::ONE);
        assert_eq!(result.method, CalibrationMethod::PlattScaling);
    }
    
    #[test]
    fn test_isotonic_calibration() {
        let cal = ProbabilityCalibrator::with_isotonic();
        
        // Test that output is monotonic with input
        let low = cal.calibrate(dec!(0.2)).calibrated_probability;
        let mid = cal.calibrate(dec!(0.5)).calibrated_probability;
        let high = cal.calibrate(dec!(0.8)).calibrated_probability;
        
        assert!(low <= mid);
        assert!(mid <= high);
    }
    
    #[test]
    fn test_temperature_scaling() {
        // High temperature should move probabilities toward 0.5
        let hot = ProbabilityCalibrator::with_temperature(dec!(2.0));
        let cold = ProbabilityCalibrator::with_temperature(dec!(0.5));
        
        let hot_result = hot.calibrate(dec!(0.8)).calibrated_probability;
        let cold_result = cold.calibrate(dec!(0.8)).calibrated_probability;
        
        // Hot should be closer to 0.5
        assert!((hot_result - dec!(0.5)).abs() < (cold_result - dec!(0.5)).abs());
    }
    
    #[test]
    fn test_calibration_bounds() {
        let cal = ProbabilityCalibrator::with_platt_scaling();
        
        // Extreme inputs should still produce valid probabilities
        let low = cal.calibrate(dec!(0.001)).calibrated_probability;
        let high = cal.calibrate(dec!(0.999)).calibrated_probability;
        
        assert!(low >= dec!(0.001));
        assert!(high <= dec!(0.999));
    }
    
    #[test]
    fn test_online_learning() {
        let mut cal = ProbabilityCalibrator::with_platt_scaling();
        
        // Add samples where model is overconfident
        for _ in 0..50 {
            cal.add_sample(dec!(0.8), false); // Predicted 80%, actually false
            cal.add_sample(dec!(0.8), true);  // Predicted 80%, actually true
            cal.add_sample(dec!(0.2), false);
            cal.add_sample(dec!(0.2), true);
        }
        
        // After training, high predictions should be pulled down
        let before_a = dec!(-1.0);
        assert_ne!(cal.platt_a, before_a);
    }
    
    #[test]
    fn test_calibration_confidence() {
        let mut cal = ProbabilityCalibrator::with_platt_scaling();
        
        // Low confidence with few samples
        let initial_conf = cal.calibrate(dec!(0.5)).calibration_confidence;
        
        // Add samples
        for _ in 0..100 {
            cal.add_sample(dec!(0.5), true);
        }
        
        let trained_conf = cal.calibrate(dec!(0.5)).calibration_confidence;
        assert!(trained_conf > initial_conf);
    }
    
    #[test]
    fn test_no_calibration() {
        let cal = ProbabilityCalibrator::new(CalibrationMethod::None);
        let result = cal.calibrate(dec!(0.73));
        
        assert_eq!(result.calibrated_probability, dec!(0.73));
    }
    
    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(Decimal::ZERO) - dec!(0.5)).abs() < dec!(0.001));
        assert!(sigmoid(dec!(5.0)) > dec!(0.99));
        assert!(sigmoid(dec!(-5.0)) < dec!(0.01));
    }
    
    #[test]
    fn test_calibration_stats() {
        let mut cal = ProbabilityCalibrator::with_temperature(dec!(1.5));
        cal.add_sample(dec!(0.5), true);
        
        let stats = cal.stats();
        assert_eq!(stats.method, CalibrationMethod::TemperatureScaling);
        assert_eq!(stats.temperature, dec!(1.5));
        assert_eq!(stats.samples_seen, 1);
    }
    
    #[test]
    fn test_isotonic_monotonicity_enforcement() {
        let mut cal = ProbabilityCalibrator::with_isotonic();
        
        // Add samples that would create non-monotonic bins
        for i in 0..100 {
            let pred = Decimal::from(i) / dec!(100.0);
            // Intentionally make middle range less accurate
            let actual = if i < 30 || i > 70 { i % 2 == 0 } else { i % 3 == 0 };
            cal.add_sample(pred, actual);
        }
        
        cal.refit();
        
        // Verify monotonicity
        let mut prev = Decimal::ZERO;
        for p in [dec!(0.1), dec!(0.3), dec!(0.5), dec!(0.7), dec!(0.9)] {
            let result = cal.calibrate(p).calibrated_probability;
            assert!(result >= prev, "Isotonic should be monotonic");
            prev = result;
        }
    }
}
