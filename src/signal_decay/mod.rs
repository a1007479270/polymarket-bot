//! Signal Decay Analysis Module
//!
//! Analyzes how trading signal strength degrades over time, enabling optimal
//! execution timing and signal freshness validation for hourly crypto markets.
//!
//! # Key Features
//!
//! - **Decay Models**: Exponential, linear, power-law, and step decay functions
//! - **Half-Life Estimation**: Calculate signal half-life from historical data
//! - **Optimal Execution Window**: Find best time to act on a signal
//! - **Signal Freshness Score**: Real-time signal validity assessment
//! - **Decay-Adjusted Sizing**: Scale position size by signal strength
//! - **Multi-Signal Decay**: Track decay across multiple concurrent signals
//!
//! # Example
//!
//! ```ignore
//! use polymarket_bot::signal_decay::{SignalDecayAnalyzer, DecayModel};
//!
//! let mut analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
//!     half_life_minutes: 15.0,
//! });
//!
//! // Record signal and outcome
//! analyzer.record_signal(signal_strength, timestamp);
//! analyzer.record_outcome(pnl, execution_timestamp);
//!
//! // Get current signal strength
//! let current_strength = analyzer.get_decayed_strength(signal_timestamp);
//! ```

use chrono::{DateTime, Duration, Utc};
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Decay function type for signal strength degradation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecayModel {
    /// Exponential decay: S(t) = S0 * e^(-λt)
    /// Most common for financial signals
    Exponential {
        /// Time for signal to reach 50% strength (in minutes)
        half_life_minutes: f64,
    },
    
    /// Linear decay: S(t) = S0 * (1 - t/T)
    /// Signal dies completely at time T
    Linear {
        /// Total signal lifetime in minutes
        lifetime_minutes: f64,
    },
    
    /// Power-law decay: S(t) = S0 * (1 + t/τ)^(-α)
    /// Heavy tail - signal persists longer than exponential
    PowerLaw {
        /// Characteristic time scale
        tau_minutes: f64,
        /// Decay exponent (typically 0.5-2.0)
        alpha: f64,
    },
    
    /// Step decay: Signal maintains strength then drops
    /// Good for time-sensitive events
    Step {
        /// Time before first decay step
        step_minutes: f64,
        /// Decay factor at each step (0.5 = halve)
        step_decay: f64,
        /// Number of steps before signal dies
        num_steps: u32,
    },
    
    /// Adaptive decay: learns from historical performance
    Adaptive {
        /// Initial half-life estimate
        initial_half_life: f64,
        /// Learning rate for updates (0.01-0.1)
        learning_rate: f64,
    },
}

impl Default for DecayModel {
    fn default() -> Self {
        DecayModel::Exponential {
            half_life_minutes: 15.0,
        }
    }
}

/// A recorded signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    /// Unique signal identifier
    pub signal_id: String,
    /// Signal type/source (e.g., "RSI", "OBI", "ML")
    pub signal_type: String,
    /// Original signal strength [0, 1]
    pub initial_strength: Decimal,
    /// Direction: 1 for long, -1 for short
    pub direction: i8,
    /// Timestamp when signal was generated
    pub generated_at: DateTime<Utc>,
    /// Market/symbol this signal applies to
    pub symbol: String,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Outcome of acting on a signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalOutcome {
    /// Reference to the signal
    pub signal_id: String,
    /// When the trade was executed
    pub executed_at: DateTime<Utc>,
    /// Delay from signal generation to execution (minutes)
    pub execution_delay_minutes: f64,
    /// Realized P&L from the trade
    pub pnl: Decimal,
    /// Was the trade profitable?
    pub profitable: bool,
    /// Signal strength at execution time
    pub strength_at_execution: Decimal,
}

/// Statistics for a signal type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalTypeStats {
    /// Number of signals analyzed
    pub total_signals: u64,
    /// Number of profitable outcomes
    pub profitable_count: u64,
    /// Average execution delay for profitable trades
    pub avg_delay_profitable: f64,
    /// Average execution delay for losing trades
    pub avg_delay_losing: f64,
    /// Estimated optimal half-life
    pub estimated_half_life: f64,
    /// Win rate by delay bucket (0-5min, 5-15min, 15-30min, 30+min)
    pub win_rate_by_delay: Vec<(f64, f64, f64)>, // (min_delay, max_delay, win_rate)
    /// Average strength at profitable execution
    pub avg_strength_profitable: Decimal,
    /// Average strength at losing execution  
    pub avg_strength_losing: Decimal,
}

/// Main analyzer for signal decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDecayAnalyzer {
    /// The decay model in use
    model: DecayModel,
    /// Active signals awaiting outcome
    active_signals: HashMap<String, SignalRecord>,
    /// Historical signal outcomes
    outcomes: VecDeque<SignalOutcome>,
    /// Maximum outcomes to retain
    max_outcomes: usize,
    /// Statistics by signal type
    type_stats: HashMap<String, SignalTypeStats>,
    /// Adaptive half-life (for Adaptive model)
    adaptive_half_life: f64,
    /// Minimum signal strength threshold
    min_strength_threshold: Decimal,
}

impl SignalDecayAnalyzer {
    /// Create a new analyzer with specified decay model
    pub fn new(model: DecayModel) -> Self {
        let initial_half_life = match &model {
            DecayModel::Exponential { half_life_minutes } => *half_life_minutes,
            DecayModel::Adaptive { initial_half_life, .. } => *initial_half_life,
            _ => 15.0,
        };
        
        Self {
            model,
            active_signals: HashMap::new(),
            outcomes: VecDeque::with_capacity(1000),
            max_outcomes: 1000,
            type_stats: HashMap::new(),
            adaptive_half_life: initial_half_life,
            min_strength_threshold: dec!(0.1),
        }
    }
    
    /// Set minimum strength threshold
    pub fn with_min_threshold(mut self, threshold: Decimal) -> Self {
        self.min_strength_threshold = threshold;
        self
    }
    
    /// Record a new signal
    pub fn record_signal(&mut self, signal: SignalRecord) {
        self.active_signals.insert(signal.signal_id.clone(), signal);
    }
    
    /// Record the outcome of acting on a signal
    pub fn record_outcome(
        &mut self,
        signal_id: &str,
        executed_at: DateTime<Utc>,
        pnl: Decimal,
    ) -> Option<SignalOutcome> {
        let signal = self.active_signals.remove(signal_id)?;
        
        let delay_minutes = (executed_at - signal.generated_at).num_seconds() as f64 / 60.0;
        let strength_at_execution = self.calculate_decay(signal.initial_strength, delay_minutes);
        
        let outcome = SignalOutcome {
            signal_id: signal_id.to_string(),
            executed_at,
            execution_delay_minutes: delay_minutes,
            pnl,
            profitable: pnl > Decimal::ZERO,
            strength_at_execution,
        };
        
        // Update statistics
        self.update_stats(&signal.signal_type, &outcome);
        
        // Update adaptive model if applicable
        if let DecayModel::Adaptive { learning_rate, .. } = &self.model {
            self.update_adaptive_half_life(&outcome, *learning_rate);
        }
        
        // Store outcome
        self.outcomes.push_back(outcome.clone());
        while self.outcomes.len() > self.max_outcomes {
            self.outcomes.pop_front();
        }
        
        Some(outcome)
    }
    
    /// Calculate decayed signal strength
    pub fn calculate_decay(&self, initial_strength: Decimal, elapsed_minutes: f64) -> Decimal {
        if elapsed_minutes < 0.0 {
            return initial_strength;
        }
        
        let decay_factor = match &self.model {
            DecayModel::Exponential { half_life_minutes } => {
                let half_life = if matches!(self.model, DecayModel::Adaptive { .. }) {
                    self.adaptive_half_life
                } else {
                    *half_life_minutes
                };
                let lambda = 0.693147 / half_life; // ln(2) / half_life
                (-lambda * elapsed_minutes).exp()
            }
            
            DecayModel::Linear { lifetime_minutes } => {
                let ratio = elapsed_minutes / lifetime_minutes;
                (1.0 - ratio).max(0.0)
            }
            
            DecayModel::PowerLaw { tau_minutes, alpha } => {
                (1.0 + elapsed_minutes / tau_minutes).powf(-alpha)
            }
            
            DecayModel::Step { step_minutes, step_decay, num_steps } => {
                let steps_elapsed = (elapsed_minutes / step_minutes).floor() as u32;
                if steps_elapsed >= *num_steps {
                    0.0
                } else {
                    step_decay.powi(steps_elapsed as i32)
                }
            }
            
            DecayModel::Adaptive { .. } => {
                let lambda = 0.693147 / self.adaptive_half_life;
                (-lambda * elapsed_minutes).exp()
            }
        };
        
        let decayed = initial_strength * Decimal::from_f64(decay_factor).unwrap_or(Decimal::ZERO);
        decayed.max(Decimal::ZERO)
    }
    
    /// Get current strength of an active signal
    pub fn get_signal_strength(&self, signal_id: &str) -> Option<Decimal> {
        let signal = self.active_signals.get(signal_id)?;
        let elapsed = (Utc::now() - signal.generated_at).num_seconds() as f64 / 60.0;
        Some(self.calculate_decay(signal.initial_strength, elapsed))
    }
    
    /// Check if a signal is still valid (above threshold)
    pub fn is_signal_valid(&self, signal_id: &str) -> bool {
        self.get_signal_strength(signal_id)
            .map(|s| s >= self.min_strength_threshold)
            .unwrap_or(false)
    }
    
    /// Get optimal execution window for a signal type
    pub fn get_optimal_execution_window(&self, signal_type: &str) -> Option<(f64, f64)> {
        let stats = self.type_stats.get(signal_type)?;
        
        if stats.win_rate_by_delay.is_empty() {
            return None;
        }
        
        // Find bucket with highest win rate
        let best_bucket = stats.win_rate_by_delay
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;
        
        Some((best_bucket.0, best_bucket.1))
    }
    
    /// Calculate position size adjustment based on signal decay
    pub fn get_decay_adjusted_size(
        &self,
        base_size: Decimal,
        signal_strength: Decimal,
        elapsed_minutes: f64,
    ) -> Decimal {
        let decayed_strength = self.calculate_decay(signal_strength, elapsed_minutes);
        
        // Scale position by remaining signal strength
        // But don't go below 25% of base size to maintain market presence
        let min_scale = dec!(0.25);
        let scale = decayed_strength.max(min_scale);
        
        (base_size * scale).round_dp(2)
    }
    
    /// Estimate half-life from historical outcomes
    pub fn estimate_half_life(&self, signal_type: &str) -> Option<f64> {
        let outcomes: Vec<_> = self.outcomes
            .iter()
            .filter(|o| {
                self.active_signals.get(&o.signal_id)
                    .map(|s| s.signal_type == signal_type)
                    .unwrap_or(false)
                    || true // Include all if we can't find signal type
            })
            .collect();
        
        if outcomes.len() < 10 {
            return None;
        }
        
        // Group by delay buckets and calculate win rates
        let mut buckets: HashMap<i32, (u32, u32)> = HashMap::new(); // bucket -> (wins, total)
        
        for outcome in outcomes {
            let bucket = (outcome.execution_delay_minutes / 5.0).floor() as i32;
            let entry = buckets.entry(bucket).or_insert((0, 0));
            if outcome.profitable {
                entry.0 += 1;
            }
            entry.1 += 1;
        }
        
        // Find where win rate drops below 50%
        let mut sorted_buckets: Vec<_> = buckets.into_iter().collect();
        sorted_buckets.sort_by_key(|b| b.0);
        
        for (bucket, (wins, total)) in sorted_buckets {
            if total >= 5 {
                let win_rate = wins as f64 / total as f64;
                if win_rate < 0.5 {
                    return Some((bucket as f64 + 0.5) * 5.0);
                }
            }
        }
        
        None
    }
    
    /// Get statistics for a signal type
    pub fn get_type_stats(&self, signal_type: &str) -> Option<&SignalTypeStats> {
        self.type_stats.get(signal_type)
    }
    
    /// Get all active signals sorted by remaining strength
    pub fn get_active_signals_by_strength(&self) -> Vec<(String, Decimal)> {
        let mut signals: Vec<_> = self.active_signals
            .iter()
            .map(|(id, signal)| {
                let elapsed = (Utc::now() - signal.generated_at).num_seconds() as f64 / 60.0;
                let strength = self.calculate_decay(signal.initial_strength, elapsed);
                (id.clone(), strength)
            })
            .collect();
        
        signals.sort_by(|a, b| b.1.cmp(&a.1));
        signals
    }
    
    /// Clean up expired signals
    pub fn cleanup_expired_signals(&mut self, max_age_minutes: f64) {
        let now = Utc::now();
        self.active_signals.retain(|_, signal| {
            let age = (now - signal.generated_at).num_seconds() as f64 / 60.0;
            age <= max_age_minutes
        });
    }
    
    /// Update statistics for a signal type
    fn update_stats(&mut self, signal_type: &str, outcome: &SignalOutcome) {
        let stats = self.type_stats
            .entry(signal_type.to_string())
            .or_default();
        
        stats.total_signals += 1;
        
        if outcome.profitable {
            stats.profitable_count += 1;
            let n = stats.profitable_count as f64;
            stats.avg_delay_profitable = stats.avg_delay_profitable * (n - 1.0) / n 
                + outcome.execution_delay_minutes / n;
            stats.avg_strength_profitable = (stats.avg_strength_profitable 
                * Decimal::from(stats.profitable_count - 1) 
                + outcome.strength_at_execution) 
                / Decimal::from(stats.profitable_count);
        } else {
            let losing_count = stats.total_signals - stats.profitable_count;
            let n = losing_count as f64;
            stats.avg_delay_losing = stats.avg_delay_losing * (n - 1.0) / n 
                + outcome.execution_delay_minutes / n;
            stats.avg_strength_losing = (stats.avg_strength_losing 
                * Decimal::from(losing_count - 1) 
                + outcome.strength_at_execution) 
                / Decimal::from(losing_count);
        }
        
        // Update win rate by delay bucket (inline to avoid borrow issues)
        let buckets = [
            (0.0, 5.0),
            (5.0, 15.0),
            (15.0, 30.0),
            (30.0, f64::MAX),
        ];
        
        for (i, (min, max)) in buckets.iter().enumerate() {
            if outcome.execution_delay_minutes >= *min && outcome.execution_delay_minutes < *max {
                if stats.win_rate_by_delay.len() <= i {
                    stats.win_rate_by_delay.resize(i + 1, (*min, *max, 0.0));
                }
                
                let current = &stats.win_rate_by_delay[i];
                let new_rate = if outcome.profitable {
                    current.2 * 0.9 + 0.1 // EMA towards 1
                } else {
                    current.2 * 0.9 // EMA towards 0
                };
                stats.win_rate_by_delay[i] = (*min, *max, new_rate);
                break;
            }
        }
    }
    
    /// Update adaptive half-life based on outcome
    fn update_adaptive_half_life(&mut self, outcome: &SignalOutcome, learning_rate: f64) {
        // If profitable with high strength, half-life might be longer
        // If losing with low strength, our decay estimate was about right
        // If losing with high strength, half-life might be shorter than estimated
        
        let strength = outcome.strength_at_execution.to_f64().unwrap_or(0.5);
        
        if outcome.profitable && strength > 0.5 {
            // Signal still had value, maybe extend half-life slightly
            self.adaptive_half_life *= 1.0 + learning_rate;
        } else if !outcome.profitable && strength > 0.6 {
            // Lost money even with strong signal - shorten half-life
            self.adaptive_half_life *= 1.0 - learning_rate;
        }
        
        // Bound half-life to reasonable range (5 min to 60 min)
        self.adaptive_half_life = self.adaptive_half_life.clamp(5.0, 60.0);
    }
    
    /// Get the current adaptive half-life
    pub fn get_adaptive_half_life(&self) -> f64 {
        self.adaptive_half_life
    }
    
    /// Get summary of analyzer state
    pub fn get_summary(&self) -> DecayAnalyzerSummary {
        let total_outcomes = self.outcomes.len();
        let profitable_count = self.outcomes.iter().filter(|o| o.profitable).count();
        
        let avg_profitable_delay = if profitable_count > 0 {
            self.outcomes.iter()
                .filter(|o| o.profitable)
                .map(|o| o.execution_delay_minutes)
                .sum::<f64>() / profitable_count as f64
        } else {
            0.0
        };
        
        DecayAnalyzerSummary {
            active_signals: self.active_signals.len(),
            total_outcomes,
            profitable_count,
            current_half_life: self.adaptive_half_life,
            avg_profitable_delay,
            min_strength_threshold: self.min_strength_threshold,
            signal_types: self.type_stats.keys().cloned().collect(),
        }
    }
}

/// Summary of decay analyzer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayAnalyzerSummary {
    pub active_signals: usize,
    pub total_outcomes: usize,
    pub profitable_count: usize,
    pub current_half_life: f64,
    pub avg_profitable_delay: f64,
    pub min_strength_threshold: Decimal,
    pub signal_types: Vec<String>,
}

/// Multi-signal decay manager for tracking concurrent signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSignalDecayManager {
    /// Individual analyzers per signal type
    analyzers: HashMap<String, SignalDecayAnalyzer>,
    /// Global signal correlation tracking
    correlation_window: Duration,
    /// Recent signal timestamps for correlation
    recent_signals: VecDeque<(DateTime<Utc>, String, Decimal)>,
}

impl MultiSignalDecayManager {
    /// Create new multi-signal manager
    pub fn new() -> Self {
        Self {
            analyzers: HashMap::new(),
            correlation_window: Duration::minutes(30),
            recent_signals: VecDeque::with_capacity(100),
        }
    }
    
    /// Register a signal type with its decay model
    pub fn register_signal_type(&mut self, signal_type: &str, model: DecayModel) {
        self.analyzers.insert(
            signal_type.to_string(),
            SignalDecayAnalyzer::new(model),
        );
    }
    
    /// Record a signal
    pub fn record_signal(&mut self, signal: SignalRecord) {
        let signal_type = signal.signal_type.clone();
        
        // Track for correlation
        self.recent_signals.push_back((
            signal.generated_at,
            signal_type.clone(),
            signal.initial_strength,
        ));
        
        // Trim old signals
        let cutoff = Utc::now() - self.correlation_window;
        while self.recent_signals.front().map(|s| s.0 < cutoff).unwrap_or(false) {
            self.recent_signals.pop_front();
        }
        
        // Record in type-specific analyzer
        if let Some(analyzer) = self.analyzers.get_mut(&signal_type) {
            analyzer.record_signal(signal);
        }
    }
    
    /// Get combined signal strength from all concurrent signals
    pub fn get_combined_strength(&self, symbol: &str) -> Decimal {
        let mut total_strength = Decimal::ZERO;
        let mut weight_sum = Decimal::ZERO;
        
        for (_signal_type, analyzer) in &self.analyzers {
            for (id, signal) in &analyzer.active_signals {
                if signal.symbol == symbol {
                    let strength = analyzer.get_signal_strength(id).unwrap_or(Decimal::ZERO);
                    // Weight by signal type reliability (could be configurable)
                    let weight = dec!(1.0);
                    total_strength += strength * weight;
                    weight_sum += weight;
                }
            }
        }
        
        if weight_sum > Decimal::ZERO {
            total_strength / weight_sum
        } else {
            Decimal::ZERO
        }
    }
    
    /// Check signal agreement across types
    pub fn get_signal_consensus(&self, symbol: &str) -> SignalConsensus {
        let mut long_signals = 0;
        let mut short_signals = 0;
        let mut total_strength = Decimal::ZERO;
        
        for analyzer in self.analyzers.values() {
            for signal in analyzer.active_signals.values() {
                if signal.symbol == symbol {
                    let elapsed = (Utc::now() - signal.generated_at).num_seconds() as f64 / 60.0;
                    let strength = analyzer.calculate_decay(signal.initial_strength, elapsed);
                    
                    if strength >= analyzer.min_strength_threshold {
                        if signal.direction > 0 {
                            long_signals += 1;
                        } else {
                            short_signals += 1;
                        }
                        total_strength += strength;
                    }
                }
            }
        }
        
        let total = long_signals + short_signals;
        let (direction, agreement) = if total == 0 {
            (0, dec!(0.0))
        } else if long_signals > short_signals {
            (1, Decimal::from(long_signals) / Decimal::from(total))
        } else if short_signals > long_signals {
            (-1, Decimal::from(short_signals) / Decimal::from(total))
        } else {
            (0, dec!(0.5))
        };
        
        SignalConsensus {
            direction,
            agreement_ratio: agreement,
            long_signals,
            short_signals,
            average_strength: if total > 0 {
                total_strength / Decimal::from(total)
            } else {
                Decimal::ZERO
            },
        }
    }
}

impl Default for MultiSignalDecayManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Consensus result from multiple signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConsensus {
    /// Net direction: 1 long, -1 short, 0 neutral
    pub direction: i8,
    /// Ratio of signals agreeing (0-1)
    pub agreement_ratio: Decimal,
    /// Count of long signals
    pub long_signals: usize,
    /// Count of short signals
    pub short_signals: usize,
    /// Average strength of active signals
    pub average_strength: Decimal,
}

/// Time-weighted signal for gradual execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWeightedSignal {
    /// Original signal
    pub signal: SignalRecord,
    /// Planned execution slices
    pub slices: Vec<ExecutionSlice>,
    /// Completed slices
    pub completed: usize,
}

/// Single execution slice in a time-weighted strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSlice {
    /// Scheduled execution time
    pub scheduled_at: DateTime<Utc>,
    /// Portion of total size (0-1)
    pub size_fraction: Decimal,
    /// Expected signal strength at execution
    pub expected_strength: Decimal,
    /// Whether this slice has been executed
    pub executed: bool,
}

impl TimeWeightedSignal {
    /// Create execution slices accounting for signal decay
    pub fn create_decay_aware_slices(
        signal: SignalRecord,
        total_duration_minutes: f64,
        num_slices: usize,
        analyzer: &SignalDecayAnalyzer,
    ) -> Self {
        let mut slices = Vec::with_capacity(num_slices);
        let slice_interval = total_duration_minutes / num_slices as f64;
        
        let mut strength_sum = 0.0;
        let strengths: Vec<f64> = (0..num_slices)
            .map(|i| {
                let delay = slice_interval * (i as f64 + 0.5);
                let strength = analyzer
                    .calculate_decay(signal.initial_strength, delay)
                    .to_f64()
                    .unwrap_or(0.0);
                strength_sum += strength;
                strength
            })
            .collect();
        
        for (i, strength) in strengths.iter().enumerate() {
            let scheduled_at = signal.generated_at 
                + Duration::seconds((slice_interval * (i as f64 + 0.5) * 60.0) as i64);
            
            // Weight slice size by expected strength (front-load execution)
            let size_fraction = if strength_sum > 0.0 {
                Decimal::from_f64(strength / strength_sum).unwrap_or(dec!(0.0))
            } else {
                Decimal::from(1) / Decimal::from(num_slices)
            };
            
            slices.push(ExecutionSlice {
                scheduled_at,
                size_fraction,
                expected_strength: Decimal::from_f64(*strength).unwrap_or(Decimal::ZERO),
                executed: false,
            });
        }
        
        Self {
            signal,
            slices,
            completed: 0,
        }
    }
    
    /// Get next pending slice
    pub fn next_slice(&self) -> Option<&ExecutionSlice> {
        self.slices.get(self.completed)
    }
    
    /// Mark current slice as completed
    pub fn complete_slice(&mut self) {
        if self.completed < self.slices.len() {
            self.slices[self.completed].executed = true;
            self.completed += 1;
        }
    }
    
    /// Check if all slices completed
    pub fn is_complete(&self) -> bool {
        self.completed >= self.slices.len()
    }
    
    /// Get remaining size fraction
    pub fn remaining_fraction(&self) -> Decimal {
        self.slices[self.completed..]
            .iter()
            .map(|s| s.size_fraction)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_signal(id: &str, strength: Decimal, direction: i8) -> SignalRecord {
        SignalRecord {
            signal_id: id.to_string(),
            signal_type: "RSI".to_string(),
            initial_strength: strength,
            direction,
            generated_at: Utc::now(),
            symbol: "BTC".to_string(),
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_exponential_decay() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        let strength = analyzer.calculate_decay(dec!(1.0), 15.0);
        // After one half-life, should be ~0.5
        assert!(strength > dec!(0.49) && strength < dec!(0.51));
        
        let strength = analyzer.calculate_decay(dec!(1.0), 30.0);
        // After two half-lives, should be ~0.25
        assert!(strength > dec!(0.24) && strength < dec!(0.26));
    }
    
    #[test]
    fn test_linear_decay() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Linear {
            lifetime_minutes: 60.0,
        });
        
        let strength = analyzer.calculate_decay(dec!(1.0), 30.0);
        // At 50% of lifetime, should be 0.5
        assert_eq!(strength, dec!(0.5));
        
        let strength = analyzer.calculate_decay(dec!(1.0), 60.0);
        // At end of lifetime, should be 0
        assert_eq!(strength, Decimal::ZERO);
    }
    
    #[test]
    fn test_power_law_decay() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::PowerLaw {
            tau_minutes: 10.0,
            alpha: 1.0,
        });
        
        let strength = analyzer.calculate_decay(dec!(1.0), 10.0);
        // (1 + 10/10)^(-1) = 0.5
        assert!(strength > dec!(0.49) && strength < dec!(0.51));
    }
    
    #[test]
    fn test_step_decay() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Step {
            step_minutes: 10.0,
            step_decay: 0.5,
            num_steps: 3,
        });
        
        // Before first step
        let strength = analyzer.calculate_decay(dec!(1.0), 5.0);
        assert_eq!(strength, dec!(1.0));
        
        // After first step
        let strength = analyzer.calculate_decay(dec!(1.0), 15.0);
        assert_eq!(strength, dec!(0.5));
        
        // After second step
        let strength = analyzer.calculate_decay(dec!(1.0), 25.0);
        assert_eq!(strength, dec!(0.25));
        
        // After all steps (dead)
        let strength = analyzer.calculate_decay(dec!(1.0), 35.0);
        assert_eq!(strength, Decimal::ZERO);
    }
    
    #[test]
    fn test_signal_record_and_validity() {
        let mut analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        analyzer.min_strength_threshold = dec!(0.3);
        
        let signal = make_signal("test1", dec!(0.8), 1);
        analyzer.record_signal(signal);
        
        // Signal should be valid immediately
        assert!(analyzer.is_signal_valid("test1"));
        
        // Check strength retrieval
        let strength = analyzer.get_signal_strength("test1").unwrap();
        assert!(strength > dec!(0.79)); // Should be close to 0.8
    }
    
    #[test]
    fn test_outcome_recording() {
        let mut analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        let signal = SignalRecord {
            signal_id: "test1".to_string(),
            signal_type: "RSI".to_string(),
            initial_strength: dec!(0.8),
            direction: 1,
            generated_at: Utc::now() - Duration::minutes(10),
            symbol: "BTC".to_string(),
            metadata: HashMap::new(),
        };
        analyzer.record_signal(signal);
        
        let outcome = analyzer.record_outcome("test1", Utc::now(), dec!(50.0));
        assert!(outcome.is_some());
        
        let outcome = outcome.unwrap();
        assert!(outcome.profitable);
        assert!(outcome.execution_delay_minutes > 9.0 && outcome.execution_delay_minutes < 11.0);
    }
    
    #[test]
    fn test_decay_adjusted_sizing() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        let base_size = dec!(100.0);
        
        // Fresh signal - full size
        let size = analyzer.get_decay_adjusted_size(base_size, dec!(1.0), 0.0);
        assert_eq!(size, dec!(100.0));
        
        // Half-decayed signal - ~50% size
        let size = analyzer.get_decay_adjusted_size(base_size, dec!(1.0), 15.0);
        assert!(size > dec!(49.0) && size < dec!(51.0));
        
        // Very decayed signal - minimum 25%
        let size = analyzer.get_decay_adjusted_size(base_size, dec!(1.0), 60.0);
        assert!(size >= dec!(25.0));
    }
    
    #[test]
    fn test_multi_signal_consensus() {
        let mut manager = MultiSignalDecayManager::new();
        
        manager.register_signal_type("RSI", DecayModel::Exponential { half_life_minutes: 15.0 });
        manager.register_signal_type("OBI", DecayModel::Exponential { half_life_minutes: 10.0 });
        
        // Add agreeing long signals
        manager.record_signal(make_signal("rsi1", dec!(0.8), 1));
        manager.record_signal(SignalRecord {
            signal_id: "obi1".to_string(),
            signal_type: "OBI".to_string(),
            initial_strength: dec!(0.7),
            direction: 1,
            generated_at: Utc::now(),
            symbol: "BTC".to_string(),
            metadata: HashMap::new(),
        });
        
        let consensus = manager.get_signal_consensus("BTC");
        assert_eq!(consensus.direction, 1);
        assert_eq!(consensus.long_signals, 2);
        assert_eq!(consensus.short_signals, 0);
    }
    
    #[test]
    fn test_time_weighted_slices() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        let signal = make_signal("test1", dec!(1.0), 1);
        let tws = TimeWeightedSignal::create_decay_aware_slices(
            signal,
            30.0, // 30 minute duration
            3,     // 3 slices
            &analyzer,
        );
        
        assert_eq!(tws.slices.len(), 3);
        
        // First slice should have larger fraction (front-loaded due to decay)
        assert!(tws.slices[0].size_fraction > tws.slices[2].size_fraction);
        
        // Total fractions should sum to ~1
        let total: Decimal = tws.slices.iter().map(|s| s.size_fraction).sum();
        assert!(total > dec!(0.99) && total < dec!(1.01));
    }
    
    #[test]
    fn test_cleanup_expired() {
        let mut analyzer = SignalDecayAnalyzer::new(DecayModel::default());
        
        // Add old signal
        let old_signal = SignalRecord {
            signal_id: "old".to_string(),
            signal_type: "RSI".to_string(),
            initial_strength: dec!(0.8),
            direction: 1,
            generated_at: Utc::now() - Duration::hours(2),
            symbol: "BTC".to_string(),
            metadata: HashMap::new(),
        };
        analyzer.record_signal(old_signal);
        
        // Add fresh signal
        let fresh_signal = make_signal("fresh", dec!(0.8), 1);
        analyzer.record_signal(fresh_signal);
        
        assert_eq!(analyzer.active_signals.len(), 2);
        
        // Cleanup signals older than 60 minutes
        analyzer.cleanup_expired_signals(60.0);
        
        assert_eq!(analyzer.active_signals.len(), 1);
        assert!(analyzer.active_signals.contains_key("fresh"));
    }
    
    #[test]
    fn test_get_summary() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        let summary = analyzer.get_summary();
        assert_eq!(summary.active_signals, 0);
        assert_eq!(summary.total_outcomes, 0);
        assert_eq!(summary.current_half_life, 15.0);
    }
    
    #[test]
    fn test_active_signals_by_strength() {
        let mut analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        // Add signals with different strengths
        let strong = make_signal("strong", dec!(0.9), 1);
        let weak = make_signal("weak", dec!(0.3), 1);
        let medium = make_signal("medium", dec!(0.6), 1);
        
        analyzer.record_signal(strong);
        analyzer.record_signal(weak);
        analyzer.record_signal(medium);
        
        let sorted = analyzer.get_active_signals_by_strength();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, "strong");
        assert_eq!(sorted[1].0, "medium");
        assert_eq!(sorted[2].0, "weak");
    }
    
    #[test]
    fn test_negative_elapsed_time() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::Exponential {
            half_life_minutes: 15.0,
        });
        
        // Negative elapsed time should return initial strength
        let strength = analyzer.calculate_decay(dec!(0.8), -5.0);
        assert_eq!(strength, dec!(0.8));
    }
    
    #[test]
    fn test_adaptive_model_bounds() {
        let mut analyzer = SignalDecayAnalyzer::new(DecayModel::Adaptive {
            initial_half_life: 15.0,
            learning_rate: 0.5, // High learning rate for testing
        });
        
        // Record signal
        let signal = SignalRecord {
            signal_id: "test1".to_string(),
            signal_type: "RSI".to_string(),
            initial_strength: dec!(0.9),
            direction: 1,
            generated_at: Utc::now() - Duration::minutes(1),
            symbol: "BTC".to_string(),
            metadata: HashMap::new(),
        };
        analyzer.record_signal(signal);
        
        // Record losing outcome with high strength (should shorten half-life)
        analyzer.record_outcome("test1", Utc::now(), dec!(-50.0));
        
        // Half-life should decrease but stay within bounds
        assert!(analyzer.adaptive_half_life >= 5.0);
        assert!(analyzer.adaptive_half_life <= 60.0);
    }
    
    #[test]
    fn test_combined_strength() {
        let mut manager = MultiSignalDecayManager::new();
        
        manager.register_signal_type("RSI", DecayModel::Exponential { half_life_minutes: 15.0 });
        
        // Add signal
        manager.record_signal(make_signal("rsi1", dec!(0.8), 1));
        
        let strength = manager.get_combined_strength("BTC");
        assert!(strength > dec!(0.79));
        
        // Different symbol should return zero
        let strength = manager.get_combined_strength("ETH");
        assert_eq!(strength, Decimal::ZERO);
    }
    
    #[test]
    fn test_slice_completion() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::default());
        let signal = make_signal("test", dec!(1.0), 1);
        
        let mut tws = TimeWeightedSignal::create_decay_aware_slices(
            signal,
            30.0,
            3,
            &analyzer,
        );
        
        assert!(!tws.is_complete());
        assert_eq!(tws.completed, 0);
        
        tws.complete_slice();
        assert_eq!(tws.completed, 1);
        assert!(!tws.is_complete());
        
        tws.complete_slice();
        tws.complete_slice();
        assert!(tws.is_complete());
    }
    
    #[test]
    fn test_remaining_fraction() {
        let analyzer = SignalDecayAnalyzer::new(DecayModel::default());
        let signal = make_signal("test", dec!(1.0), 1);
        
        let mut tws = TimeWeightedSignal::create_decay_aware_slices(
            signal,
            30.0,
            3,
            &analyzer,
        );
        
        let initial = tws.remaining_fraction();
        assert!(initial > dec!(0.99));
        
        tws.complete_slice();
        let remaining = tws.remaining_fraction();
        assert!(remaining < initial);
    }
}
