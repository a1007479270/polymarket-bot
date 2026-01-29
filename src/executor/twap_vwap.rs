//! TWAP/VWAP Execution Algorithms
//!
//! Professional order execution strategies for minimizing market impact:
//! - TWAP (Time-Weighted Average Price): Spreads orders evenly over time
//! - VWAP (Volume-Weighted Average Price): Weights orders by historical volume profile
//! - Adaptive execution with real-time adjustment based on market conditions

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Execution algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    /// Time-weighted average price - even distribution over time
    TWAP,
    /// Volume-weighted average price - follows historical volume curve
    VWAP,
    /// Adaptive - switches between algorithms based on conditions
    Adaptive,
    /// Immediate - execute all at once (for small orders)
    Immediate,
}

/// Volume profile bucket for VWAP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBucket {
    /// Start minute of the hour (0-59)
    pub minute: u8,
    /// Relative volume weight (0.0 - 1.0)
    pub weight: Decimal,
    /// Historical average volume in this bucket
    pub avg_volume: Decimal,
}

/// Configuration for execution algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Total execution duration
    pub duration: Duration,
    /// Number of slices to divide the order into
    pub num_slices: u32,
    /// Minimum slice size (won't split below this)
    pub min_slice_size: Decimal,
    /// Maximum deviation from target price before pausing
    pub max_price_deviation: Decimal,
    /// Whether to use limit orders (vs market)
    pub use_limit_orders: bool,
    /// Limit order offset from mid price
    pub limit_offset: Decimal,
    /// Randomize timing within slice windows
    pub randomize_timing: bool,
    /// Participation rate cap (% of market volume)
    pub max_participation_rate: Decimal,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(600), // 10 minutes default
            num_slices: 10,
            min_slice_size: dec!(1.0),
            max_price_deviation: dec!(0.02), // 2%
            use_limit_orders: true,
            limit_offset: dec!(0.001), // 0.1%
            randomize_timing: true,
            max_participation_rate: dec!(0.10), // 10% of volume
        }
    }
}

/// State of a single execution slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSlice {
    /// Slice index (0-based)
    pub index: u32,
    /// Target quantity for this slice
    pub target_quantity: Decimal,
    /// Actual filled quantity
    pub filled_quantity: Decimal,
    /// Scheduled execution time (seconds from start)
    pub scheduled_time: f64,
    /// Actual execution time
    pub executed_time: Option<f64>,
    /// Execution price achieved
    pub execution_price: Option<Decimal>,
    /// Status of this slice
    pub status: SliceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceStatus {
    Pending,
    Executing,
    Filled,
    PartialFill,
    Cancelled,
    Failed,
}

/// TWAP/VWAP execution engine
#[derive(Debug, Clone)]
pub struct ExecutionEngine {
    /// Algorithm type
    pub algorithm: ExecutionAlgorithm,
    /// Configuration
    pub config: ExecutionConfig,
    /// Total order quantity
    pub total_quantity: Decimal,
    /// Order side (true = buy, false = sell)
    pub is_buy: bool,
    /// Target price (benchmark)
    pub target_price: Decimal,
    /// Execution slices
    pub slices: Vec<ExecutionSlice>,
    /// Volume profile for VWAP (24 buckets for 24 hours)
    pub volume_profile: Option<Vec<VolumeBucket>>,
    /// Start time
    pub start_time: Option<Instant>,
    /// Current market price
    pub current_price: Decimal,
    /// Total filled quantity
    pub filled_quantity: Decimal,
    /// Volume-weighted average execution price
    pub vwap_achieved: Decimal,
    /// Recent market volumes for participation rate
    pub recent_volumes: VecDeque<Decimal>,
    /// Whether execution is paused
    pub is_paused: bool,
    /// Pause reason
    pub pause_reason: Option<String>,
}

impl ExecutionEngine {
    /// Create a new TWAP execution engine
    pub fn new_twap(
        total_quantity: Decimal,
        is_buy: bool,
        target_price: Decimal,
        config: ExecutionConfig,
    ) -> Self {
        let mut engine = Self {
            algorithm: ExecutionAlgorithm::TWAP,
            config: config.clone(),
            total_quantity,
            is_buy,
            target_price,
            slices: Vec::new(),
            volume_profile: None,
            start_time: None,
            current_price: target_price,
            filled_quantity: Decimal::ZERO,
            vwap_achieved: Decimal::ZERO,
            recent_volumes: VecDeque::with_capacity(60),
            is_paused: false,
            pause_reason: None,
        };
        engine.generate_twap_slices();
        engine
    }

    /// Create a new VWAP execution engine
    pub fn new_vwap(
        total_quantity: Decimal,
        is_buy: bool,
        target_price: Decimal,
        config: ExecutionConfig,
        volume_profile: Vec<VolumeBucket>,
    ) -> Self {
        let mut engine = Self {
            algorithm: ExecutionAlgorithm::VWAP,
            config: config.clone(),
            total_quantity,
            is_buy,
            target_price,
            slices: Vec::new(),
            volume_profile: Some(volume_profile),
            start_time: None,
            current_price: target_price,
            filled_quantity: Decimal::ZERO,
            vwap_achieved: Decimal::ZERO,
            recent_volumes: VecDeque::with_capacity(60),
            is_paused: false,
            pause_reason: None,
        };
        engine.generate_vwap_slices();
        engine
    }

    /// Create an adaptive execution engine
    pub fn new_adaptive(
        total_quantity: Decimal,
        is_buy: bool,
        target_price: Decimal,
        config: ExecutionConfig,
        volume_profile: Option<Vec<VolumeBucket>>,
    ) -> Self {
        // Start with TWAP, switch to VWAP if volume profile available and beneficial
        let algorithm = if volume_profile.is_some() {
            ExecutionAlgorithm::Adaptive
        } else {
            ExecutionAlgorithm::TWAP
        };

        let mut engine = Self {
            algorithm,
            config: config.clone(),
            total_quantity,
            is_buy,
            target_price,
            slices: Vec::new(),
            volume_profile,
            start_time: None,
            current_price: target_price,
            filled_quantity: Decimal::ZERO,
            vwap_achieved: Decimal::ZERO,
            recent_volumes: VecDeque::with_capacity(60),
            is_paused: false,
            pause_reason: None,
        };

        if engine.volume_profile.is_some() {
            engine.generate_vwap_slices();
        } else {
            engine.generate_twap_slices();
        }

        engine
    }

    /// Generate TWAP slices (even distribution)
    fn generate_twap_slices(&mut self) {
        let duration_secs = self.config.duration.as_secs_f64();
        let slice_interval = duration_secs / self.config.num_slices as f64;

        // Calculate base quantity per slice
        let base_quantity = self.total_quantity / Decimal::from(self.config.num_slices);

        // Ensure minimum slice size
        let actual_slices = if base_quantity < self.config.min_slice_size {
            (self.total_quantity / self.config.min_slice_size)
                .floor()
                .to_string()
                .parse::<u32>()
                .unwrap_or(1)
                .max(1)
        } else {
            self.config.num_slices
        };

        let quantity_per_slice = self.total_quantity / Decimal::from(actual_slices);
        let actual_interval = duration_secs / actual_slices as f64;

        self.slices.clear();
        let mut remaining = self.total_quantity;

        for i in 0..actual_slices {
            let quantity = if i == actual_slices - 1 {
                remaining // Last slice gets remainder
            } else {
                quantity_per_slice.min(remaining)
            };

            remaining -= quantity;

            let scheduled_time = if self.config.randomize_timing {
                // Add small random offset within window
                let base_time = i as f64 * actual_interval;
                let jitter = actual_interval * 0.2 * (i as f64 % 3.0 - 1.0) / 3.0;
                (base_time + jitter).max(0.0)
            } else {
                i as f64 * actual_interval
            };

            self.slices.push(ExecutionSlice {
                index: i,
                target_quantity: quantity,
                filled_quantity: Decimal::ZERO,
                scheduled_time,
                executed_time: None,
                execution_price: None,
                status: SliceStatus::Pending,
            });
        }
    }

    /// Generate VWAP slices (volume-weighted distribution)
    fn generate_vwap_slices(&mut self) {
        let duration_secs = self.config.duration.as_secs_f64();
        let slice_interval = duration_secs / self.config.num_slices as f64;

        // Get volume profile or use uniform
        let profile = self.volume_profile.clone().unwrap_or_else(|| {
            (0..24)
                .map(|h| VolumeBucket {
                    minute: (h * 60 / 24) as u8,
                    weight: dec!(1.0) / dec!(24),
                    avg_volume: dec!(1000),
                })
                .collect()
        });

        // Calculate total weight for normalization
        let total_weight: Decimal = profile.iter().map(|b| b.weight).sum();

        self.slices.clear();
        let mut remaining = self.total_quantity;

        for i in 0..self.config.num_slices {
            // Map slice index to volume bucket
            let bucket_index = (i as usize * profile.len() / self.config.num_slices as usize)
                .min(profile.len() - 1);
            let bucket = &profile[bucket_index];

            // Weight quantity by volume profile
            let weight_factor = if total_weight > Decimal::ZERO {
                bucket.weight / total_weight * Decimal::from(profile.len())
            } else {
                Decimal::ONE
            };

            let base_quantity = self.total_quantity / Decimal::from(self.config.num_slices);
            let quantity = (base_quantity * weight_factor)
                .max(self.config.min_slice_size)
                .min(remaining);

            if quantity <= Decimal::ZERO {
                continue;
            }

            remaining = (remaining - quantity).max(Decimal::ZERO);

            let scheduled_time = i as f64 * slice_interval;

            self.slices.push(ExecutionSlice {
                index: i,
                target_quantity: quantity,
                filled_quantity: Decimal::ZERO,
                scheduled_time,
                executed_time: None,
                execution_price: None,
                status: SliceStatus::Pending,
            });
        }

        // Distribute any remaining quantity to last slice
        if remaining > Decimal::ZERO {
            if let Some(last) = self.slices.last_mut() {
                last.target_quantity += remaining;
            }
        }
    }

    /// Start execution
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.is_paused = false;
        self.pause_reason = None;
    }

    /// Update current market price
    pub fn update_price(&mut self, price: Decimal) {
        self.current_price = price;

        // Check price deviation
        let deviation = if self.target_price > Decimal::ZERO {
            ((price - self.target_price) / self.target_price).abs()
        } else {
            Decimal::ZERO
        };

        if deviation > self.config.max_price_deviation {
            if !self.is_paused {
                self.is_paused = true;
                self.pause_reason = Some(format!(
                    "Price deviation {:.2}% exceeds max {:.2}%",
                    deviation * dec!(100),
                    self.config.max_price_deviation * dec!(100)
                ));
            }
        } else if self.is_paused && self.pause_reason.as_ref().map_or(false, |r| r.contains("deviation")) {
            // Resume if deviation is back within limits
            self.is_paused = false;
            self.pause_reason = None;
        }
    }

    /// Update recent market volume (for participation rate)
    pub fn update_volume(&mut self, volume: Decimal) {
        self.recent_volumes.push_back(volume);
        if self.recent_volumes.len() > 60 {
            self.recent_volumes.pop_front();
        }
    }

    /// Get next slice to execute (if any)
    pub fn get_next_slice(&self) -> Option<&ExecutionSlice> {
        if self.is_paused {
            return None;
        }

        let elapsed = self.start_time.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);

        self.slices.iter().find(|s| {
            s.status == SliceStatus::Pending && s.scheduled_time <= elapsed
        })
    }

    /// Get slice quantity adjusted for participation rate
    pub fn get_adjusted_quantity(&self, slice: &ExecutionSlice) -> Decimal {
        if self.recent_volumes.is_empty() {
            return slice.target_quantity;
        }

        // Calculate recent average volume
        let avg_volume: Decimal = self.recent_volumes.iter().cloned().sum::<Decimal>()
            / Decimal::from(self.recent_volumes.len() as u32);

        // Calculate max quantity based on participation rate
        let max_participation = avg_volume * self.config.max_participation_rate;

        slice.target_quantity.min(max_participation).max(self.config.min_slice_size)
    }

    /// Record slice execution
    pub fn record_execution(
        &mut self,
        slice_index: u32,
        filled_quantity: Decimal,
        execution_price: Decimal,
    ) {
        let elapsed = self.start_time.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);

        if let Some(slice) = self.slices.iter_mut().find(|s| s.index == slice_index) {
            slice.filled_quantity = filled_quantity;
            slice.executed_time = Some(elapsed);
            slice.execution_price = Some(execution_price);

            slice.status = if filled_quantity >= slice.target_quantity {
                SliceStatus::Filled
            } else if filled_quantity > Decimal::ZERO {
                SliceStatus::PartialFill
            } else {
                SliceStatus::Failed
            };

            // Update total filled and VWAP
            let prev_filled = self.filled_quantity;
            self.filled_quantity += filled_quantity;

            if self.filled_quantity > Decimal::ZERO {
                // Update VWAP incrementally
                self.vwap_achieved = (self.vwap_achieved * prev_filled
                    + execution_price * filled_quantity)
                    / self.filled_quantity;
            }
        }
    }

    /// Cancel remaining execution
    pub fn cancel(&mut self, reason: &str) {
        for slice in &mut self.slices {
            if slice.status == SliceStatus::Pending {
                slice.status = SliceStatus::Cancelled;
            }
        }
        self.is_paused = true;
        self.pause_reason = Some(format!("Cancelled: {}", reason));
    }

    /// Get execution progress (0.0 - 1.0)
    pub fn progress(&self) -> Decimal {
        if self.total_quantity == Decimal::ZERO {
            return Decimal::ONE;
        }
        self.filled_quantity / self.total_quantity
    }

    /// Get execution summary
    pub fn summary(&self) -> ExecutionSummary {
        let filled_slices = self.slices.iter().filter(|s| {
            matches!(s.status, SliceStatus::Filled | SliceStatus::PartialFill)
        }).count();

        let avg_price = if filled_slices > 0 {
            self.vwap_achieved
        } else {
            Decimal::ZERO
        };

        let price_improvement = if self.target_price > Decimal::ZERO && avg_price > Decimal::ZERO {
            if self.is_buy {
                (self.target_price - avg_price) / self.target_price
            } else {
                (avg_price - self.target_price) / self.target_price
            }
        } else {
            Decimal::ZERO
        };

        let remaining_quantity = self.total_quantity - self.filled_quantity;

        ExecutionSummary {
            algorithm: self.algorithm,
            total_quantity: self.total_quantity,
            filled_quantity: self.filled_quantity,
            remaining_quantity,
            target_price: self.target_price,
            achieved_vwap: avg_price,
            price_improvement,
            total_slices: self.slices.len(),
            completed_slices: filled_slices,
            elapsed_seconds: self.start_time.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0),
            is_complete: remaining_quantity <= Decimal::ZERO,
            is_paused: self.is_paused,
            pause_reason: self.pause_reason.clone(),
        }
    }
}

/// Execution summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    pub algorithm: ExecutionAlgorithm,
    pub total_quantity: Decimal,
    pub filled_quantity: Decimal,
    pub remaining_quantity: Decimal,
    pub target_price: Decimal,
    pub achieved_vwap: Decimal,
    pub price_improvement: Decimal,
    pub total_slices: usize,
    pub completed_slices: usize,
    pub elapsed_seconds: f64,
    pub is_complete: bool,
    pub is_paused: bool,
    pub pause_reason: Option<String>,
}

/// Helper to create default volume profile (typical crypto pattern)
pub fn default_crypto_volume_profile() -> Vec<VolumeBucket> {
    // Crypto typically has higher volume during US/EU market overlap
    // Weights sum to 1.0
    let weights = [
        0.025, 0.020, 0.018, 0.018, 0.018, 0.021, // 00-06 UTC (Asia evening) = 0.120
        0.028, 0.033, 0.038, 0.043, 0.048, 0.053, // 06-12 UTC (EU morning) = 0.243
        0.063, 0.068, 0.072, 0.072, 0.068, 0.062, // 12-18 UTC (US/EU overlap - peak) = 0.405
        0.055, 0.050, 0.045, 0.040, 0.025, 0.017, // 18-24 UTC (US afternoon/evening) = 0.232
    ];

    weights
        .iter()
        .enumerate()
        .map(|(h, &w)| VolumeBucket {
            minute: (h * 60 / 24) as u8,
            weight: Decimal::from_f64_retain(w).unwrap_or(dec!(0.04)),
            avg_volume: Decimal::from_f64_retain(w * 100000.0).unwrap_or(dec!(4000)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_slices_even_distribution() {
        let config = ExecutionConfig {
            duration: Duration::from_secs(100),
            num_slices: 10,
            min_slice_size: dec!(1.0),
            randomize_timing: false,
            ..Default::default()
        };

        let engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
        );

        assert_eq!(engine.slices.len(), 10);

        // Each slice should have ~10.0 quantity
        for slice in &engine.slices {
            assert_eq!(slice.target_quantity, dec!(10.0));
        }

        // Check timing is evenly spaced
        assert!((engine.slices[0].scheduled_time - 0.0).abs() < 0.01);
        assert!((engine.slices[1].scheduled_time - 10.0).abs() < 0.01);
        assert!((engine.slices[9].scheduled_time - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_twap_min_slice_size() {
        let config = ExecutionConfig {
            duration: Duration::from_secs(100),
            num_slices: 100, // Would result in 0.1 per slice
            min_slice_size: dec!(5.0),
            randomize_timing: false,
            ..Default::default()
        };

        let engine = ExecutionEngine::new_twap(
            dec!(10.0),
            true,
            dec!(0.50),
            config,
        );

        // Should only create 2 slices (10/5 = 2)
        assert_eq!(engine.slices.len(), 2);
        assert_eq!(engine.slices[0].target_quantity, dec!(5.0));
        assert_eq!(engine.slices[1].target_quantity, dec!(5.0));
    }

    #[test]
    fn test_vwap_weighted_distribution() {
        let profile = vec![
            VolumeBucket { minute: 0, weight: dec!(0.1), avg_volume: dec!(1000) },
            VolumeBucket { minute: 30, weight: dec!(0.3), avg_volume: dec!(3000) },
            VolumeBucket { minute: 45, weight: dec!(0.6), avg_volume: dec!(6000) },
        ];

        let config = ExecutionConfig {
            duration: Duration::from_secs(60),
            num_slices: 3,
            min_slice_size: dec!(1.0),
            randomize_timing: false,
            ..Default::default()
        };

        let engine = ExecutionEngine::new_vwap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
            profile,
        );

        assert_eq!(engine.slices.len(), 3);

        // Heavier weight buckets should have more quantity
        let total: Decimal = engine.slices.iter().map(|s| s.target_quantity).sum();
        assert!((total - dec!(100.0)).abs() < dec!(0.01));
    }

    #[test]
    fn test_price_deviation_pause() {
        let config = ExecutionConfig {
            max_price_deviation: dec!(0.05), // 5%
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
        );

        engine.start();
        assert!(!engine.is_paused);

        // Price moves 3% - should not pause
        engine.update_price(dec!(0.515));
        assert!(!engine.is_paused);

        // Price moves 10% - should pause
        engine.update_price(dec!(0.55));
        assert!(engine.is_paused);
        assert!(engine.pause_reason.as_ref().unwrap().contains("deviation"));

        // Price returns to normal - should resume
        engine.update_price(dec!(0.51));
        assert!(!engine.is_paused);
    }

    #[test]
    fn test_execution_recording() {
        let config = ExecutionConfig {
            duration: Duration::from_secs(100),
            num_slices: 5,
            randomize_timing: false,
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
        );

        engine.start();

        // Record first execution
        engine.record_execution(0, dec!(20.0), dec!(0.49));
        assert_eq!(engine.filled_quantity, dec!(20.0));
        assert_eq!(engine.vwap_achieved, dec!(0.49));
        assert_eq!(engine.slices[0].status, SliceStatus::Filled);

        // Record second execution at different price
        engine.record_execution(1, dec!(20.0), dec!(0.51));
        assert_eq!(engine.filled_quantity, dec!(40.0));

        // VWAP should be average: (0.49*20 + 0.51*20) / 40 = 0.50
        assert_eq!(engine.vwap_achieved, dec!(0.50));
    }

    #[test]
    fn test_participation_rate_limit() {
        let config = ExecutionConfig {
            max_participation_rate: dec!(0.10), // 10%
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(1000.0),
            true,
            dec!(0.50),
            config,
        );

        // Add volume observations
        for _ in 0..10 {
            engine.update_volume(dec!(100.0));
        }

        // Target slice is 100, but 10% of 100 volume = 10
        let slice = &engine.slices[0];
        let adjusted = engine.get_adjusted_quantity(slice);

        assert!(adjusted <= dec!(10.0)); // Capped by participation rate
    }

    #[test]
    fn test_summary_calculation() {
        let config = ExecutionConfig {
            duration: Duration::from_secs(100),
            num_slices: 4,
            randomize_timing: false,
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true, // buying
            dec!(0.50),
            config,
        );

        engine.start();

        // Execute 2 slices at better price (0.48 instead of 0.50)
        engine.record_execution(0, dec!(25.0), dec!(0.48));
        engine.record_execution(1, dec!(25.0), dec!(0.48));

        let summary = engine.summary();

        assert_eq!(summary.total_quantity, dec!(100.0));
        assert_eq!(summary.filled_quantity, dec!(50.0));
        assert_eq!(summary.remaining_quantity, dec!(50.0));
        assert_eq!(summary.completed_slices, 2);
        assert!(!summary.is_complete);

        // Price improvement: (0.50 - 0.48) / 0.50 = 0.04 = 4%
        assert_eq!(summary.price_improvement, dec!(0.04));
    }

    #[test]
    fn test_cancel_execution() {
        let config = ExecutionConfig {
            num_slices: 5,
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
        );

        engine.start();
        engine.record_execution(0, dec!(20.0), dec!(0.50));

        engine.cancel("Market conditions changed");

        assert!(engine.is_paused);
        assert!(engine.pause_reason.as_ref().unwrap().contains("Cancelled"));

        // Remaining slices should be cancelled
        assert_eq!(engine.slices[1].status, SliceStatus::Cancelled);
        assert_eq!(engine.slices[2].status, SliceStatus::Cancelled);

        // First slice should remain filled
        assert_eq!(engine.slices[0].status, SliceStatus::Filled);
    }

    #[test]
    fn test_default_volume_profile() {
        let profile = default_crypto_volume_profile();

        assert_eq!(profile.len(), 24);

        // Weights should sum to ~1.0
        let total_weight: Decimal = profile.iter().map(|b| b.weight).sum();
        assert!((total_weight - dec!(1.0)).abs() < dec!(0.01));

        // Peak hours (12-18 UTC) should have higher weights
        let peak_weight: Decimal = profile[12..18].iter().map(|b| b.weight).sum();
        let off_peak_weight: Decimal = profile[0..6].iter().map(|b| b.weight).sum();

        assert!(peak_weight > off_peak_weight);
    }

    #[test]
    fn test_adaptive_algorithm() {
        let profile = default_crypto_volume_profile();
        let config = ExecutionConfig::default();

        // With profile - should use Adaptive
        let engine = ExecutionEngine::new_adaptive(
            dec!(100.0),
            true,
            dec!(0.50),
            config.clone(),
            Some(profile),
        );
        assert_eq!(engine.algorithm, ExecutionAlgorithm::Adaptive);

        // Without profile - should fall back to TWAP
        let engine2 = ExecutionEngine::new_adaptive(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
            None,
        );
        assert_eq!(engine2.algorithm, ExecutionAlgorithm::TWAP);
    }

    #[test]
    fn test_progress_tracking() {
        let config = ExecutionConfig {
            num_slices: 4,
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new_twap(
            dec!(100.0),
            true,
            dec!(0.50),
            config,
        );

        assert_eq!(engine.progress(), dec!(0.0));

        engine.start();
        engine.record_execution(0, dec!(25.0), dec!(0.50));
        assert_eq!(engine.progress(), dec!(0.25));

        engine.record_execution(1, dec!(25.0), dec!(0.50));
        assert_eq!(engine.progress(), dec!(0.50));

        engine.record_execution(2, dec!(25.0), dec!(0.50));
        engine.record_execution(3, dec!(25.0), dec!(0.50));
        assert_eq!(engine.progress(), dec!(1.0));
    }
}
