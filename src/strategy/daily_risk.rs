//! Daily risk management and position limits
//!
//! Prevents catastrophic losses by enforcing:
//! 1. Daily max loss limit - stops trading after hitting loss threshold
//! 2. Daily max trades limit - prevents overtrading
//! 3. Concurrent position limit - diversification enforcement
//! 4. Correlation-based exposure limit - prevents concentrated bets

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for daily risk limits
#[derive(Debug, Clone)]
pub struct DailyRiskConfig {
    /// Maximum daily loss as percentage of starting balance (e.g., 0.05 = 5%)
    pub max_daily_loss_pct: Decimal,
    /// Maximum daily loss in absolute terms (USD)
    pub max_daily_loss_abs: Option<Decimal>,
    /// Maximum number of trades per day
    pub max_daily_trades: u32,
    /// Maximum concurrent open positions
    pub max_concurrent_positions: u32,
    /// Maximum exposure to any single category (e.g., "politics")
    pub max_category_exposure_pct: Decimal,
    /// Cooldown period after hitting daily loss (hours)
    pub loss_cooldown_hours: i64,
    /// Reset time for daily limits (UTC hour, 0-23)
    pub reset_hour_utc: u32,
}

impl Default for DailyRiskConfig {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: dec!(0.05),        // 5% max daily loss
            max_daily_loss_abs: Some(dec!(500)),   // $500 hard cap
            max_daily_trades: 20,                  // Max 20 trades per day
            max_concurrent_positions: 10,          // Max 10 open positions
            max_category_exposure_pct: dec!(0.30), // Max 30% in any category
            loss_cooldown_hours: 4,                // 4 hour cooldown after loss limit
            reset_hour_utc: 0,                     // Reset at midnight UTC
        }
    }
}

/// Tracks a single trade for risk calculation
#[derive(Debug, Clone)]
pub struct TrackedTrade {
    pub market_id: String,
    pub category: Option<String>,
    pub entry_time: DateTime<Utc>,
    pub entry_price: Decimal,
    pub size_usd: Decimal,
    pub realized_pnl: Option<Decimal>,
    pub is_closed: bool,
}

/// Current risk state
#[derive(Debug, Clone)]
pub struct RiskState {
    pub daily_pnl: Decimal,
    pub daily_trades: u32,
    pub open_positions: u32,
    pub can_trade: bool,
    pub blocked_reason: Option<String>,
    pub category_exposures: HashMap<String, Decimal>,
    pub time_until_reset: Duration,
}

/// Daily risk limiter - enforces risk constraints
pub struct DailyRiskLimiter {
    config: DailyRiskConfig,
    /// Today's starting balance
    starting_balance: RwLock<Decimal>,
    /// Today's realized PnL
    daily_pnl: RwLock<Decimal>,
    /// Trade count today
    trade_count: RwLock<u32>,
    /// Open positions by market_id
    open_positions: RwLock<HashMap<String, TrackedTrade>>,
    /// Time when daily loss limit was hit (for cooldown)
    loss_limit_hit_at: RwLock<Option<DateTime<Utc>>>,
    /// Last reset timestamp
    last_reset: RwLock<DateTime<Utc>>,
}

impl DailyRiskLimiter {
    pub fn new(config: DailyRiskConfig, starting_balance: Decimal) -> Self {
        Self {
            config,
            starting_balance: RwLock::new(starting_balance),
            daily_pnl: RwLock::new(dec!(0)),
            trade_count: RwLock::new(0),
            open_positions: RwLock::new(HashMap::new()),
            loss_limit_hit_at: RwLock::new(None),
            last_reset: RwLock::new(Utc::now()),
        }
    }

    /// Check if we should reset daily counters
    fn maybe_reset(&self) {
        let now = Utc::now();
        let last = *self.last_reset.read().unwrap();
        
        // Check if we've passed the reset hour since last reset
        let today_reset = now.date_naive().and_hms_opt(
            self.config.reset_hour_utc, 0, 0
        ).unwrap().and_utc();
        
        if now >= today_reset && last < today_reset {
            // Reset all counters
            *self.daily_pnl.write().unwrap() = dec!(0);
            *self.trade_count.write().unwrap() = 0;
            *self.loss_limit_hit_at.write().unwrap() = None;
            *self.last_reset.write().unwrap() = now;
        }
    }

    /// Update starting balance (call at start of day or after deposit)
    pub fn update_starting_balance(&self, balance: Decimal) {
        *self.starting_balance.write().unwrap() = balance;
    }

    /// Check if trading is allowed
    pub fn can_open_position(&self, category: Option<&str>, size_usd: Decimal) -> RiskCheckResult {
        self.maybe_reset();

        let mut reasons = Vec::new();
        let mut allowed = true;

        // Check daily loss limit
        let daily_pnl = *self.daily_pnl.read().unwrap();
        let starting = *self.starting_balance.read().unwrap();
        let loss_pct = if starting > Decimal::ZERO {
            -daily_pnl / starting
        } else {
            dec!(0)
        };

        if loss_pct >= self.config.max_daily_loss_pct {
            allowed = false;
            reasons.push(format!(
                "Daily loss limit reached: {:.2}% >= {:.2}% max",
                loss_pct * dec!(100),
                self.config.max_daily_loss_pct * dec!(100)
            ));
        }

        // Check absolute loss limit
        if let Some(max_abs) = self.config.max_daily_loss_abs {
            if -daily_pnl >= max_abs {
                allowed = false;
                reasons.push(format!(
                    "Daily loss limit reached: ${:.2} >= ${:.2} max",
                    -daily_pnl, max_abs
                ));
            }
        }

        // Check cooldown period
        if let Some(hit_at) = *self.loss_limit_hit_at.read().unwrap() {
            let cooldown_end = hit_at + Duration::hours(self.config.loss_cooldown_hours);
            if Utc::now() < cooldown_end {
                allowed = false;
                let remaining = cooldown_end - Utc::now();
                reasons.push(format!(
                    "Loss cooldown active: {}h {}m remaining",
                    remaining.num_hours(),
                    remaining.num_minutes() % 60
                ));
            }
        }

        // Check trade count
        let trade_count = *self.trade_count.read().unwrap();
        if trade_count >= self.config.max_daily_trades {
            allowed = false;
            reasons.push(format!(
                "Daily trade limit reached: {} >= {} max",
                trade_count, self.config.max_daily_trades
            ));
        }

        // Check concurrent positions
        let open_count = self.open_positions.read().unwrap().len() as u32;
        if open_count >= self.config.max_concurrent_positions {
            allowed = false;
            reasons.push(format!(
                "Position limit reached: {} >= {} max",
                open_count, self.config.max_concurrent_positions
            ));
        }

        // Check category exposure
        if let Some(cat) = category {
            let positions = self.open_positions.read().unwrap();
            let category_exposure: Decimal = positions
                .values()
                .filter(|t| t.category.as_deref() == Some(cat))
                .map(|t| t.size_usd)
                .sum();
            
            let total_exposure: Decimal = positions.values().map(|t| t.size_usd).sum();
            let new_total = total_exposure + size_usd;
            let new_category = category_exposure + size_usd;
            
            if new_total > Decimal::ZERO {
                let category_pct = new_category / new_total;
                if category_pct > self.config.max_category_exposure_pct {
                    allowed = false;
                    reasons.push(format!(
                        "Category exposure too high: {} at {:.1}% > {:.1}% max",
                        cat,
                        category_pct * dec!(100),
                        self.config.max_category_exposure_pct * dec!(100)
                    ));
                }
            }
        }

        RiskCheckResult {
            allowed,
            reasons,
            daily_pnl,
            daily_loss_pct: loss_pct,
            trade_count,
            open_positions: open_count,
        }
    }

    /// Record a new trade being opened
    pub fn record_trade_open(
        &self,
        market_id: &str,
        category: Option<String>,
        entry_price: Decimal,
        size_usd: Decimal,
    ) {
        self.maybe_reset();

        let trade = TrackedTrade {
            market_id: market_id.to_string(),
            category,
            entry_time: Utc::now(),
            entry_price,
            size_usd,
            realized_pnl: None,
            is_closed: false,
        };

        self.open_positions.write().unwrap().insert(market_id.to_string(), trade);
        *self.trade_count.write().unwrap() += 1;
    }

    /// Record a trade being closed
    pub fn record_trade_close(&self, market_id: &str, pnl: Decimal) {
        self.maybe_reset();

        let mut positions = self.open_positions.write().unwrap();
        if let Some(trade) = positions.get_mut(market_id) {
            trade.realized_pnl = Some(pnl);
            trade.is_closed = true;
        }
        positions.remove(market_id);

        // Update daily PnL
        *self.daily_pnl.write().unwrap() += pnl;

        // Check if we hit loss limit
        let daily_pnl = *self.daily_pnl.read().unwrap();
        let starting = *self.starting_balance.read().unwrap();
        let loss_pct = if starting > Decimal::ZERO {
            -daily_pnl / starting
        } else {
            dec!(0)
        };

        if loss_pct >= self.config.max_daily_loss_pct {
            *self.loss_limit_hit_at.write().unwrap() = Some(Utc::now());
        }

        if let Some(max_abs) = self.config.max_daily_loss_abs {
            if -daily_pnl >= max_abs {
                *self.loss_limit_hit_at.write().unwrap() = Some(Utc::now());
            }
        }
    }

    /// Update unrealized PnL for an open position
    pub fn update_position_value(&self, market_id: &str, current_price: Decimal) {
        let mut positions = self.open_positions.write().unwrap();
        if let Some(trade) = positions.get_mut(market_id) {
            // Unrealized PnL = (current - entry) * size / entry
            let unrealized = (current_price - trade.entry_price) * trade.size_usd / trade.entry_price;
            trade.realized_pnl = Some(unrealized); // Store as unrealized
        }
    }

    /// Get current risk state
    pub fn get_state(&self) -> RiskState {
        self.maybe_reset();

        let daily_pnl = *self.daily_pnl.read().unwrap();
        let starting = *self.starting_balance.read().unwrap();
        let positions = self.open_positions.read().unwrap();

        // Calculate category exposures
        let mut category_exposures: HashMap<String, Decimal> = HashMap::new();
        for trade in positions.values() {
            if let Some(ref cat) = trade.category {
                *category_exposures.entry(cat.clone()).or_default() += trade.size_usd;
            }
        }

        // Calculate time until reset
        let now = Utc::now();
        let today_reset = now.date_naive().and_hms_opt(
            self.config.reset_hour_utc, 0, 0
        ).unwrap().and_utc();
        
        let next_reset = if now >= today_reset {
            today_reset + Duration::days(1)
        } else {
            today_reset
        };
        let time_until_reset = next_reset - now;

        // Check if can trade
        let check = self.can_open_position(None, dec!(0));

        RiskState {
            daily_pnl,
            daily_trades: *self.trade_count.read().unwrap(),
            open_positions: positions.len() as u32,
            can_trade: check.allowed,
            blocked_reason: check.reasons.first().cloned(),
            category_exposures,
            time_until_reset,
        }
    }

    /// Force close all positions recommendation (for emergency)
    pub fn should_force_close(&self) -> bool {
        let daily_pnl = *self.daily_pnl.read().unwrap();
        let starting = *self.starting_balance.read().unwrap();
        
        if starting <= Decimal::ZERO {
            return false;
        }

        // Force close if loss exceeds 1.5x daily limit
        let loss_pct = -daily_pnl / starting;
        loss_pct >= self.config.max_daily_loss_pct * dec!(1.5)
    }

    /// Get remaining risk budget
    pub fn remaining_budget(&self) -> RiskBudget {
        self.maybe_reset();

        let daily_pnl = *self.daily_pnl.read().unwrap();
        let starting = *self.starting_balance.read().unwrap();
        let trade_count = *self.trade_count.read().unwrap();
        let open_count = self.open_positions.read().unwrap().len() as u32;

        let max_loss = starting * self.config.max_daily_loss_pct;
        let current_loss = -daily_pnl.min(dec!(0));
        let remaining_loss_budget = (max_loss - current_loss).max(dec!(0));

        RiskBudget {
            remaining_loss_usd: remaining_loss_budget,
            remaining_loss_pct: if starting > Decimal::ZERO {
                remaining_loss_budget / starting
            } else {
                dec!(0)
            },
            remaining_trades: self.config.max_daily_trades.saturating_sub(trade_count),
            remaining_positions: self.config.max_concurrent_positions.saturating_sub(open_count),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub allowed: bool,
    pub reasons: Vec<String>,
    pub daily_pnl: Decimal,
    pub daily_loss_pct: Decimal,
    pub trade_count: u32,
    pub open_positions: u32,
}

#[derive(Debug, Clone)]
pub struct RiskBudget {
    pub remaining_loss_usd: Decimal,
    pub remaining_loss_pct: Decimal,
    pub remaining_trades: u32,
    pub remaining_positions: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter() -> DailyRiskLimiter {
        let config = DailyRiskConfig {
            max_daily_loss_pct: dec!(0.05),
            max_daily_loss_abs: Some(dec!(100)),
            max_daily_trades: 10,
            max_concurrent_positions: 5,
            max_category_exposure_pct: dec!(0.40),
            loss_cooldown_hours: 2,
            reset_hour_utc: 0,
        };
        DailyRiskLimiter::new(config, dec!(1000))
    }

    #[test]
    fn test_initial_state_allows_trading() {
        let limiter = make_limiter();
        let result = limiter.can_open_position(None, dec!(100));
        assert!(result.allowed);
        assert!(result.reasons.is_empty());
    }

    #[test]
    fn test_trade_count_limit() {
        let limiter = make_limiter();

        // Open max trades
        for i in 0..10 {
            limiter.record_trade_open(&format!("market_{}", i), None, dec!(0.50), dec!(50));
            limiter.record_trade_close(&format!("market_{}", i), dec!(1)); // Small profit
        }

        // 11th trade should be blocked
        let result = limiter.can_open_position(None, dec!(50));
        assert!(!result.allowed);
        assert!(result.reasons.iter().any(|r| r.contains("trade limit")));
    }

    #[test]
    fn test_concurrent_position_limit() {
        let limiter = make_limiter();

        // Open max positions
        for i in 0..5 {
            limiter.record_trade_open(&format!("market_{}", i), None, dec!(0.50), dec!(50));
        }

        // 6th position should be blocked
        let result = limiter.can_open_position(None, dec!(50));
        assert!(!result.allowed);
        assert!(result.reasons.iter().any(|r| r.contains("Position limit")));

        // Close one position
        limiter.record_trade_close("market_0", dec!(5));

        // Now should be allowed
        let result = limiter.can_open_position(None, dec!(50));
        assert!(result.allowed);
    }

    #[test]
    fn test_daily_loss_limit_percentage() {
        let limiter = make_limiter();

        // Lose 5% (the limit)
        limiter.record_trade_open("market_1", None, dec!(0.50), dec!(100));
        limiter.record_trade_close("market_1", dec!(-50)); // -$50 = -5%

        // Should be blocked
        let result = limiter.can_open_position(None, dec!(50));
        assert!(!result.allowed);
        assert!(result.reasons.iter().any(|r| r.contains("Daily loss limit")));
    }

    #[test]
    fn test_daily_loss_limit_absolute() {
        let limiter = make_limiter();

        // Lose $100 (the absolute limit)
        limiter.record_trade_open("market_1", None, dec!(0.50), dec!(200));
        limiter.record_trade_close("market_1", dec!(-100));

        // Should be blocked
        let result = limiter.can_open_position(None, dec!(50));
        assert!(!result.allowed);
    }

    #[test]
    fn test_category_exposure_limit() {
        let limiter = make_limiter();

        // Open positions in same category
        limiter.record_trade_open("market_1", Some("politics".to_string()), dec!(0.50), dec!(100));
        limiter.record_trade_open("market_2", Some("politics".to_string()), dec!(0.50), dec!(100));

        // Adding more to politics (would be >40% of total)
        // Current: 200 politics, 0 other. New: 200 politics, would be 100% which > 40%
        let result = limiter.can_open_position(Some("politics"), dec!(100));
        // Actually already at 100%, so should be blocked
        assert!(!result.allowed);
        assert!(result.reasons.iter().any(|r| r.contains("Category exposure")));

        // Different category should be fine
        let result = limiter.can_open_position(Some("crypto"), dec!(100));
        assert!(result.allowed);
    }

    #[test]
    fn test_remaining_budget() {
        let limiter = make_limiter();

        limiter.record_trade_open("market_1", None, dec!(0.50), dec!(50));
        limiter.record_trade_close("market_1", dec!(-20)); // -$20 loss

        let budget = limiter.remaining_budget();
        assert_eq!(budget.remaining_loss_usd, dec!(30)); // $50 max - $20 lost = $30
        assert_eq!(budget.remaining_trades, 9); // 10 - 1 = 9
        assert_eq!(budget.remaining_positions, 5); // Position was closed
    }

    #[test]
    fn test_force_close_recommendation() {
        let limiter = make_limiter();

        // Normal loss - no force close
        limiter.record_trade_open("market_1", None, dec!(0.50), dec!(100));
        limiter.record_trade_close("market_1", dec!(-40));
        assert!(!limiter.should_force_close());

        // Severe loss (> 1.5x limit) - recommend force close
        limiter.record_trade_open("market_2", None, dec!(0.50), dec!(100));
        limiter.record_trade_close("market_2", dec!(-40)); // Now at -$80 = 8%

        assert!(limiter.should_force_close());
    }

    #[test]
    fn test_get_state() {
        let limiter = make_limiter();

        limiter.record_trade_open("market_1", Some("politics".to_string()), dec!(0.50), dec!(100));
        limiter.record_trade_open("market_2", Some("crypto".to_string()), dec!(0.50), dec!(50));

        let state = limiter.get_state();
        assert_eq!(state.open_positions, 2);
        assert_eq!(state.daily_trades, 2);
        assert!(state.can_trade);
        assert_eq!(state.category_exposures.get("politics"), Some(&dec!(100)));
        assert_eq!(state.category_exposures.get("crypto"), Some(&dec!(50)));
    }

    #[test]
    fn test_profit_allows_more_risk() {
        let limiter = make_limiter();

        // Make profit
        limiter.record_trade_open("market_1", None, dec!(0.50), dec!(100));
        limiter.record_trade_close("market_1", dec!(20)); // +$20

        let budget = limiter.remaining_budget();
        // Still have full $50 loss budget (profits don't increase it, but we haven't lost anything)
        assert_eq!(budget.remaining_loss_usd, dec!(50));
    }
}
