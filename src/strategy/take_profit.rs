//! Take-profit strategy - exit positions early when profitable
//!
//! Don't wait for market settlement, sell when price moves in our favor.

use crate::types::{Side, Signal};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::info;

/// Position tracking for take-profit
#[derive(Debug, Clone)]
pub struct Position {
    pub market_id: String,
    pub token_id: String,
    pub side: Side,
    pub entry_price: Decimal,
    pub size: Decimal,
    pub entry_time: DateTime<Utc>,
}

/// Take-profit manager with dynamic adjustment
pub struct TakeProfitManager {
    /// Active positions
    positions: HashMap<String, Position>,
    /// Base take-profit threshold
    base_take_profit: Decimal,
    /// Base stop-loss threshold  
    base_stop_loss: Decimal,
    /// Max hold time before forced exit (hours)
    max_hold_hours: u32,
    /// Highest price seen (for trailing stop)
    high_water_marks: HashMap<String, Decimal>,
    /// Consecutive wins/losses for dynamic adjustment
    recent_wins: i32,
}

impl TakeProfitManager {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            base_take_profit: dec!(0.06),  // 6% base profit target
            base_stop_loss: dec!(0.10),    // 10% base stop loss
            max_hold_hours: 4,
            high_water_marks: HashMap::new(),
            recent_wins: 0,
        }
    }

    /// Check if market is a 15-min crypto market
    fn is_15min_market(market_id: &str) -> bool {
        market_id.contains("15m") || market_id.contains("15min")
    }

    /// Check if market is hourly crypto
    fn is_hourly_market(market_id: &str) -> bool {
        market_id.contains("hourly") || market_id.contains("1h")
    }

    /// Get max hold time based on market type
    fn get_max_hold_mins(&self, market_id: &str) -> i64 {
        if Self::is_15min_market(market_id) {
            12  // Exit before 15-min market settles
        } else if Self::is_hourly_market(market_id) {
            50  // Exit before hourly market settles
        } else {
            self.max_hold_hours as i64 * 60
        }
    }

    /// Get take-profit for market type
    fn get_market_take_profit(&self, market_id: &str) -> Decimal {
        if Self::is_15min_market(market_id) {
            dec!(0.04)  // 4% for fast 15-min markets
        } else if Self::is_hourly_market(market_id) {
            dec!(0.05)  // 5% for hourly
        } else {
            self.base_take_profit
        }
    }

    /// Get stop-loss for market type  
    fn get_market_stop_loss(&self, market_id: &str) -> Decimal {
        if Self::is_15min_market(market_id) {
            dec!(0.06)  // 6% tight stop for 15-min
        } else if Self::is_hourly_market(market_id) {
            dec!(0.08)  // 8% for hourly
        } else {
            self.base_stop_loss
        }
    }

    /// Dynamic take-profit based on conditions
    fn dynamic_take_profit(&self, position: &Position, current_price: Decimal) -> Decimal {
        self.dynamic_take_profit_with_base(position, current_price, self.base_take_profit)
    }

    fn dynamic_take_profit_with_base(&self, position: &Position, _current_price: Decimal, base: Decimal) -> Decimal {
        let hold_mins = (Utc::now() - position.entry_time).num_minutes();
        
        let mut tp = base;
        
        // Time decay: lower target over time
        if hold_mins > 5 {
            tp = tp * dec!(0.85);
        }
        if hold_mins > 10 {
            tp = tp * dec!(0.75);
        }
        
        // Winning streak: slightly higher target
        if self.recent_wins > 2 {
            tp = tp * dec!(1.1);
        }
        
        tp.max(dec!(0.02)) // Minimum 2% target
    }

    /// Dynamic stop-loss with trailing
    fn dynamic_stop_loss(&self, position: &Position, current_price: Decimal) -> Decimal {
        self.dynamic_stop_loss_with_base(position, current_price, self.base_stop_loss)
    }

    fn dynamic_stop_loss_with_base(&self, position: &Position, current_price: Decimal, base: Decimal) -> Decimal {
        let market_id = &position.market_id;
        
        // Trailing stop: move stop-loss up as price rises
        let high = self.high_water_marks.get(market_id)
            .copied()
            .unwrap_or(position.entry_price);
        
        let profit_from_entry = (current_price - position.entry_price) / position.entry_price;
        
        // If we're up, tighten stop-loss
        if profit_from_entry > dec!(0.02) {
            let trail_stop = high * dec!(0.97); // 3% below high
            return (current_price - trail_stop) / current_price;
        }
        
        // Losing streak: tighter stop
        if self.recent_wins < -2 {
            return base * dec!(0.75);
        }
        
        base
    }

    /// Update high water mark
    pub fn update_price(&mut self, market_id: &str, price: Decimal) {
        let high = self.high_water_marks.entry(market_id.to_string())
            .or_insert(price);
        if price > *high {
            *high = price;
        }
    }

    /// Record a win/loss for dynamic adjustment
    pub fn record_result(&mut self, won: bool) {
        if won {
            self.recent_wins = (self.recent_wins + 1).min(5);
        } else {
            self.recent_wins = (self.recent_wins - 1).max(-5);
        }
    }

    /// Record a new position
    pub fn open_position(&mut self, signal: &Signal, size: Decimal) {
        let position = Position {
            market_id: signal.market_id.clone(),
            token_id: signal.token_id.clone(),
            side: signal.side.clone(),
            entry_price: signal.market_probability,
            size,
            entry_time: Utc::now(),
        };
        
        info!("📈 Opened position: {} @ {:.1}% size ${:.2}", 
            signal.market_id, signal.market_probability * dec!(100), size);
        
        self.positions.insert(signal.market_id.clone(), position);
    }

    /// Check if we should exit a position (dynamic)
    pub fn check_exit(&self, market_id: &str, current_price: Decimal) -> Option<ExitSignal> {
        let position = self.positions.get(market_id)?;
        
        let pnl_pct = match position.side {
            Side::Buy => (current_price - position.entry_price) / position.entry_price,
            Side::Sell => (position.entry_price - current_price) / position.entry_price,
        };
        
        let hold_mins = (Utc::now() - position.entry_time).num_minutes();
        let max_hold_mins = self.get_max_hold_mins(market_id);
        
        // Market-specific base thresholds
        let base_tp = self.get_market_take_profit(market_id);
        let base_sl = self.get_market_stop_loss(market_id);
        
        // Dynamic adjustments on top
        let take_profit_threshold = self.dynamic_take_profit_with_base(position, current_price, base_tp);
        let stop_loss_threshold = self.dynamic_stop_loss_with_base(position, current_price, base_sl);
        
        // Take profit
        if pnl_pct >= take_profit_threshold {
            info!("💰 TAKE PROFIT: {} +{:.1}% @ {:.1}% (target was {:.1}%)", 
                market_id, pnl_pct * dec!(100), current_price * dec!(100), 
                take_profit_threshold * dec!(100));
            return Some(ExitSignal {
                market_id: market_id.to_string(),
                token_id: position.token_id.clone(),
                reason: ExitReason::TakeProfit,
                pnl_pct,
                size: position.size,
            });
        }
        
        // Stop loss (dynamic)
        if pnl_pct <= -stop_loss_threshold {
            info!("🛑 STOP LOSS: {} {:.1}% @ {:.1}% (limit was {:.1}%)", 
                market_id, pnl_pct * dec!(100), current_price * dec!(100),
                stop_loss_threshold * dec!(100));
            return Some(ExitSignal {
                market_id: market_id.to_string(),
                token_id: position.token_id.clone(),
                reason: ExitReason::StopLoss,
                pnl_pct,
                size: position.size,
            });
        }
        
        // Break-even exit: take small profit before time runs out
        let break_even_time = (max_hold_mins as f64 * 0.7) as i64;
        if hold_mins > break_even_time && pnl_pct > dec!(0.01) {
            info!("⏰ BREAK-EVEN EXIT: {} +{:.1}% after {}min (max {})", 
                market_id, pnl_pct * dec!(100), hold_mins, max_hold_mins);
            return Some(ExitSignal {
                market_id: market_id.to_string(),
                token_id: position.token_id.clone(),
                reason: ExitReason::TimeLimit,
                pnl_pct,
                size: position.size,
            });
        }
        
        // Force exit before market settles
        if hold_mins >= max_hold_mins {
            info!("⏰ TIME EXIT: {} held {}min (max {}), pnl {:.1}%", 
                market_id, hold_mins, max_hold_mins, pnl_pct * dec!(100));
            return Some(ExitSignal {
                market_id: market_id.to_string(),
                token_id: position.token_id.clone(),
                reason: ExitReason::TimeLimit,
                pnl_pct,
                size: position.size,
            });
        }
        
        None
    }

    /// Close a position
    pub fn close_position(&mut self, market_id: &str) -> Option<Position> {
        self.positions.remove(market_id)
    }

    /// Get all open positions
    pub fn get_positions(&self) -> Vec<&Position> {
        self.positions.values().collect()
    }

    /// Check all positions for exits
    pub fn check_all_exits(&self, current_prices: &HashMap<String, Decimal>) -> Vec<ExitSignal> {
        let mut exits = Vec::new();
        
        for (market_id, price) in current_prices {
            if let Some(exit) = self.check_exit(market_id, *price) {
                exits.push(exit);
            }
        }
        
        exits
    }
}

#[derive(Debug, Clone)]
pub struct ExitSignal {
    pub market_id: String,
    pub token_id: String,
    pub reason: ExitReason,
    pub pnl_pct: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone)]
pub enum ExitReason {
    TakeProfit,
    StopLoss,
    TimeLimit,
}

impl Default for TakeProfitManager {
    fn default() -> Self {
        Self::new()
    }
}
