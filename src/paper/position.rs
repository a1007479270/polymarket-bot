//! Position tracking for paper trading

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Position side (YES or NO outcome)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Yes,
    No,
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Yes => write!(f, "YES"),
            PositionSide::No => write!(f, "NO"),
        }
    }
}

/// Position status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionStatus {
    /// Position is open
    Open,
    /// Position closed with profit
    ClosedWin,
    /// Position closed with loss
    ClosedLoss,
    /// Market resolved in favor
    ResolvedWin,
    /// Market resolved against
    ResolvedLoss,
}

/// A simulated position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Unique position ID
    pub id: String,
    /// Market ID
    pub market_id: String,
    /// Market question
    pub question: String,
    /// Token ID
    pub token_id: String,
    /// Position side
    pub side: PositionSide,
    /// Number of shares
    pub shares: Decimal,
    /// Entry price per share
    pub entry_price: Decimal,
    /// Total cost (shares * entry_price)
    pub cost_basis: Decimal,
    /// Current market price
    pub current_price: Decimal,
    /// Current value (shares * current_price)
    pub current_value: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Unrealized P&L percentage
    pub unrealized_pnl_pct: Decimal,
    /// Exit price (if closed)
    pub exit_price: Option<Decimal>,
    /// Realized P&L (if closed)
    pub realized_pnl: Option<Decimal>,
    /// Position status
    pub status: PositionStatus,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: Option<DateTime<Utc>>,
    /// Entry reason/signal
    pub entry_reason: String,
    /// Exit reason
    pub exit_reason: Option<String>,
}

impl Position {
    /// Create a new open position
    pub fn new(
        market_id: String,
        question: String,
        token_id: String,
        side: PositionSide,
        shares: Decimal,
        entry_price: Decimal,
        entry_reason: String,
    ) -> Self {
        let cost_basis = shares * entry_price;
        let id = format!("{}_{}", market_id, Utc::now().timestamp_millis());
        
        Self {
            id,
            market_id,
            question,
            token_id,
            side,
            shares,
            entry_price,
            cost_basis,
            current_price: entry_price,
            current_value: cost_basis,
            unrealized_pnl: dec!(0),
            unrealized_pnl_pct: dec!(0),
            exit_price: None,
            realized_pnl: None,
            status: PositionStatus::Open,
            entry_time: Utc::now(),
            exit_time: None,
            entry_reason,
            exit_reason: None,
        }
    }

    /// Update current price and recalculate unrealized P&L
    pub fn update_price(&mut self, new_price: Decimal) {
        self.current_price = new_price;
        self.current_value = self.shares * new_price;
        self.unrealized_pnl = self.current_value - self.cost_basis;
        
        if self.cost_basis > dec!(0) {
            self.unrealized_pnl_pct = (self.unrealized_pnl / self.cost_basis) * dec!(100);
        }
    }

    /// Close the position at given price
    pub fn close(&mut self, exit_price: Decimal, reason: String) {
        self.exit_price = Some(exit_price);
        self.exit_time = Some(Utc::now());
        self.exit_reason = Some(reason);
        
        let exit_value = self.shares * exit_price;
        let pnl = exit_value - self.cost_basis;
        self.realized_pnl = Some(pnl);
        
        self.status = if pnl >= dec!(0) {
            PositionStatus::ClosedWin
        } else {
            PositionStatus::ClosedLoss
        };
    }

    /// Mark as resolved (market outcome determined)
    pub fn resolve(&mut self, won: bool) {
        let exit_price = if won { dec!(1) } else { dec!(0) };
        self.exit_price = Some(exit_price);
        self.exit_time = Some(Utc::now());
        
        let exit_value = self.shares * exit_price;
        let pnl = exit_value - self.cost_basis;
        self.realized_pnl = Some(pnl);
        
        self.status = if won {
            PositionStatus::ResolvedWin
        } else {
            PositionStatus::ResolvedLoss
        };
        
        self.exit_reason = Some(format!("Market resolved: {}", if won { "WON" } else { "LOST" }));
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        self.status == PositionStatus::Open
    }

    /// Get hold duration
    pub fn hold_duration(&self) -> chrono::Duration {
        let end = self.exit_time.unwrap_or_else(Utc::now);
        end - self.entry_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new(
            "market123".to_string(),
            "Will BTC > 100k?".to_string(),
            "token_yes".to_string(),
            PositionSide::Yes,
            dec!(100),
            dec!(0.65),
            "RSI oversold".to_string(),
        );

        assert_eq!(pos.shares, dec!(100));
        assert_eq!(pos.entry_price, dec!(0.65));
        assert_eq!(pos.cost_basis, dec!(65));
        assert_eq!(pos.unrealized_pnl, dec!(0));
        assert!(pos.is_open());
    }

    #[test]
    fn test_price_update() {
        let mut pos = Position::new(
            "market123".to_string(),
            "Test market".to_string(),
            "token_yes".to_string(),
            PositionSide::Yes,
            dec!(100),
            dec!(0.50),
            "Test".to_string(),
        );

        // Price goes up
        pos.update_price(dec!(0.60));
        assert_eq!(pos.current_value, dec!(60));
        assert_eq!(pos.unrealized_pnl, dec!(10)); // 60 - 50
        assert_eq!(pos.unrealized_pnl_pct, dec!(20)); // 10/50 * 100

        // Price goes down
        pos.update_price(dec!(0.40));
        assert_eq!(pos.unrealized_pnl, dec!(-10)); // 40 - 50
    }

    #[test]
    fn test_close_position() {
        let mut pos = Position::new(
            "market123".to_string(),
            "Test".to_string(),
            "token".to_string(),
            PositionSide::Yes,
            dec!(100),
            dec!(0.50),
            "Test".to_string(),
        );

        pos.close(dec!(0.70), "Take profit".to_string());

        assert!(!pos.is_open());
        assert_eq!(pos.exit_price, Some(dec!(0.70)));
        assert_eq!(pos.realized_pnl, Some(dec!(20))); // (0.70 - 0.50) * 100
        assert_eq!(pos.status, PositionStatus::ClosedWin);
    }

    #[test]
    fn test_resolve_win() {
        let mut pos = Position::new(
            "market123".to_string(),
            "Test".to_string(),
            "token".to_string(),
            PositionSide::Yes,
            dec!(100),
            dec!(0.30),
            "Test".to_string(),
        );

        pos.resolve(true);

        assert_eq!(pos.exit_price, Some(dec!(1)));
        assert_eq!(pos.realized_pnl, Some(dec!(70))); // (1.0 - 0.30) * 100
        assert_eq!(pos.status, PositionStatus::ResolvedWin);
    }

    #[test]
    fn test_resolve_loss() {
        let mut pos = Position::new(
            "market123".to_string(),
            "Test".to_string(),
            "token".to_string(),
            PositionSide::Yes,
            dec!(100),
            dec!(0.70),
            "Test".to_string(),
        );

        pos.resolve(false);

        assert_eq!(pos.exit_price, Some(dec!(0)));
        assert_eq!(pos.realized_pnl, Some(dec!(-70))); // (0 - 0.70) * 100
        assert_eq!(pos.status, PositionStatus::ResolvedLoss);
    }
}
