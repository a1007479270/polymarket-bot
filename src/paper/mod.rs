//! Paper Trading Module
//!
//! Simulates trades using real market data without actual execution.
//! Perfect for strategy validation before going live.

mod position;
mod trader;
mod evaluator;
mod llm_trader;

pub use position::{Position, PositionSide, PositionStatus};
pub use trader::{PaperTrader, PaperTraderConfig, TradeRecord, TradeAction};
pub use evaluator::{MarketEvaluator, EvaluationResult, ConfidenceLevel};
pub use llm_trader::{LlmTrader, TradeDecision, PositionContext, MarketContext};

use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Paper trading portfolio summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    /// Starting balance
    pub initial_balance: Decimal,
    /// Current cash balance
    pub cash_balance: Decimal,
    /// Total value of open positions (at current prices)
    pub positions_value: Decimal,
    /// Total portfolio value (cash + positions)
    pub total_value: Decimal,
    /// Total realized P&L
    pub realized_pnl: Decimal,
    /// Total unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Total P&L (realized + unrealized)
    pub total_pnl: Decimal,
    /// ROI percentage
    pub roi_percent: Decimal,
    /// Number of trades
    pub trade_count: u32,
    /// Win rate
    pub win_rate: Decimal,
    /// Number of open positions
    pub open_positions: u32,
    /// Last update time
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_portfolio_summary() {
        let summary = PortfolioSummary {
            initial_balance: dec!(100),
            cash_balance: dec!(50),
            positions_value: dec!(60),
            total_value: dec!(110),
            realized_pnl: dec!(5),
            unrealized_pnl: dec!(5),
            total_pnl: dec!(10),
            roi_percent: dec!(10),
            trade_count: 5,
            win_rate: dec!(0.6),
            open_positions: 2,
            updated_at: Utc::now(),
        };
        
        assert_eq!(summary.total_value, dec!(110));
        assert_eq!(summary.roi_percent, dec!(10));
    }
}
