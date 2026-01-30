//! Paper trader - simulated trading execution

use super::{Position, PositionSide, PortfolioSummary};
use crate::client::GammaClient;
use crate::error::{BotError, Result};
use crate::types::Market;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Paper trader configuration
#[derive(Debug, Clone)]
pub struct PaperTraderConfig {
    /// Initial balance
    pub initial_balance: Decimal,
    /// Maximum position size as % of portfolio
    pub max_position_pct: Decimal,
    /// Simulated slippage percentage
    pub slippage_pct: Decimal,
    /// Simulated fee percentage
    pub fee_pct: Decimal,
    /// Auto-save interval (seconds)
    pub save_interval: u64,
    /// State file path
    pub state_file: Option<String>,
}

impl Default for PaperTraderConfig {
    fn default() -> Self {
        Self {
            initial_balance: dec!(1000),
            max_position_pct: dec!(10), // 10% max per position
            slippage_pct: dec!(0.1),    // 0.1% slippage
            fee_pct: dec!(0),           // Polymarket has 0 fees currently
            save_interval: 60,
            state_file: None,
        }
    }
}

/// Trade record for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub id: String,
    pub market_id: String,
    pub question: String,
    pub side: PositionSide,
    pub action: TradeAction,
    pub shares: Decimal,
    pub price: Decimal,
    pub total_value: Decimal,
    pub pnl: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeAction {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeAction::Buy => write!(f, "BUY"),
            TradeAction::Sell => write!(f, "SELL"),
        }
    }
}

/// Paper trading state (persisted)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTraderState {
    pub initial_balance: Decimal,
    pub cash_balance: Decimal,
    pub positions: Vec<Position>,
    pub trade_history: Vec<TradeRecord>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Paper trader - simulates trading with real market data
pub struct PaperTrader {
    config: PaperTraderConfig,
    state: Arc<RwLock<PaperTraderState>>,
    client: GammaClient,
}

impl PaperTrader {
    /// Create new paper trader
    pub fn new(config: PaperTraderConfig, client: GammaClient) -> Self {
        let state = PaperTraderState {
            initial_balance: config.initial_balance,
            cash_balance: config.initial_balance,
            positions: Vec::new(),
            trade_history: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            client,
        }
    }

    /// Load state from file
    pub async fn load_state(&self, path: &str) -> Result<()> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| BotError::Internal(format!("Failed to read state: {}", e)))?;
        
        let loaded_state: PaperTraderState = serde_json::from_str(&content)
            .map_err(|e| BotError::Internal(format!("Failed to parse state: {}", e)))?;
        
        let mut state = self.state.write().await;
        *state = loaded_state;
        
        info!("Loaded paper trading state with {} positions", state.positions.len());
        Ok(())
    }

    /// Save state to file
    pub async fn save_state(&self, path: &str) -> Result<()> {
        let state = self.state.read().await;
        let json = serde_json::to_string_pretty(&*state)
            .map_err(|e| BotError::Internal(format!("Failed to serialize state: {}", e)))?;
        
        tokio::fs::write(path, json.as_bytes()).await
            .map_err(|e| BotError::Internal(format!("Failed to write state: {}", e)))?;
        
        debug!("Saved paper trading state");
        Ok(())
    }

    /// Simulate buying shares in a market
    pub async fn buy(
        &self,
        market: &Market,
        side: PositionSide,
        amount_usd: Decimal,
        reason: String,
    ) -> Result<Position> {
        // Get current price from market outcomes
        let (token_id, current_price) = match side {
            PositionSide::Yes => {
                let outcome = market.outcomes.first()
                    .ok_or_else(|| BotError::MarketNotFound("No YES outcome".to_string()))?;
                (outcome.token_id.clone(), outcome.price)
            }
            PositionSide::No => {
                let outcome = market.outcomes.get(1)
                    .ok_or_else(|| BotError::MarketNotFound("No NO outcome".to_string()))?;
                (outcome.token_id.clone(), outcome.price)
            }
        };

        // Apply slippage (buy at slightly higher price)
        let slippage = current_price * self.config.slippage_pct / dec!(100);
        let execution_price = current_price + slippage;
        
        // Calculate shares
        let fee = amount_usd * self.config.fee_pct / dec!(100);
        let net_amount = amount_usd - fee;
        let shares = net_amount / execution_price;

        let mut state = self.state.write().await;

        // Check balance
        if amount_usd > state.cash_balance {
            return Err(BotError::InsufficientBalance {
                required: amount_usd,
                available: state.cash_balance,
            });
        }

        // Check position size limit
        let max_allowed = (state.cash_balance + self.positions_value_internal(&state.positions))
            * self.config.max_position_pct / dec!(100);
        if amount_usd > max_allowed {
            warn!(
                "Position size {} exceeds max {} ({}% of portfolio)",
                amount_usd, max_allowed, self.config.max_position_pct
            );
        }

        // Create position
        let position = Position::new(
            market.id.clone(),
            market.question.clone(),
            token_id,
            side,
            shares,
            execution_price,
            reason.clone(),
        );

        // Update state
        state.cash_balance -= amount_usd;
        
        // Record trade
        let trade = TradeRecord {
            id: position.id.clone(),
            market_id: position.market_id.clone(),
            question: position.question.clone(),
            side,
            action: TradeAction::Buy,
            shares,
            price: execution_price,
            total_value: amount_usd,
            pnl: None,
            timestamp: Utc::now(),
            reason,
        };
        state.trade_history.push(trade);
        state.positions.push(position.clone());
        state.updated_at = Utc::now();

        info!(
            "📝 PAPER BUY: {} {} @ {:.4} = ${:.2}",
            shares.round_dp(2), side, execution_price, amount_usd
        );

        Ok(position)
    }

    /// Simulate selling/closing a position
    pub async fn sell(&self, position_id: &str, reason: String) -> Result<TradeRecord> {
        let mut state = self.state.write().await;

        // Find position
        let pos_idx = state.positions.iter()
            .position(|p| p.id == position_id && p.is_open())
            .ok_or_else(|| BotError::Execution(format!("Position not found: {}", position_id)))?;

        // Extract data we need before mutating
        let position = &state.positions[pos_idx];
        let pos_id = position.id.clone();
        let market_id = position.market_id.clone();
        let question = position.question.clone();
        let side = position.side;
        let shares = position.shares;
        let cost_basis = position.cost_basis;
        let current_price = position.current_price;

        // Apply slippage (sell at slightly lower price)
        let slippage = current_price * self.config.slippage_pct / dec!(100);
        let execution_price = current_price - slippage;

        // Calculate proceeds
        let gross_proceeds = shares * execution_price;
        let fee = gross_proceeds * self.config.fee_pct / dec!(100);
        let net_proceeds = gross_proceeds - fee;
        let pnl = net_proceeds - cost_basis;

        // Now mutate: close position
        state.positions[pos_idx].close(execution_price, reason.clone());

        // Update cash
        state.cash_balance += net_proceeds;

        // Record trade
        let trade = TradeRecord {
            id: format!("{}_sell", pos_id),
            market_id,
            question,
            side,
            action: TradeAction::Sell,
            shares,
            price: execution_price,
            total_value: net_proceeds,
            pnl: Some(pnl),
            timestamp: Utc::now(),
            reason,
        };
        state.trade_history.push(trade.clone());
        state.updated_at = Utc::now();

        let emoji = if pnl >= dec!(0) { "✅" } else { "❌" };
        info!(
            "{} PAPER SELL: {} shares @ {:.4} = ${:.2} (PnL: ${:.2})",
            emoji, shares.round_dp(2), execution_price, net_proceeds, pnl
        );

        Ok(trade)
    }

    /// Update all position prices from market
    pub async fn update_prices(&self) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Collect market IDs for open positions
        let market_ids: Vec<String> = state.positions.iter()
            .filter(|p| p.is_open())
            .map(|p| p.market_id.clone())
            .collect();

        if market_ids.is_empty() {
            return Ok(());
        }

        // Fetch markets one by one
        for market_id in market_ids {
            if let Ok(market) = self.client.get_market(&market_id).await {
                // Update matching positions
                for pos in state.positions.iter_mut().filter(|p| p.market_id == market_id && p.is_open()) {
                    let new_price = match pos.side {
                        PositionSide::Yes => market.outcomes.first().map(|o| o.price),
                        PositionSide::No => market.outcomes.get(1).map(|o| o.price),
                    };
                    
                    if let Some(price) = new_price {
                        pos.update_price(price);
                    }
                }
            }
        }

        state.updated_at = Utc::now();
        Ok(())
    }

    /// Get portfolio summary
    pub async fn get_summary(&self) -> PortfolioSummary {
        let state = self.state.read().await;
        
        let positions_value = self.positions_value_internal(&state.positions);
        let total_value = state.cash_balance + positions_value;
        
        // Calculate realized PnL
        let realized_pnl: Decimal = state.trade_history.iter()
            .filter_map(|t| t.pnl)
            .sum();
        
        // Calculate unrealized PnL
        let unrealized_pnl: Decimal = state.positions.iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl)
            .sum();
        
        let total_pnl = total_value - state.initial_balance;
        let roi_percent = if state.initial_balance > dec!(0) {
            (total_pnl / state.initial_balance) * dec!(100)
        } else {
            dec!(0)
        };

        // Win rate
        let closed_trades: Vec<&TradeRecord> = state.trade_history.iter()
            .filter(|t| t.action == TradeAction::Sell && t.pnl.is_some())
            .collect();
        
        let wins = closed_trades.iter()
            .filter(|t| t.pnl.unwrap_or(dec!(0)) >= dec!(0))
            .count();
        
        let win_rate = if !closed_trades.is_empty() {
            Decimal::from(wins as u32) / Decimal::from(closed_trades.len() as u32)
        } else {
            dec!(0)
        };

        let open_positions = state.positions.iter()
            .filter(|p| p.is_open())
            .count() as u32;

        PortfolioSummary {
            initial_balance: state.initial_balance,
            cash_balance: state.cash_balance,
            positions_value,
            total_value,
            realized_pnl,
            unrealized_pnl,
            total_pnl,
            roi_percent,
            trade_count: state.trade_history.len() as u32,
            win_rate,
            open_positions,
            updated_at: state.updated_at,
        }
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Vec<Position> {
        self.state.read().await.positions.clone()
    }

    /// Get open positions only
    pub async fn get_open_positions(&self) -> Vec<Position> {
        self.state.read().await.positions.iter()
            .filter(|p| p.is_open())
            .cloned()
            .collect()
    }

    /// Get trade history
    pub async fn get_history(&self) -> Vec<TradeRecord> {
        self.state.read().await.trade_history.clone()
    }

    /// Get cash balance
    pub async fn get_balance(&self) -> Decimal {
        self.state.read().await.cash_balance
    }

    fn positions_value_internal(&self, positions: &[Position]) -> Decimal {
        positions.iter()
            .filter(|p| p.is_open())
            .map(|p| p.current_value)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Outcome;

    fn create_test_market() -> Market {
        Market {
            id: "test_market_123".to_string(),
            question: "Will BTC reach 100k?".to_string(),
            description: None,
            end_date: None,
            outcomes: vec![
                Outcome {
                    token_id: "yes_token".to_string(),
                    outcome: "Yes".to_string(),
                    price: dec!(0.65),
                },
                Outcome {
                    token_id: "no_token".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(0.35),
                },
            ],
            volume: dec!(50000),
            liquidity: dec!(10000),
            active: true,
            closed: false,
        }
    }

    fn create_test_client() -> GammaClient {
        GammaClient::new("https://gamma-api.polymarket.com").unwrap()
    }

    #[tokio::test]
    async fn test_paper_buy() {
        let config = PaperTraderConfig {
            initial_balance: dec!(1000),
            slippage_pct: dec!(0), // No slippage for test
            fee_pct: dec!(0),
            ..Default::default()
        };
        
        let client = create_test_client();
        let trader = PaperTrader::new(config, client);
        let market = create_test_market();

        let position = trader.buy(
            &market,
            PositionSide::Yes,
            dec!(100),
            "Test buy".to_string(),
        ).await.unwrap();

        assert_eq!(position.side, PositionSide::Yes);
        assert_eq!(position.cost_basis, dec!(100));
        assert!(position.shares > dec!(0));
        
        // Check balance reduced
        let balance = trader.get_balance().await;
        assert_eq!(balance, dec!(900));
    }

    #[tokio::test]
    async fn test_insufficient_balance() {
        let config = PaperTraderConfig {
            initial_balance: dec!(100),
            ..Default::default()
        };
        
        let client = create_test_client();
        let trader = PaperTrader::new(config, client);
        let market = create_test_market();

        let result = trader.buy(
            &market,
            PositionSide::Yes,
            dec!(200), // More than balance
            "Test".to_string(),
        ).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sell_position() {
        let config = PaperTraderConfig {
            initial_balance: dec!(1000),
            slippage_pct: dec!(0),
            fee_pct: dec!(0),
            ..Default::default()
        };
        
        let client = create_test_client();
        let trader = PaperTrader::new(config, client);
        let market = create_test_market();

        // Buy
        let position = trader.buy(
            &market,
            PositionSide::Yes,
            dec!(100),
            "Test".to_string(),
        ).await.unwrap();

        // Sell
        let trade = trader.sell(&position.id, "Take profit".to_string()).await.unwrap();

        assert_eq!(trade.action, TradeAction::Sell);
        assert!(trade.pnl.is_some());

        // Check position closed
        let positions = trader.get_open_positions().await;
        assert!(positions.is_empty());
    }

    #[tokio::test]
    async fn test_portfolio_summary() {
        let config = PaperTraderConfig {
            initial_balance: dec!(1000),
            slippage_pct: dec!(0),
            fee_pct: dec!(0),
            ..Default::default()
        };
        
        let client = create_test_client();
        let trader = PaperTrader::new(config, client);
        let market = create_test_market();

        // Buy position
        trader.buy(
            &market,
            PositionSide::Yes,
            dec!(100),
            "Test".to_string(),
        ).await.unwrap();

        let summary = trader.get_summary().await;
        
        assert_eq!(summary.initial_balance, dec!(1000));
        assert_eq!(summary.cash_balance, dec!(900));
        assert_eq!(summary.open_positions, 1);
        assert_eq!(summary.trade_count, 1);
    }
}
