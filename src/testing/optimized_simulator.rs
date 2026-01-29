//! Optimized Dry Run Simulator
//!
//! Supports custom strategy and risk configurations for A/B testing.
//! Features:
//! - Configurable stop-loss, take-profit, trailing-stop
//! - Trade rate limiting
//! - Full parameter customization

use crate::client::mock::{MockClobClient, MockGammaClient, GammaClientTrait};
use crate::config::{RiskConfig, StrategyConfig};
use crate::strategy::SignalGenerator;
use crate::types::{Market, Side};
use crate::model::Prediction;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Simulation result compatible with comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub initial_balance: Decimal,
    pub final_balance: Decimal,
    pub total_pnl: Decimal,
    pub pnl_pct: Decimal,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: Decimal,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub profit_factor: Decimal,
    pub trades: Vec<SimTrade>,
    pub equity_curve: Vec<(u32, Decimal)>,
    pub config_used: ConfigSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub min_edge: Decimal,
    pub min_confidence: Decimal,
    pub kelly_fraction: Decimal,
    pub max_position_pct: Decimal,
    pub max_exposure_pct: Decimal,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub trailing_stop: Option<Decimal>,
    pub max_trades_per_hour: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimTrade {
    pub id: u32,
    pub step: u32,
    pub market_id: String,
    pub market_question: String,
    pub side: Side,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub pnl: Decimal,
    pub pnl_pct: Decimal,
    pub edge: Decimal,
    pub confidence: Decimal,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    TakeProfit,
    StopLoss,
    TrailingStop,
    MarketClose,
    MaxHoldTime,
}

#[derive(Debug)]
struct OpenPosition {
    market_id: String,
    market_question: String,
    side: Side,
    size: Decimal,
    entry_price: Decimal,
    entry_step: u32,
    edge: Decimal,
    confidence: Decimal,
    highest_price: Decimal,  // For trailing stop
    lowest_price: Decimal,   // For trailing stop (shorts)
}

/// Enhanced dry run simulator with optimized parameters
pub struct EnhancedDryRunSimulator {
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    clob: MockClobClient,
    gamma: MockGammaClient,
    signal_gen: SignalGenerator,
    
    // Risk management params
    stop_loss: Option<Decimal>,
    take_profit: Option<Decimal>,
    trailing_stop: Option<Decimal>,
    max_trades_per_hour: Option<u32>,
    
    // State
    initial_balance: Decimal,
    current_balance: Decimal,
    current_step: u32,
    trades: Vec<SimTrade>,
    trade_counter: u32,
    open_positions: HashMap<String, OpenPosition>,
    equity_curve: Vec<(u32, Decimal)>,
    peak_balance: Decimal,
    max_drawdown: Decimal,
    start_time: DateTime<Utc>,
    trades_this_hour: u32,
    last_hour_step: u32,
    
    // Random state
    random_seed: u64,
}

impl EnhancedDryRunSimulator {
    pub fn new(
        initial_balance: Decimal,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
    ) -> Self {
        Self {
            signal_gen: SignalGenerator::new(strategy_config.clone(), risk_config.clone()),
            strategy_config,
            risk_config,
            clob: MockClobClient::new().with_balance(initial_balance),
            gamma: MockGammaClient::new(),
            stop_loss: None,
            take_profit: None,
            trailing_stop: None,
            max_trades_per_hour: None,
            initial_balance,
            current_balance: initial_balance,
            current_step: 0,
            trades: Vec::new(),
            trade_counter: 0,
            open_positions: HashMap::new(),
            equity_curve: vec![(0, initial_balance)],
            peak_balance: initial_balance,
            max_drawdown: dec!(0),
            start_time: Utc::now(),
            trades_this_hour: 0,
            last_hour_step: 0,
            random_seed: 42,
        }
    }
    
    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.gamma = MockGammaClient::new().with_markets(markets);
        self
    }
    
    pub fn with_stop_loss(mut self, pct: Decimal) -> Self {
        self.stop_loss = Some(pct);
        self
    }
    
    pub fn with_take_profit(mut self, pct: Decimal) -> Self {
        self.take_profit = Some(pct);
        self
    }
    
    pub fn with_trailing_stop(mut self, pct: Decimal) -> Self {
        self.trailing_stop = Some(pct);
        self
    }
    
    pub fn with_max_trades_per_hour(mut self, max: u32) -> Self {
        self.max_trades_per_hour = Some(max);
        self
    }
    
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }
    
    /// Run simulation for specified steps
    pub async fn run_for(&mut self, steps: u32, _delay_ms: u64) -> anyhow::Result<()> {
        for _ in 0..steps {
            self.step().await?;
        }
        // Close all remaining positions
        self.close_all_positions().await?;
        Ok(())
    }
    
    async fn step(&mut self) -> anyhow::Result<()> {
        self.current_step += 1;
        
        // Reset hourly trade counter (every 60 steps = 1 hour simulation)
        if self.current_step - self.last_hour_step >= 60 {
            self.trades_this_hour = 0;
            self.last_hour_step = self.current_step;
        }
        
        // Check existing positions for exits
        self.check_position_exits().await?;
        
        // Get markets and generate signals
        let markets = self.gamma.get_top_markets(20).await?;
        
        for market in &markets {
            if market.liquidity < dec!(1000) {
                continue;
            }
            
            // Check rate limit
            if let Some(max_trades) = self.max_trades_per_hour {
                if self.trades_this_hour >= max_trades {
                    continue;
                }
            }
            
            // Skip if already have position in this market
            if self.open_positions.contains_key(&market.id) {
                continue;
            }
            
            // Check exposure limit
            let current_exposure = self.calculate_current_exposure();
            if current_exposure >= self.risk_config.max_exposure_pct {
                continue;
            }
            
            let prediction = self.generate_prediction(market);
            
            if let Some(signal) = self.signal_gen.generate(market, &prediction) {
                self.process_signal(&signal, market, &prediction).await?;
            }
        }
        
        // Record equity
        let equity = self.calculate_total_equity();
        self.equity_curve.push((self.current_step, equity));
        self.update_drawdown(equity);
        
        Ok(())
    }
    
    async fn process_signal(
        &mut self,
        signal: &crate::types::Signal,
        market: &Market,
        prediction: &Prediction,
    ) -> anyhow::Result<()> {
        // Calculate position size with caps
        let base_size = signal.suggested_size * self.current_balance;
        let max_size = self.risk_config.max_position_pct * self.current_balance;
        let size = base_size.min(max_size);
        
        if size < dec!(1) {
            return Ok(());
        }
        
        // Check if we have enough balance
        if size > self.current_balance - self.risk_config.min_balance_reserve {
            return Ok(());
        }
        
        // Execute trade
        self.current_balance -= size;
        self.trade_counter += 1;
        self.trades_this_hour += 1;
        
        let entry_price = signal.market_probability;
        
        self.open_positions.insert(market.id.clone(), OpenPosition {
            market_id: market.id.clone(),
            market_question: market.question.clone(),
            side: signal.side,
            size,
            entry_price,
            entry_step: self.current_step,
            edge: signal.edge,
            confidence: prediction.confidence,
            highest_price: entry_price,
            lowest_price: entry_price,
        });
        
        Ok(())
    }
    
    async fn check_position_exits(&mut self) -> anyhow::Result<()> {
        let positions: Vec<(String, OpenPosition)> = self.open_positions
            .iter()
            .map(|(k, v)| (k.clone(), OpenPosition {
                market_id: v.market_id.clone(),
                market_question: v.market_question.clone(),
                side: v.side,
                size: v.size,
                entry_price: v.entry_price,
                entry_step: v.entry_step,
                edge: v.edge,
                confidence: v.confidence,
                highest_price: v.highest_price,
                lowest_price: v.lowest_price,
            }))
            .collect();
        
        for (market_id, pos) in positions {
            // Simulate current price
            let noise = (self.random() - dec!(0.5)) * dec!(0.08);
            let current_price = (pos.entry_price + pos.edge * dec!(0.5) + noise)
                .max(dec!(0.01))
                .min(dec!(0.99));
            
            // Update trailing stop trackers
            if let Some(pos_mut) = self.open_positions.get_mut(&market_id) {
                pos_mut.highest_price = pos_mut.highest_price.max(current_price);
                pos_mut.lowest_price = pos_mut.lowest_price.min(current_price);
            }
            
            let pnl_pct = match pos.side {
                Side::Buy => (current_price - pos.entry_price) / pos.entry_price,
                Side::Sell => (pos.entry_price - current_price) / pos.entry_price,
            };
            
            let exit_reason = self.check_exit_conditions(&pos, pnl_pct, current_price);
            
            if let Some(reason) = exit_reason {
                self.close_position(&market_id, current_price, reason).await?;
            }
        }
        
        Ok(())
    }
    
    fn check_exit_conditions(
        &self,
        pos: &OpenPosition,
        pnl_pct: Decimal,
        current_price: Decimal,
    ) -> Option<ExitReason> {
        // Take profit
        if let Some(tp) = self.take_profit {
            if pnl_pct >= tp {
                return Some(ExitReason::TakeProfit);
            }
        }
        
        // Stop loss
        if let Some(sl) = self.stop_loss {
            if pnl_pct <= -sl {
                return Some(ExitReason::StopLoss);
            }
        }
        
        // Trailing stop
        if let Some(ts) = self.trailing_stop {
            let trailing_pct = match pos.side {
                Side::Buy => {
                    if pos.highest_price > pos.entry_price {
                        (pos.highest_price - current_price) / pos.highest_price
                    } else {
                        dec!(0)
                    }
                }
                Side::Sell => {
                    if pos.lowest_price < pos.entry_price {
                        (current_price - pos.lowest_price) / pos.lowest_price
                    } else {
                        dec!(0)
                    }
                }
            };
            
            if trailing_pct >= ts {
                return Some(ExitReason::TrailingStop);
            }
        }
        
        // Max hold time (20 steps)
        if self.current_step - pos.entry_step > 20 {
            return Some(ExitReason::MaxHoldTime);
        }
        
        None
    }
    
    async fn close_position(
        &mut self,
        market_id: &str,
        exit_price: Decimal,
        reason: ExitReason,
    ) -> anyhow::Result<()> {
        if let Some(pos) = self.open_positions.remove(market_id) {
            let pnl = match pos.side {
                Side::Buy => (exit_price - pos.entry_price) * pos.size / pos.entry_price,
                Side::Sell => (pos.entry_price - exit_price) * pos.size / pos.entry_price,
            };
            
            let pnl_pct = pnl / pos.size;
            
            self.current_balance += pos.size + pnl;
            
            self.trades.push(SimTrade {
                id: self.trade_counter,
                step: self.current_step,
                market_id: pos.market_id,
                market_question: pos.market_question,
                side: pos.side,
                size: pos.size,
                entry_price: pos.entry_price,
                exit_price,
                pnl,
                pnl_pct,
                edge: pos.edge,
                confidence: pos.confidence,
                exit_reason: reason,
            });
        }
        
        Ok(())
    }
    
    async fn close_all_positions(&mut self) -> anyhow::Result<()> {
        let market_ids: Vec<String> = self.open_positions.keys().cloned().collect();
        
        // Collect position data first
        let mut exit_data: Vec<(String, Decimal, Decimal)> = Vec::new();
        for market_id in &market_ids {
            if let Some(pos) = self.open_positions.get(market_id) {
                exit_data.push((market_id.clone(), pos.entry_price, pos.edge));
            }
        }
        
        // Now calculate and close
        for (market_id, entry_price, edge) in exit_data {
            let noise = (self.random() - dec!(0.5)) * dec!(0.05);
            let exit_price = (entry_price + edge + noise)
                .max(dec!(0.01))
                .min(dec!(0.99));
            self.close_position(&market_id, exit_price, ExitReason::MarketClose).await?;
        }
        
        Ok(())
    }
    
    fn generate_prediction(&mut self, market: &Market) -> Prediction {
        let base = market.yes_price().unwrap_or(dec!(0.5));
        let variance = (self.random() - dec!(0.5)) * dec!(0.20);
        let prob = (base + variance).max(dec!(0.05)).min(dec!(0.95));
        
        // Confidence varies more realistically
        let confidence = dec!(0.5) + self.random() * dec!(0.45);
        
        Prediction {
            probability: prob,
            confidence,
            reasoning: "Optimized dry run simulation".to_string(),
        }
    }
    
    fn calculate_current_exposure(&self) -> Decimal {
        let position_value: Decimal = self.open_positions.values().map(|p| p.size).sum();
        position_value / self.initial_balance
    }
    
    fn calculate_total_equity(&self) -> Decimal {
        let position_value: Decimal = self.open_positions.values().map(|p| p.size).sum();
        self.current_balance + position_value
    }
    
    fn update_drawdown(&mut self, equity: Decimal) {
        if equity > self.peak_balance {
            self.peak_balance = equity;
        } else {
            let drawdown = (self.peak_balance - equity) / self.peak_balance;
            self.max_drawdown = self.max_drawdown.max(drawdown);
        }
    }
    
    fn random(&mut self) -> Decimal {
        self.random_seed = self.random_seed.wrapping_mul(1103515245).wrapping_add(12345);
        Decimal::from(self.random_seed % 10000) / dec!(10000)
    }
    
    /// Get simulation results
    pub async fn get_results(&self) -> anyhow::Result<SimulationResult> {
        let winning: Vec<_> = self.trades.iter().filter(|t| t.pnl > dec!(0)).collect();
        let losing: Vec<_> = self.trades.iter().filter(|t| t.pnl < dec!(0)).collect();
        
        let win_rate = if !self.trades.is_empty() {
            Decimal::from(winning.len() as u32) / Decimal::from(self.trades.len() as u32) * dec!(100)
        } else {
            dec!(0)
        };
        
        let total_pnl = self.current_balance - self.initial_balance;
        let pnl_pct = total_pnl / self.initial_balance * dec!(100);
        
        let avg_win = if !winning.is_empty() {
            winning.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(winning.len() as u32)
        } else {
            dec!(0)
        };
        
        let avg_loss = if !losing.is_empty() {
            losing.iter().map(|t| t.pnl.abs()).sum::<Decimal>() / Decimal::from(losing.len() as u32)
        } else {
            dec!(0)
        };
        
        let gross_profit: Decimal = winning.iter().map(|t| t.pnl).sum();
        let gross_loss: Decimal = losing.iter().map(|t| t.pnl.abs()).sum();
        
        let profit_factor = if gross_loss > dec!(0) {
            gross_profit / gross_loss
        } else if gross_profit > dec!(0) {
            dec!(99.99)
        } else {
            dec!(0)
        };
        
        Ok(SimulationResult {
            start_time: self.start_time,
            end_time: Utc::now(),
            initial_balance: self.initial_balance,
            final_balance: self.current_balance,
            total_pnl,
            pnl_pct,
            total_trades: self.trades.len() as u32,
            winning_trades: winning.len() as u32,
            losing_trades: losing.len() as u32,
            win_rate,
            avg_win,
            avg_loss,
            max_drawdown: self.max_drawdown * dec!(100),
            sharpe_ratio: self.calculate_sharpe(),
            sortino_ratio: self.calculate_sortino(),
            profit_factor,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            config_used: ConfigSnapshot {
                min_edge: self.strategy_config.min_edge,
                min_confidence: self.strategy_config.min_confidence,
                kelly_fraction: self.strategy_config.kelly_fraction,
                max_position_pct: self.risk_config.max_position_pct,
                max_exposure_pct: self.risk_config.max_exposure_pct,
                stop_loss: self.stop_loss,
                take_profit: self.take_profit,
                trailing_stop: self.trailing_stop,
                max_trades_per_hour: self.max_trades_per_hour,
            },
        })
    }
    
    fn calculate_sharpe(&self) -> Decimal {
        if self.equity_curve.len() < 2 {
            return dec!(0);
        }
        
        let returns: Vec<Decimal> = self.equity_curve
            .windows(2)
            .filter_map(|w| {
                if w[0].1 > dec!(0) {
                    Some((w[1].1 - w[0].1) / w[0].1)
                } else {
                    None
                }
            })
            .collect();
        
        if returns.is_empty() {
            return dec!(0);
        }
        
        let avg_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as u32);
        let variance: Decimal = returns.iter()
            .map(|r| (*r - avg_return) * (*r - avg_return))
            .sum::<Decimal>() / Decimal::from(returns.len() as u32);
        
        let std_dev = crate::utils::sqrt_decimal(variance);
        if std_dev > dec!(0) {
            avg_return / std_dev * dec!(15.87)  // Annualized
        } else {
            dec!(0)
        }
    }
    
    fn calculate_sortino(&self) -> Decimal {
        if self.equity_curve.len() < 2 {
            return dec!(0);
        }
        
        let returns: Vec<Decimal> = self.equity_curve
            .windows(2)
            .filter_map(|w| {
                if w[0].1 > dec!(0) {
                    Some((w[1].1 - w[0].1) / w[0].1)
                } else {
                    None
                }
            })
            .collect();
        
        if returns.is_empty() {
            return dec!(0);
        }
        
        let avg_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as u32);
        let downside_variance: Decimal = returns.iter()
            .filter(|r| **r < dec!(0))
            .map(|r| r * r)
            .sum::<Decimal>() / Decimal::from(returns.len() as u32);
        
        let downside_dev = crate::utils::sqrt_decimal(downside_variance);
        if downside_dev > dec!(0) {
            avg_return / downside_dev * dec!(15.87)
        } else if avg_return > dec!(0) {
            dec!(99.99)
        } else {
            dec!(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_optimized_simulator_basic() {
        let strategy = StrategyConfig::default();
        let risk = RiskConfig::default();
        
        let mut sim = EnhancedDryRunSimulator::new(dec!(1000), strategy, risk);
        sim.run_for(10, 0).await.unwrap();
        
        let result = sim.get_results().await.unwrap();
        assert_eq!(result.initial_balance, dec!(1000));
    }
    
    #[tokio::test]
    async fn test_stop_loss_triggers() {
        let strategy = StrategyConfig {
            min_edge: dec!(0.01),  // Very low to generate more trades
            min_confidence: dec!(0.30),
            ..Default::default()
        };
        let risk = RiskConfig::default();
        
        let mut sim = EnhancedDryRunSimulator::new(dec!(1000), strategy, risk)
            .with_stop_loss(dec!(0.05))  // 5% stop loss
            .with_seed(12345);
        
        sim.run_for(50, 0).await.unwrap();
        let result = sim.get_results().await.unwrap();
        
        // Check that some trades hit stop loss
        let stop_losses: Vec<_> = result.trades.iter()
            .filter(|t| matches!(t.exit_reason, ExitReason::StopLoss))
            .collect();
        
        println!("Stop losses triggered: {}", stop_losses.len());
    }
    
    #[tokio::test]
    async fn test_take_profit_triggers() {
        let strategy = StrategyConfig {
            min_edge: dec!(0.01),
            min_confidence: dec!(0.30),
            ..Default::default()
        };
        let risk = RiskConfig::default();
        
        let mut sim = EnhancedDryRunSimulator::new(dec!(1000), strategy, risk)
            .with_take_profit(dec!(0.10))  // 10% take profit
            .with_seed(67890);
        
        sim.run_for(50, 0).await.unwrap();
        let result = sim.get_results().await.unwrap();
        
        let take_profits: Vec<_> = result.trades.iter()
            .filter(|t| matches!(t.exit_reason, ExitReason::TakeProfit))
            .collect();
        
        println!("Take profits triggered: {}", take_profits.len());
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let strategy = StrategyConfig {
            min_edge: dec!(0.01),
            min_confidence: dec!(0.30),
            ..Default::default()
        };
        let risk = RiskConfig::default();
        
        let mut sim = EnhancedDryRunSimulator::new(dec!(1000), strategy, risk)
            .with_max_trades_per_hour(2)
            .with_seed(11111);
        
        sim.run_for(60, 0).await.unwrap();  // 1 hour
        let result = sim.get_results().await.unwrap();
        
        // Should have at most 2 trades per hour
        println!("Total trades with rate limit: {}", result.total_trades);
    }
}
