//! Dry Run Simulation - 1000 USDC
//! Records all trades for optimization analysis

use polymarket_bot::testing::dry_run::{DryRunConfig, DryRunEngine};
use polymarket_bot::config::{RiskConfig, StrategyConfig};
use rust_decimal_macros::dec;
use chrono::Utc;
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Starting Dry Run Simulation");
    println!("💰 Initial Balance: 1000 USDC");
    println!("📅 Start Time: {}", Utc::now());
    println!("─".repeat(50));
    
    // Configure simulation
    let config = DryRunConfig {
        initial_balance: dec!(1000),
        simulation_days: 7,
        trades_per_day: 10,
        record_all_signals: true,
        ..Default::default()
    };
    
    let strategy_config = StrategyConfig {
        min_edge: dec!(0.02),      // 2% minimum edge
        min_confidence: dec!(0.6), // 60% confidence threshold
        kelly_fraction: dec!(0.25), // Quarter Kelly for safety
        ..Default::default()
    };
    
    let risk_config = RiskConfig {
        max_position_pct: dec!(0.05),  // 5% max per position
        max_daily_loss_pct: dec!(0.10), // 10% daily loss limit
        max_total_exposure_pct: dec!(0.50), // 50% max exposure
        ..Default::default()
    };
    
    // Run simulation
    let mut engine = DryRunEngine::new(config, strategy_config, risk_config);
    let result = engine.run_simulation().await?;
    
    // Print results
    println!("\n📊 SIMULATION RESULTS");
    println!("─".repeat(50));
    println!("Starting Balance:  ${:.2}", result.starting_balance);
    println!("Ending Balance:    ${:.2}", result.ending_balance);
    println!("Total P&L:         ${:.2} ({:+.2}%)", 
        result.total_pnl, 
        result.total_pnl / result.starting_balance * dec!(100));
    println!("─".repeat(50));
    println!("Total Trades:      {}", result.total_trades);
    println!("Winning Trades:    {} ({:.1}%)", 
        result.winning_trades,
        result.winning_trades as f64 / result.total_trades as f64 * 100.0);
    println!("Losing Trades:     {}", result.losing_trades);
    println!("─".repeat(50));
    println!("Best Trade:        ${:.2}", result.best_trade);
    println!("Worst Trade:       ${:.2}", result.worst_trade);
    println!("Avg Win:           ${:.2}", result.avg_win);
    println!("Avg Loss:          ${:.2}", result.avg_loss);
    println!("─".repeat(50));
    println!("Max Drawdown:      {:.2}%", result.max_drawdown * dec!(100));
    println!("Sharpe Ratio:      {:.2}", result.sharpe_ratio);
    println!("Win Rate:          {:.1}%", result.win_rate * dec!(100));
    
    // Save detailed results to file
    let log_file = format!("logs/dry_run_{}.json", Utc::now().format("%Y%m%d_%H%M%S"));
    std::fs::create_dir_all("logs")?;
    let mut file = File::create(&log_file)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&result)?)?;
    println!("\n📁 Detailed results saved to: {}", log_file);
    
    // Save trade history
    let trades_file = format!("logs/trades_{}.csv", Utc::now().format("%Y%m%d_%H%M%S"));
    let mut trades_csv = File::create(&trades_file)?;
    writeln!(trades_csv, "timestamp,market_id,side,size,entry_price,exit_price,pnl,edge,confidence")?;
    for trade in &result.trades {
        writeln!(trades_csv, "{},{},{:?},{},{},{},{},{},{}",
            trade.timestamp,
            trade.market_id,
            trade.side,
            trade.size,
            trade.entry_price,
            trade.exit_price,
            trade.pnl,
            trade.edge,
            trade.confidence
        )?;
    }
    println!("📁 Trade history saved to: {}", trades_file);
    
    Ok(())
}
