//! Dry Run Simulation - 1000 USDC
//! Records all trades for optimization analysis

use polymarket_bot::testing::dry_run::DryRunSimulator;
use polymarket_bot::types::{Market, Outcome};
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};
use std::fs::File;
use std::io::Write;
use tracing_subscriber;

fn generate_test_markets() -> Vec<Market> {
    vec![
        Market {
            id: "btc-100k".to_string(),
            question: "Will BTC reach $100k by March 2026?".to_string(),
            description: Some("Bitcoin price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(365)),
            outcomes: vec![
                Outcome {
                    outcome: "Yes".to_string(),
                    token_id: "btc-100k-yes".to_string(),
                    price: dec!(0.45),
                },
                Outcome {
                    outcome: "No".to_string(),
                    token_id: "btc-100k-no".to_string(),
                    price: dec!(0.55),
                },
            ],
            volume: dec!(500000),
            liquidity: dec!(100000),
            active: true,
            closed: false,
        },
        Market {
            id: "eth-5k".to_string(),
            question: "Will ETH reach $5k by Feb 2026?".to_string(),
            description: Some("Ethereum price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(300)),
            outcomes: vec![
                Outcome {
                    outcome: "Yes".to_string(),
                    token_id: "eth-5k-yes".to_string(),
                    price: dec!(0.35),
                },
                Outcome {
                    outcome: "No".to_string(),
                    token_id: "eth-5k-no".to_string(),
                    price: dec!(0.65),
                },
            ],
            volume: dec!(300000),
            liquidity: dec!(80000),
            active: true,
            closed: false,
        },
        Market {
            id: "fed-rate".to_string(),
            question: "Will Fed cut rates in Q1 2026?".to_string(),
            description: Some("Federal Reserve policy".to_string()),
            end_date: Some(Utc::now() + Duration::days(200)),
            outcomes: vec![
                Outcome {
                    outcome: "Yes".to_string(),
                    token_id: "fed-rate-yes".to_string(),
                    price: dec!(0.60),
                },
                Outcome {
                    outcome: "No".to_string(),
                    token_id: "fed-rate-no".to_string(),
                    price: dec!(0.40),
                },
            ],
            volume: dec!(200000),
            liquidity: dec!(50000),
            active: true,
            closed: false,
        },
        Market {
            id: "trump-approval".to_string(),
            question: "Trump approval above 50% by June 2026?".to_string(),
            description: Some("Political approval rating".to_string()),
            end_date: Some(Utc::now() + Duration::days(400)),
            outcomes: vec![
                Outcome {
                    outcome: "Yes".to_string(),
                    token_id: "trump-approval-yes".to_string(),
                    price: dec!(0.32),
                },
                Outcome {
                    outcome: "No".to_string(),
                    token_id: "trump-approval-no".to_string(),
                    price: dec!(0.68),
                },
            ],
            volume: dec!(450000),
            liquidity: dec!(120000),
            active: true,
            closed: false,
        },
        Market {
            id: "sp500-6k".to_string(),
            question: "Will S&P 500 reach 6000 by Dec 2025?".to_string(),
            description: Some("Stock market prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(150)),
            outcomes: vec![
                Outcome {
                    outcome: "Yes".to_string(),
                    token_id: "sp500-6k-yes".to_string(),
                    price: dec!(0.72),
                },
                Outcome {
                    outcome: "No".to_string(),
                    token_id: "sp500-6k-no".to_string(),
                    price: dec!(0.28),
                },
            ],
            volume: dec!(180000),
            liquidity: dec!(45000),
            active: true,
            closed: false,
        },
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Init tracing (INFO level for normal output)
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    println!("🚀 Starting Dry Run Simulation");
    println!("💰 Initial Balance: 1000 USDC");
    println!("📅 Start Time: {}", Utc::now());
    println!("{}", "─".repeat(50));
    
    // Create simulator with 1000 USDC
    let markets = generate_test_markets();
    let mut simulator = DryRunSimulator::new(dec!(1000))
        .with_markets(markets);
    
    // Run simulation for 100 steps (no delay for speed)
    println!("Running simulation with 100 steps...");
    simulator.run_for(100, 0).await?;
    
    // Get results
    let result = simulator.get_results().await?;
    
    // Print results
    println!("\n📊 SIMULATION RESULTS");
    println!("{}", "─".repeat(50));
    println!("{}", simulator.generate_report(&result));
    
    // Save detailed results to file
    std::fs::create_dir_all("logs")?;
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let log_file = format!("logs/dry_run_{}.json", timestamp);
    let mut file = File::create(&log_file)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&result)?)?;
    println!("\n📁 Detailed results saved to: {}", log_file);
    
    // Save trade history as CSV
    let trades_file = format!("logs/trades_{}.csv", timestamp);
    let mut trades_csv = File::create(&trades_file)?;
    writeln!(trades_csv, "id,timestamp,market_id,market_question,side,size,entry_price,exit_price,pnl,pnl_pct,edge,confidence")?;
    for trade in &result.trades {
        writeln!(trades_csv, "{},{},{},{:?},{:?},{},{},{},{},{},{},{}",
            trade.id,
            trade.timestamp,
            trade.market_id,
            trade.market_question,
            trade.side,
            trade.size,
            trade.entry_price,
            trade.exit_price.unwrap_or(dec!(0)),
            trade.pnl,
            trade.pnl_pct,
            trade.edge,
            trade.confidence
        )?;
    }
    println!("📁 Trade history saved to: {}", trades_file);
    
    // Print summary
    println!("\n🎯 SUMMARY");
    println!("═══════════════════════════════════════════════════");
    println!("Initial Balance: ${:.2}", result.initial_balance);
    println!("Final Balance:   ${:.2}", result.final_balance);
    println!("Total P&L:       ${:.2} ({:.2}%)", result.total_pnl, result.pnl_pct);
    println!("───────────────────────────────────────────────────");
    println!("Total Trades:    {}", result.total_trades);
    println!("Winning Trades:  {} ({:.1}%)", result.winning_trades, result.win_rate);
    println!("Losing Trades:   {}", result.losing_trades);
    println!("Average Win:     ${:.2}", result.avg_win);
    println!("Average Loss:    ${:.2}", result.avg_loss);
    println!("Max Drawdown:    {:.2}%", result.max_drawdown);
    println!("Sharpe Ratio:    {:.2}", result.sharpe_ratio);
    println!("═══════════════════════════════════════════════════");
    
    println!("\n✅ Simulation complete!");
    
    Ok(())
}
