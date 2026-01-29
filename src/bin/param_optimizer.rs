//! Parameter Optimizer - Grid Search for Optimal Strategy Parameters
//!
//! Tests multiple parameter combinations to find the best risk-adjusted returns
//! Evaluation metrics (by importance):
//! 1. Sharpe Ratio (risk-adjusted returns)
//! 2. Total Return %
//! 3. Max Drawdown (must be < 20%)
//! 4. Profit Factor (> 2.0)

use polymarket_bot::testing::optimized_simulator::EnhancedDryRunSimulator;
use polymarket_bot::config::{StrategyConfig, RiskConfig};
use polymarket_bot::types::{Market, Outcome};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

/// Parameter combination to test
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParamSet {
    pub min_edge: Decimal,
    pub min_confidence: Decimal,
    pub kelly_fraction: Decimal,
    pub max_position_pct: Decimal,
}

/// Result for a single parameter combination
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationResult {
    pub params: ParamSet,
    pub sharpe_ratio: Decimal,
    pub total_return_pct: Decimal,
    pub max_drawdown_pct: Decimal,
    pub profit_factor: Decimal,
    pub win_rate: Decimal,
    pub total_trades: u32,
    pub final_balance: Decimal,
    pub sortino_ratio: Decimal,
    pub composite_score: Decimal,  // Weighted score for ranking
}

/// Generate all test markets
fn generate_test_markets() -> Vec<Market> {
    vec![
        Market {
            id: "btc-100k".to_string(),
            question: "Will BTC reach $100k by March 2026?".to_string(),
            description: Some("Bitcoin price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(365)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "btc-100k-yes".to_string(), price: dec!(0.45) },
                Outcome { outcome: "No".to_string(), token_id: "btc-100k-no".to_string(), price: dec!(0.55) },
            ],
            volume: dec!(500000), liquidity: dec!(100000), active: true, closed: false,
        },
        Market {
            id: "eth-5k".to_string(),
            question: "Will ETH reach $5k by Feb 2026?".to_string(),
            description: Some("Ethereum price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(300)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "eth-5k-yes".to_string(), price: dec!(0.35) },
                Outcome { outcome: "No".to_string(), token_id: "eth-5k-no".to_string(), price: dec!(0.65) },
            ],
            volume: dec!(300000), liquidity: dec!(80000), active: true, closed: false,
        },
        Market {
            id: "fed-rate".to_string(),
            question: "Will Fed cut rates in Q1 2026?".to_string(),
            description: Some("Federal Reserve policy".to_string()),
            end_date: Some(Utc::now() + Duration::days(200)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "fed-rate-yes".to_string(), price: dec!(0.60) },
                Outcome { outcome: "No".to_string(), token_id: "fed-rate-no".to_string(), price: dec!(0.40) },
            ],
            volume: dec!(200000), liquidity: dec!(50000), active: true, closed: false,
        },
        Market {
            id: "trump-approval".to_string(),
            question: "Trump approval above 50% by June 2026?".to_string(),
            description: Some("Political approval rating".to_string()),
            end_date: Some(Utc::now() + Duration::days(400)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "trump-approval-yes".to_string(), price: dec!(0.32) },
                Outcome { outcome: "No".to_string(), token_id: "trump-approval-no".to_string(), price: dec!(0.68) },
            ],
            volume: dec!(450000), liquidity: dec!(120000), active: true, closed: false,
        },
        Market {
            id: "sp500-6k".to_string(),
            question: "Will S&P 500 reach 6000 by Dec 2025?".to_string(),
            description: Some("Stock market prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(150)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "sp500-6k-yes".to_string(), price: dec!(0.72) },
                Outcome { outcome: "No".to_string(), token_id: "sp500-6k-no".to_string(), price: dec!(0.28) },
            ],
            volume: dec!(180000), liquidity: dec!(45000), active: true, closed: false,
        },
        Market {
            id: "ai-regulation".to_string(),
            question: "Will major AI regulation pass in US by 2026?".to_string(),
            description: Some("AI policy prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(250)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "ai-reg-yes".to_string(), price: dec!(0.40) },
                Outcome { outcome: "No".to_string(), token_id: "ai-reg-no".to_string(), price: dec!(0.60) },
            ],
            volume: dec!(150000), liquidity: dec!(35000), active: true, closed: false,
        },
        Market {
            id: "sol-500".to_string(),
            question: "Will SOL reach $500 by Dec 2025?".to_string(),
            description: Some("Solana price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(180)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "sol-500-yes".to_string(), price: dec!(0.25) },
                Outcome { outcome: "No".to_string(), token_id: "sol-500-no".to_string(), price: dec!(0.75) },
            ],
            volume: dec!(220000), liquidity: dec!(55000), active: true, closed: false,
        },
        Market {
            id: "gold-3k".to_string(),
            question: "Will Gold reach $3000/oz by June 2025?".to_string(),
            description: Some("Gold price prediction".to_string()),
            end_date: Some(Utc::now() + Duration::days(120)),
            outcomes: vec![
                Outcome { outcome: "Yes".to_string(), token_id: "gold-3k-yes".to_string(), price: dec!(0.55) },
                Outcome { outcome: "No".to_string(), token_id: "gold-3k-no".to_string(), price: dec!(0.45) },
            ],
            volume: dec!(280000), liquidity: dec!(70000), active: true, closed: false,
        },
    ]
}

/// Calculate composite score for ranking
fn calculate_composite_score(result: &OptimizationResult) -> Decimal {
    // Weights: Sharpe 40%, Return 30%, Drawdown 20%, Profit Factor 10%
    let sharpe_score = result.sharpe_ratio.min(dec!(5)) / dec!(5) * dec!(40);  // Cap at 5
    let return_score = (result.total_return_pct / dec!(100)).min(dec!(1)) * dec!(30);  // Cap at 100%
    let dd_score = (dec!(20) - result.max_drawdown_pct).max(dec!(0)) / dec!(20) * dec!(20);  // Inverted
    let pf_score = (result.profit_factor.min(dec!(5)) / dec!(5)) * dec!(10);  // Cap at 5
    
    sharpe_score + return_score + dd_score + pf_score
}

/// Run simulation with given parameters
async fn run_simulation(params: &ParamSet, markets: &[Market], seed: u64) -> anyhow::Result<OptimizationResult> {
    let strategy = StrategyConfig {
        min_edge: params.min_edge,
        min_confidence: params.min_confidence,
        kelly_fraction: params.kelly_fraction,
        scan_interval_secs: 180,
        model_update_interval_secs: 900,
        compound_enabled: true,
        compound_sqrt_scaling: true,
    };
    
    let risk = RiskConfig {
        max_position_pct: params.max_position_pct,
        max_exposure_pct: dec!(0.30),  // Fixed at 30%
        max_daily_loss_pct: dec!(0.05),  // Fixed at 5%
        min_balance_reserve: dec!(100),
        max_open_positions: 5,
    };
    
    let mut sim = EnhancedDryRunSimulator::new(dec!(1000), strategy, risk)
        .with_markets(markets.to_vec())
        .with_stop_loss(dec!(0.15))
        .with_take_profit(dec!(0.25))
        .with_trailing_stop(dec!(0.10))
        .with_max_trades_per_hour(3)
        .with_seed(seed);
    
    sim.run_for(100, 0).await?;
    let result = sim.get_results().await?;
    
    let mut opt_result = OptimizationResult {
        params: params.clone(),
        sharpe_ratio: result.sharpe_ratio,
        total_return_pct: result.pnl_pct,
        max_drawdown_pct: result.max_drawdown,
        profit_factor: result.profit_factor,
        win_rate: result.win_rate,
        total_trades: result.total_trades,
        final_balance: result.final_balance,
        sortino_ratio: result.sortino_ratio,
        composite_score: dec!(0),
    };
    
    opt_result.composite_score = calculate_composite_score(&opt_result);
    Ok(opt_result)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║           🔬 POLYMARKET BOT - PARAMETER OPTIMIZATION                  ║");
    println!("║           Finding the optimal strategy for maximum returns            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    let markets = generate_test_markets();
    
    // Define parameter ranges
    let _min_edges = [dec!(0.02), dec!(0.03), dec!(0.04), dec!(0.05)];
    let _min_confidences = [dec!(0.55), dec!(0.60), dec!(0.65), dec!(0.70)];
    let _kelly_fractions = [dec!(0.15), dec!(0.20), dec!(0.25)];
    let _max_position_pcts = [dec!(0.02), dec!(0.03), dec!(0.04)];
    
    // Generate strategic parameter combinations (not full grid - too expensive)
    // Focus on promising combinations based on domain knowledge
    let param_sets: Vec<ParamSet> = vec![
        // Conservative
        ParamSet { min_edge: dec!(0.05), min_confidence: dec!(0.70), kelly_fraction: dec!(0.15), max_position_pct: dec!(0.02) },
        ParamSet { min_edge: dec!(0.04), min_confidence: dec!(0.65), kelly_fraction: dec!(0.15), max_position_pct: dec!(0.02) },
        // Moderate Conservative
        ParamSet { min_edge: dec!(0.04), min_confidence: dec!(0.65), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        ParamSet { min_edge: dec!(0.03), min_confidence: dec!(0.65), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.02) },
        // Balanced
        ParamSet { min_edge: dec!(0.03), min_confidence: dec!(0.60), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        ParamSet { min_edge: dec!(0.04), min_confidence: dec!(0.60), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        // Moderate Aggressive
        ParamSet { min_edge: dec!(0.03), min_confidence: dec!(0.60), kelly_fraction: dec!(0.25), max_position_pct: dec!(0.03) },
        ParamSet { min_edge: dec!(0.02), min_confidence: dec!(0.60), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        // Aggressive
        ParamSet { min_edge: dec!(0.02), min_confidence: dec!(0.55), kelly_fraction: dec!(0.25), max_position_pct: dec!(0.04) },
        ParamSet { min_edge: dec!(0.03), min_confidence: dec!(0.55), kelly_fraction: dec!(0.25), max_position_pct: dec!(0.04) },
        // High Confidence Focus
        ParamSet { min_edge: dec!(0.03), min_confidence: dec!(0.70), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        ParamSet { min_edge: dec!(0.02), min_confidence: dec!(0.70), kelly_fraction: dec!(0.25), max_position_pct: dec!(0.03) },
        // Low Edge Tolerance
        ParamSet { min_edge: dec!(0.02), min_confidence: dec!(0.65), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.03) },
        ParamSet { min_edge: dec!(0.02), min_confidence: dec!(0.60), kelly_fraction: dec!(0.15), max_position_pct: dec!(0.04) },
        // Extra Test Combinations
        ParamSet { min_edge: dec!(0.05), min_confidence: dec!(0.60), kelly_fraction: dec!(0.25), max_position_pct: dec!(0.04) },
        ParamSet { min_edge: dec!(0.04), min_confidence: dec!(0.55), kelly_fraction: dec!(0.20), max_position_pct: dec!(0.04) },
    ];
    
    println!("📊 Testing {} parameter combinations...\n", param_sets.len());
    println!("┌────┬──────────┬──────────┬─────────┬───────────┬─────────┬────────┬──────────┬───────────┐");
    println!("│ #  │ MinEdge  │ MinConf  │ Kelly   │ MaxPos    │ Sharpe  │ Return │ Drawdown │ Score     │");
    println!("├────┼──────────┼──────────┼─────────┼───────────┼─────────┼────────┼──────────┼───────────┤");
    
    let mut all_results: Vec<OptimizationResult> = Vec::new();
    
    // Run multiple seeds for each param set and average
    for (idx, params) in param_sets.iter().enumerate() {
        let mut sharpe_sum = dec!(0);
        let mut return_sum = dec!(0);
        let mut drawdown_sum = dec!(0);
        let mut profit_factor_sum = dec!(0);
        let mut win_rate_sum = dec!(0);
        let mut trades_sum = 0u32;
        let mut final_balance_sum = dec!(0);
        let mut sortino_sum = dec!(0);
        
        let num_runs = 3;  // Average over 3 random seeds
        
        for seed in 0..num_runs {
            let result = run_simulation(params, &markets, seed as u64 * 12345 + 42).await?;
            sharpe_sum += result.sharpe_ratio;
            return_sum += result.total_return_pct;
            drawdown_sum += result.max_drawdown_pct;
            profit_factor_sum += result.profit_factor;
            win_rate_sum += result.win_rate;
            trades_sum += result.total_trades;
            final_balance_sum += result.final_balance;
            sortino_sum += result.sortino_ratio;
        }
        
        let avg_result = OptimizationResult {
            params: params.clone(),
            sharpe_ratio: sharpe_sum / Decimal::from(num_runs),
            total_return_pct: return_sum / Decimal::from(num_runs),
            max_drawdown_pct: drawdown_sum / Decimal::from(num_runs),
            profit_factor: profit_factor_sum / Decimal::from(num_runs),
            win_rate: win_rate_sum / Decimal::from(num_runs),
            total_trades: trades_sum / num_runs as u32,
            final_balance: final_balance_sum / Decimal::from(num_runs),
            sortino_ratio: sortino_sum / Decimal::from(num_runs),
            composite_score: dec!(0),
        };
        
        let mut avg_result = avg_result;
        avg_result.composite_score = calculate_composite_score(&avg_result);
        
        // Check constraint: max drawdown < 20%
        let dd_ok = if avg_result.max_drawdown_pct < dec!(20) { "✓" } else { "✗" };
        
        println!("│ {:>2} │ {:>7}% │ {:>7}% │ {:>6}% │ {:>8}% │ {:>7.2} │ {:>5.1}% │ {:>5.1}% {} │ {:>8.2} │",
            idx + 1,
            (params.min_edge * dec!(100)).round_dp(0),
            (params.min_confidence * dec!(100)).round_dp(0),
            (params.kelly_fraction * dec!(100)).round_dp(0),
            (params.max_position_pct * dec!(100)).round_dp(0),
            avg_result.sharpe_ratio,
            avg_result.total_return_pct,
            avg_result.max_drawdown_pct,
            dd_ok,
            avg_result.composite_score,
        );
        
        all_results.push(avg_result);
    }
    
    println!("└────┴──────────┴──────────┴─────────┴───────────┴─────────┴────────┴──────────┴───────────┘");
    
    // Filter valid results (drawdown < 20%) and sort by composite score
    let mut valid_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.max_drawdown_pct < dec!(20))
        .cloned()
        .collect();
    
    valid_results.sort_by(|a, b| b.composite_score.partial_cmp(&a.composite_score).unwrap());
    
    // Print top 5
    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    🏆 TOP 5 PARAMETER COMBINATIONS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    for (rank, result) in valid_results.iter().take(5).enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        
        println!("{} #{}: Score={:.2}", medal, rank + 1, result.composite_score);
        println!("   Parameters:");
        println!("   - min_edge: {}%", (result.params.min_edge * dec!(100)).round_dp(0));
        println!("   - min_confidence: {}%", (result.params.min_confidence * dec!(100)).round_dp(0));
        println!("   - kelly_fraction: {}%", (result.params.kelly_fraction * dec!(100)).round_dp(0));
        println!("   - max_position_pct: {}%", (result.params.max_position_pct * dec!(100)).round_dp(0));
        println!("   Performance:");
        println!("   - Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("   - Total Return: {:.2}%", result.total_return_pct);
        println!("   - Max Drawdown: {:.2}%", result.max_drawdown_pct);
        println!("   - Profit Factor: {:.2}", result.profit_factor);
        println!("   - Win Rate: {:.1}%", result.win_rate);
        println!();
    }
    
    // Best result
    if let Some(best) = valid_results.first() {
        println!("╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║                    ⭐ OPTIMAL STRATEGY PARAMETERS                     ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  min_edge:         {}%", (best.params.min_edge * dec!(100)).round_dp(0));
        println!("  min_confidence:   {}%", (best.params.min_confidence * dec!(100)).round_dp(0));
        println!("  kelly_fraction:   {}%", (best.params.kelly_fraction * dec!(100)).round_dp(0));
        println!("  max_position_pct: {}%", (best.params.max_position_pct * dec!(100)).round_dp(0));
        println!();
        println!("  Expected Performance:");
        println!("  ├─ Sharpe Ratio:   {:.2}", best.sharpe_ratio);
        println!("  ├─ Annual Return:  ~{:.0}% (extrapolated)", best.total_return_pct * dec!(3.65));  // 100 steps ≈ 100 days
        println!("  ├─ Max Drawdown:   {:.2}%", best.max_drawdown_pct);
        println!("  ├─ Profit Factor:  {:.2}", best.profit_factor);
        println!("  └─ Win Rate:       {:.1}%", best.win_rate);
        println!();
        
        // Industry comparison
        println!("╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║                    📊 INDUSTRY COMPARISON                             ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  ┌────────────────────┬───────────┬───────────┬───────────┐");
        println!("  │ Metric             │ Our Bot   │ Industry  │ Status    │");
        println!("  │                    │           │ Average   │           │");
        println!("  ├────────────────────┼───────────┼───────────┼───────────┤");
        
        let sharpe_status = if best.sharpe_ratio > dec!(1.5) { "🟢 ELITE" } else if best.sharpe_ratio > dec!(1.0) { "🟡 GOOD" } else { "🔴 AVG" };
        println!("  │ Sharpe Ratio       │ {:>9.2} │    ~1.0   │ {} │", best.sharpe_ratio, sharpe_status);
        
        let return_status = if best.total_return_pct * dec!(3.65) > dec!(50) { "🟢 ELITE" } else if best.total_return_pct * dec!(3.65) > dec!(20) { "🟡 GOOD" } else { "🔴 AVG" };
        println!("  │ Annual Return      │ {:>8.0}% │   ~20%    │ {} │", best.total_return_pct * dec!(3.65), return_status);
        
        let dd_status = if best.max_drawdown_pct < dec!(10) { "🟢 ELITE" } else if best.max_drawdown_pct < dec!(15) { "🟡 GOOD" } else { "🔴 AVG" };
        println!("  │ Max Drawdown       │ {:>8.1}% │   ~15%    │ {} │", best.max_drawdown_pct, dd_status);
        
        let pf_status = if best.profit_factor > dec!(2.5) { "🟢 ELITE" } else if best.profit_factor > dec!(1.5) { "🟡 GOOD" } else { "🔴 AVG" };
        println!("  │ Profit Factor      │ {:>9.2} │    ~1.5   │ {} │", best.profit_factor, pf_status);
        
        println!("  └────────────────────┴───────────┴───────────┴───────────┘");
        println!();
    }
    
    // Save all results
    std::fs::create_dir_all("logs")?;
    
    let output = serde_json::json!({
        "timestamp": Utc::now().to_rfc3339(),
        "simulation_steps": 100,
        "initial_balance": "1000 USDC",
        "num_combinations_tested": all_results.len(),
        "num_runs_per_combination": 3,
        "evaluation_criteria": {
            "sharpe_ratio_weight": "40%",
            "return_weight": "30%",
            "drawdown_weight": "20%",
            "profit_factor_weight": "10%",
            "max_drawdown_constraint": "< 20%"
        },
        "optimal_parameters": valid_results.first().map(|r| &r.params),
        "optimal_performance": valid_results.first().map(|r| serde_json::json!({
            "sharpe_ratio": r.sharpe_ratio.to_string(),
            "total_return_pct": r.total_return_pct.to_string(),
            "max_drawdown_pct": r.max_drawdown_pct.to_string(),
            "profit_factor": r.profit_factor.to_string(),
            "win_rate": r.win_rate.to_string(),
            "composite_score": r.composite_score.to_string(),
        })),
        "all_results": all_results.iter().map(|r| serde_json::json!({
            "params": {
                "min_edge": r.params.min_edge.to_string(),
                "min_confidence": r.params.min_confidence.to_string(),
                "kelly_fraction": r.params.kelly_fraction.to_string(),
                "max_position_pct": r.params.max_position_pct.to_string(),
            },
            "sharpe_ratio": r.sharpe_ratio.to_string(),
            "total_return_pct": r.total_return_pct.to_string(),
            "max_drawdown_pct": r.max_drawdown_pct.to_string(),
            "profit_factor": r.profit_factor.to_string(),
            "win_rate": r.win_rate.to_string(),
            "total_trades": r.total_trades,
            "composite_score": r.composite_score.to_string(),
            "meets_constraints": r.max_drawdown_pct < dec!(20),
        })).collect::<Vec<_>>(),
        "ranked_valid_combinations": valid_results.iter().take(10).map(|r| serde_json::json!({
            "params": r.params,
            "composite_score": r.composite_score.to_string(),
        })).collect::<Vec<_>>(),
    });
    
    let output_file = "logs/optimization_results.json";
    let mut file = File::create(output_file)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&output)?)?;
    
    println!("📁 Results saved to: {}", output_file);
    println!("\n✅ Optimization complete!");
    
    Ok(())
}
