//! Paper Trading CLI
//!
//! Commands:
//! - status: Show account summary
//! - buy: Simulate buying shares
//! - sell: Simulate selling a position
//! - positions: List open positions
//! - history: Show trade history

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "paper")]
#[command(about = "Paper trading CLI for Polymarket simulation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show account status and summary
    Status,
    /// Simulate buying shares in a market
    Buy {
        /// Market ID or slug
        #[arg(short, long)]
        market: String,
        /// Side: yes or no
        #[arg(short, long)]
        side: String,
        /// Amount in USD
        #[arg(short, long)]
        amount: f64,
    },
    /// Sell/close a position
    Sell {
        /// Position ID
        #[arg(short, long)]
        position: String,
    },
    /// List open positions
    Positions,
    /// Show trade history
    History {
        /// Number of records to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Status => {
            println!("Status command - not implemented");
        }
        Commands::Buy { market, side, amount } => {
            println!("Buy {} {} ${}", side, market, amount);
        }
        Commands::Sell { position } => {
            println!("Sell position {}", position);
        }
        Commands::Positions => {
            println!("Positions - not implemented");
        }
        Commands::History { limit } => {
            println!("History (last {}) - not implemented", limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_help() {
        // Verify CLI parses correctly
        Cli::command().debug_assert();
    }

    #[test]
    fn test_cli_status_parses() {
        let cli = Cli::parse_from(["paper", "status"]);
        assert!(matches!(cli.command, Commands::Status));
    }

    #[test]
    fn test_cli_buy_parses() {
        let cli = Cli::parse_from([
            "paper", "buy", 
            "--market", "btc-100k",
            "--side", "yes",
            "--amount", "50"
        ]);
        if let Commands::Buy { market, side, amount } = cli.command {
            assert_eq!(market, "btc-100k");
            assert_eq!(side, "yes");
            assert!((amount - 50.0).abs() < 0.01);
        } else {
            panic!("Expected Buy command");
        }
    }

    #[test]
    fn test_cli_sell_parses() {
        let cli = Cli::parse_from([
            "paper", "sell",
            "--position", "pos123"
        ]);
        if let Commands::Sell { position } = cli.command {
            assert_eq!(position, "pos123");
        } else {
            panic!("Expected Sell command");
        }
    }

    #[test]
    fn test_cli_positions_parses() {
        let cli = Cli::parse_from(["paper", "positions"]);
        assert!(matches!(cli.command, Commands::Positions));
    }

    #[test]
    fn test_cli_history_parses() {
        let cli = Cli::parse_from(["paper", "history", "--limit", "20"]);
        if let Commands::History { limit } = cli.command {
            assert_eq!(limit, 20);
        } else {
            panic!("Expected History command");
        }
    }

    #[test]
    fn test_cli_history_default_limit() {
        let cli = Cli::parse_from(["paper", "history"]);
        if let Commands::History { limit } = cli.command {
            assert_eq!(limit, 10);
        } else {
            panic!("Expected History command");
        }
    }
}
