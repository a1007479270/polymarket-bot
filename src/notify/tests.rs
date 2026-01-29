//! Tests for notify module

#[cfg(test)]
mod tests {
    use super::super::Notifier;
    use crate::types::{Signal, Side, Trade};
    use crate::monitor::PerformanceStats;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_notifier_creation() {
        let notifier = Notifier::new("token123".to_string(), "chat456".to_string());
        // Just verify it creates
        let _ = notifier;
    }

    #[test]
    fn test_notifier_disabled() {
        let notifier = Notifier::disabled();
        // Should create a disabled notifier
        let _ = notifier;
    }

    #[test]
    fn test_notifier_clone() {
        let notifier = Notifier::new("token".to_string(), "chat".to_string());
        let cloned = notifier.clone();
        let _ = cloned;
    }

    #[test]
    fn test_signal_for_notification() {
        let signal = Signal {
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.70),
            market_probability: dec!(0.55),
            edge: dec!(0.15),
            confidence: dec!(0.80),
            suggested_size: dec!(100),
            timestamp: Utc::now(),
        };
        
        assert_eq!(signal.side, Side::Buy);
        assert_eq!(signal.edge, dec!(0.15));
    }

    #[test]
    fn test_sell_signal_for_notification() {
        let signal = Signal {
            market_id: "m2".to_string(),
            token_id: "t2".to_string(),
            side: Side::Sell,
            model_probability: dec!(0.30),
            market_probability: dec!(0.45),
            edge: dec!(-0.15),
            confidence: dec!(0.75),
            suggested_size: dec!(50),
            timestamp: Utc::now(),
        };
        
        assert_eq!(signal.side, Side::Sell);
    }

    #[test]
    fn test_trade_for_notification() {
        let trade = Trade {
            id: "trade1".to_string(),
            order_id: "order1".to_string(),
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100),
            fee: dec!(0.50),
            timestamp: Utc::now(),
        };
        
        assert_eq!(trade.price, dec!(0.55));
    }

    #[test]
    fn test_performance_stats_for_notification() {
        let stats = PerformanceStats {
            total_trades: 100,
            winning_trades: 65,
            losing_trades: 35,
            win_rate: dec!(0.65),
            total_pnl: dec!(5000),
            avg_pnl_per_trade: dec!(50),
            sharpe_ratio: Some(dec!(1.8)),
        };
        
        assert_eq!(stats.total_trades, 100);
        assert_eq!(stats.win_rate, dec!(0.65));
    }

    #[test]
    fn test_format_decimal_percentage() {
        let value = dec!(0.65);
        let formatted = format!("{:.1}%", value * dec!(100));
        assert_eq!(formatted, "65.0%");
    }

    #[test]
    fn test_format_pnl_positive() {
        let pnl = dec!(1234.56);
        let formatted = format!("${:.2}", pnl);
        assert_eq!(formatted, "$1234.56");
    }

    #[test]
    fn test_format_pnl_negative() {
        let pnl = dec!(-567.89);
        let formatted = format!("${:.2}", pnl);
        assert_eq!(formatted, "$-567.89");
    }

    #[test]
    fn test_side_emoji() {
        let buy_emoji = match Side::Buy {
            Side::Buy => "📈",
            Side::Sell => "📉",
        };
        assert_eq!(buy_emoji, "📈");
        
        let sell_emoji = match Side::Sell {
            Side::Buy => "📈",
            Side::Sell => "📉",
        };
        assert_eq!(sell_emoji, "📉");
    }

    #[test]
    fn test_edge_formatting() {
        let edge = dec!(0.08);
        let formatted = format!("{:.1}%", edge * dec!(100));
        assert_eq!(formatted, "8.0%");
    }

    #[test]
    fn test_high_confidence_signal() {
        let signal = Signal {
            market_id: "m".to_string(),
            token_id: "t".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.85),
            market_probability: dec!(0.70),
            edge: dec!(0.15),
            confidence: dec!(0.90),
            suggested_size: dec!(200),
            timestamp: Utc::now(),
        };
        
        assert!(signal.confidence >= dec!(0.90));
    }

    #[test]
    fn test_low_confidence_signal() {
        let signal = Signal {
            market_id: "m".to_string(),
            token_id: "t".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.55),
            market_probability: dec!(0.50),
            edge: dec!(0.05),
            confidence: dec!(0.50),
            suggested_size: dec!(20),
            timestamp: Utc::now(),
        };
        
        assert!(signal.confidence <= dec!(0.50));
    }

    #[tokio::test]
    async fn test_disabled_notifier_send() {
        let notifier = Notifier::disabled();
        // Should succeed even though disabled
        let result = notifier.send("test message").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_disabled_notifier_send_raw() {
        let notifier = Notifier::disabled();
        let result = notifier.send_raw("test message").await;
        assert!(result.is_ok());
    }
}
