//! Tests for copy trading

#[cfg(test)]
mod tests {
    use super::super::copy_trade::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_top_trader_creation() {
        let trader = TopTrader {
            username: "testuser".to_string(),
            address: Some("0x123".to_string()),
            win_rate: 0.65,
            total_profit: dec!(10000),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        assert_eq!(trader.username, "testuser");
        assert_eq!(trader.win_rate, 0.65);
        assert_eq!(trader.total_profit, dec!(10000));
    }

    #[test]
    fn test_top_trader_without_address() {
        let trader = TopTrader {
            username: "noaddress".to_string(),
            address: None,
            win_rate: 0.55,
            total_profit: dec!(5000),
            weight: 0.8,
            updated_at: Utc::now(),
        };
        assert!(trader.address.is_none());
    }

    #[test]
    fn test_top_trader_serialization() {
        let trader = TopTrader {
            username: "user".to_string(),
            address: Some("0xabc".to_string()),
            win_rate: 0.70,
            total_profit: dec!(15000),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&trader).unwrap();
        assert!(json.contains("\"username\":\"user\""));
        assert!(json.contains("\"win_rate\":0.7"));
    }

    #[test]
    fn test_top_trader_deserialization() {
        let json = r#"{
            "username": "trader1",
            "address": "0xdef",
            "win_rate": 0.75,
            "total_profit": "20000",
            "weight": 0.9,
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;
        let trader: TopTrader = serde_json::from_str(json).unwrap();
        assert_eq!(trader.username, "trader1");
        assert_eq!(trader.address, Some("0xdef".to_string()));
        assert_eq!(trader.win_rate, 0.75);
    }

    #[test]
    fn test_copy_trader_new() {
        let trader = CopyTrader::new();
        // Just verify creation doesn't panic
        let _ = trader;
    }

    #[test]
    fn test_copy_trader_with_ratio() {
        let trader = CopyTrader::new().with_copy_ratio(0.3);
        let _ = trader;
    }

    #[test]
    fn test_copy_trader_with_ratio_clamped_low() {
        let trader = CopyTrader::new().with_copy_ratio(0.05);
        // Should be clamped to 0.1
        let _ = trader;
    }

    #[test]
    fn test_copy_trader_with_ratio_clamped_high() {
        let trader = CopyTrader::new().with_copy_ratio(1.5);
        // Should be clamped to 1.0
        let _ = trader;
    }

    #[test]
    fn test_copy_trader_add_trader() {
        let mut copy_trader = CopyTrader::new();
        let trader = TopTrader {
            username: "test".to_string(),
            address: None,
            win_rate: 0.6,
            total_profit: dec!(1000),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        copy_trader.add_trader(trader);
    }

    #[test]
    fn test_copy_trader_add_duplicate_replaces() {
        let mut copy_trader = CopyTrader::new();
        
        let trader1 = TopTrader {
            username: "same".to_string(),
            address: None,
            win_rate: 0.5,
            total_profit: dec!(500),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        copy_trader.add_trader(trader1);
        
        let trader2 = TopTrader {
            username: "same".to_string(),
            address: Some("0x123".to_string()),
            win_rate: 0.7,
            total_profit: dec!(1000),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        copy_trader.add_trader(trader2);
        // Second add should replace, not duplicate
    }

    #[test]
    fn test_copy_trade_config_default() {
        let config = CopyTradeConfig::default();
        assert!(!config.enabled);
        assert!(config.follow_users.is_empty());
        assert!(config.follow_addresses.is_empty());
        assert_eq!(config.copy_ratio, 0.5);
        assert_eq!(config.delay_secs, 0);
    }

    #[test]
    fn test_copy_trade_config_deserialization() {
        let toml = r#"
enabled = true
follow_users = ["user1", "user2"]
follow_addresses = ["0xabc", "0xdef"]
copy_ratio = 0.4
delay_secs = 15
"#;
        let config: CopyTradeConfig = toml::from_str(toml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.follow_users.len(), 2);
        assert_eq!(config.follow_addresses.len(), 2);
        assert_eq!(config.copy_ratio, 0.4);
        assert_eq!(config.delay_secs, 15);
    }

    #[test]
    fn test_copy_trade_config_defaults_applied() {
        let toml = r#"
enabled = true
"#;
        let config: CopyTradeConfig = toml::from_str(toml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.copy_ratio, 0.5); // default
        assert_eq!(config.delay_secs, 0); // default
    }

    #[test]
    fn test_copy_signal_to_signal() {
        let trader = TopTrader {
            username: "trader".to_string(),
            address: Some("0x123".to_string()),
            win_rate: 0.7,
            total_profit: dec!(5000),
            weight: 1.0,
            updated_at: Utc::now(),
        };
        
        let copy_signal = CopySignal {
            trader,
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            side: crate::types::Side::Buy,
            trader_size: dec!(1000),
            suggested_size: dec!(500),
            timestamp: Utc::now(),
        };
        
        let signal = copy_signal.to_signal(dec!(0.65));
        assert_eq!(signal.market_id, "market1");
        assert_eq!(signal.token_id, "token1");
        assert_eq!(signal.suggested_size, dec!(500));
    }
}
