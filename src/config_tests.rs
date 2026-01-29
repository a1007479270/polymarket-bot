//! Tests for configuration

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_strategy_config_default() {
        let config = StrategyConfig::default();
        assert_eq!(config.min_edge, dec!(0.06));
        assert_eq!(config.min_confidence, dec!(0.60));
        assert_eq!(config.kelly_fraction, dec!(0.35));
        assert_eq!(config.scan_interval_secs, 180);
        assert_eq!(config.model_update_interval_secs, 900);
        assert!(config.compound_enabled);
        assert!(config.compound_sqrt_scaling);
    }

    #[test]
    fn test_risk_config_default() {
        let config = RiskConfig::default();
        assert_eq!(config.max_position_pct, dec!(0.05));
        assert_eq!(config.max_exposure_pct, dec!(0.50));
        assert_eq!(config.max_daily_loss_pct, dec!(0.10));
        assert_eq!(config.min_balance_reserve, dec!(100));
        assert_eq!(config.max_open_positions, 10);
    }

    #[test]
    fn test_processing_config_defaults() {
        let config: ProcessingConfig = toml::from_str("").unwrap();
        assert_eq!(config.aggregation_window_secs, 300);
        assert_eq!(config.min_confidence, 0.5);
        assert_eq!(config.min_agg_score, 0.6);
    }

    #[test]
    fn test_copy_trade_config_defaults() {
        let toml_str = r#"
enabled = true
"#;
        let config: CopyTradeConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert!(config.follow_users.is_empty());
        assert!(config.follow_addresses.is_empty());
        assert_eq!(config.copy_ratio, 0.5);
        assert_eq!(config.delay_secs, 0);
    }

    #[test]
    fn test_copy_trade_config_with_users() {
        let toml_str = r#"
enabled = true
follow_users = ["user1", "user2"]
follow_addresses = ["0xabc", "0xdef"]
copy_ratio = 0.3
delay_secs = 30
"#;
        let config: CopyTradeConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.follow_users.len(), 2);
        assert_eq!(config.follow_users[0], "user1");
        assert_eq!(config.follow_addresses.len(), 2);
        assert_eq!(config.copy_ratio, 0.3);
        assert_eq!(config.delay_secs, 30);
    }

    #[test]
    fn test_telegram_config_defaults() {
        let toml_str = r#"
bot_token = "123:abc"
chat_id = "12345"
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.bot_token, "123:abc");
        assert_eq!(config.chat_id, "12345");
        assert!(config.notify_signals);
        assert!(config.notify_trades);
        assert!(config.notify_errors);
        assert!(config.notify_daily);
    }

    #[test]
    fn test_telegram_config_disabled_notifications() {
        let toml_str = r#"
bot_token = "123:abc"
chat_id = "12345"
notify_signals = false
notify_trades = false
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.notify_signals);
        assert!(!config.notify_trades);
        assert!(config.notify_errors); // defaults to true
    }

    #[test]
    fn test_polymarket_config() {
        let toml_str = r#"
clob_url = "https://clob.polymarket.com"
gamma_url = "https://gamma-api.polymarket.com"
private_key = "abc123"
chain_id = 137
signature_type = 0
"#;
        let config: PolymarketConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.clob_url, "https://clob.polymarket.com");
        assert_eq!(config.gamma_url, "https://gamma-api.polymarket.com");
        assert_eq!(config.private_key, "abc123");
        assert_eq!(config.chain_id, 137);
        assert_eq!(config.signature_type, 0);
        assert!(config.funder_address.is_none());
    }

    #[test]
    fn test_polymarket_config_with_funder() {
        let toml_str = r#"
clob_url = "https://clob.polymarket.com"
gamma_url = "https://gamma-api.polymarket.com"
private_key = "abc123"
funder_address = "0x123456"
chain_id = 137
signature_type = 1
"#;
        let config: PolymarketConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.funder_address, Some("0x123456".to_string()));
        assert_eq!(config.signature_type, 1);
    }

    #[test]
    fn test_llm_config_minimal() {
        let toml_str = r#"
provider = "deepseek"
api_key = "sk-xxx"
"#;
        let config: LlmConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.provider, "deepseek");
        assert_eq!(config.api_key, "sk-xxx");
        assert!(config.model.is_none());
        assert!(config.base_url.is_none());
    }

    #[test]
    fn test_llm_config_with_model() {
        let toml_str = r#"
provider = "openai"
api_key = "sk-xxx"
model = "gpt-4"
base_url = "https://api.openai.com/v1"
"#;
        let config: LlmConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, Some("gpt-4".to_string()));
        assert_eq!(config.base_url, Some("https://api.openai.com/v1".to_string()));
    }

    #[test]
    fn test_llm_config_ollama() {
        let toml_str = r#"
provider = "ollama"
"#;
        let config: LlmConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.api_key, ""); // defaults to empty
    }

    #[test]
    fn test_database_config() {
        let toml_str = r#"
path = "data/bot.db"
"#;
        let config: DatabaseConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.path, "data/bot.db");
    }

    #[test]
    fn test_strategy_config_deserialize() {
        let toml_str = r#"
min_edge = 0.08
min_confidence = 0.65
kelly_fraction = 0.25
scan_interval_secs = 300
model_update_interval_secs = 1800
compound_enabled = false
compound_sqrt_scaling = false
"#;
        let config: StrategyConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.min_edge, dec!(0.08));
        assert_eq!(config.min_confidence, dec!(0.65));
        assert_eq!(config.kelly_fraction, dec!(0.25));
        assert_eq!(config.scan_interval_secs, 300);
        assert!(!config.compound_enabled);
        assert!(!config.compound_sqrt_scaling);
    }

    #[test]
    fn test_risk_config_deserialize() {
        let toml_str = r#"
max_position_pct = 0.10
max_exposure_pct = 0.60
max_daily_loss_pct = 0.15
min_balance_reserve = 50
max_open_positions = 20
"#;
        let config: RiskConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_position_pct, dec!(0.10));
        assert_eq!(config.max_exposure_pct, dec!(0.60));
        assert_eq!(config.max_daily_loss_pct, dec!(0.15));
        assert_eq!(config.min_balance_reserve, dec!(50));
        assert_eq!(config.max_open_positions, 20);
    }

    #[test]
    fn test_ingester_config_minimal() {
        let toml_str = r#"
enabled = true
"#;
        let config: IngesterConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert!(config.telegram_userbot.is_none());
        assert!(config.telegram_bot.is_none());
        assert!(config.twitter.is_none());
    }

    #[test]
    fn test_ingester_config_with_twitter() {
        let toml_str = r#"
enabled = true

[twitter]
bearer_token = "bearer123"
user_ids = ["12345", "67890"]
keywords = ["polymarket", "prediction"]
"#;
        let config: IngesterConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        let twitter = config.twitter.unwrap();
        assert_eq!(twitter.bearer_token, Some("bearer123".to_string()));
        assert_eq!(twitter.user_ids.len(), 2);
        assert_eq!(twitter.keywords.len(), 2);
    }

    #[test]
    fn test_telegram_userbot_config() {
        let toml_str = r#"
api_id = 12345
api_hash = "abc123hash"
session_file = "session.dat"
watch_chats = [123, 456, 789]
"#;
        let config: TelegramUserbotConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.api_id, 12345);
        assert_eq!(config.api_hash, "abc123hash");
        assert_eq!(config.session_file, "session.dat");
        assert_eq!(config.watch_chats.len(), 3);
    }

    #[test]
    fn test_telegram_bot_ingester_config() {
        let toml_str = r#"
bot_token = "123:abc"
channels = ["@channel1", "@channel2"]
"#;
        let config: TelegramBotIngesterConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.bot_token, "123:abc");
        assert_eq!(config.channels.len(), 2);
    }
}
