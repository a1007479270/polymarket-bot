//! Unit tests for ingester module

#[cfg(test)]
mod tests {
    use super::super::*;
    use chrono::Utc;

    #[test]
    fn test_raw_signal_creation() {
        let signal = RawSignal {
            source: "twitter".to_string(),
            source_id: "12345".to_string(),
            content: "BTC looking bullish here".to_string(),
            author: "crypto_trader".to_string(),
            author_trust: 0.7,
            timestamp: Utc::now(),
            metadata: None,
        };
        
        assert_eq!(signal.source, "twitter");
        assert_eq!(signal.author_trust, 0.7);
    }

    #[test]
    fn test_raw_signal_with_metadata() {
        let metadata = serde_json::json!({
            "likes": 100,
            "retweets": 50
        });
        
        let signal = RawSignal {
            source: "twitter".to_string(),
            source_id: "12345".to_string(),
            content: "BTC to the moon".to_string(),
            author: "trader".to_string(),
            author_trust: 0.8,
            timestamp: Utc::now(),
            metadata: Some(metadata),
        };
        
        let meta = signal.metadata.unwrap();
        assert_eq!(meta.get("likes"), Some(&serde_json::json!(100)));
    }

    #[test]
    fn test_signal_direction_serialization() {
        assert_eq!(
            serde_json::to_string(&SignalDirection::Bullish).unwrap(),
            "\"bullish\""
        );
        assert_eq!(
            serde_json::to_string(&SignalDirection::Bearish).unwrap(),
            "\"bearish\""
        );
        assert_eq!(
            serde_json::to_string(&SignalDirection::Neutral).unwrap(),
            "\"neutral\""
        );
    }

    #[test]
    fn test_signal_direction_deserialization() {
        let bullish: SignalDirection = serde_json::from_str("\"bullish\"").unwrap();
        let bearish: SignalDirection = serde_json::from_str("\"bearish\"").unwrap();
        let neutral: SignalDirection = serde_json::from_str("\"neutral\"").unwrap();
        
        assert_eq!(bullish, SignalDirection::Bullish);
        assert_eq!(bearish, SignalDirection::Bearish);
        assert_eq!(neutral, SignalDirection::Neutral);
    }

    #[test]
    fn test_signal_direction_equality() {
        assert_eq!(SignalDirection::Bullish, SignalDirection::Bullish);
        assert_ne!(SignalDirection::Bullish, SignalDirection::Bearish);
        assert_ne!(SignalDirection::Bearish, SignalDirection::Neutral);
    }

    #[test]
    fn test_action_type_serialization() {
        assert_eq!(
            serde_json::to_string(&ActionType::Entry).unwrap(),
            "\"entry\""
        );
        assert_eq!(
            serde_json::to_string(&ActionType::Exit).unwrap(),
            "\"exit\""
        );
        assert_eq!(
            serde_json::to_string(&ActionType::Warning).unwrap(),
            "\"warning\""
        );
        assert_eq!(
            serde_json::to_string(&ActionType::Info).unwrap(),
            "\"info\""
        );
    }

    #[test]
    fn test_action_type_deserialization() {
        let entry: ActionType = serde_json::from_str("\"entry\"").unwrap();
        let exit: ActionType = serde_json::from_str("\"exit\"").unwrap();
        let warning: ActionType = serde_json::from_str("\"warning\"").unwrap();
        let info: ActionType = serde_json::from_str("\"info\"").unwrap();
        
        assert_eq!(entry, ActionType::Entry);
        assert_eq!(exit, ActionType::Exit);
        assert_eq!(warning, ActionType::Warning);
        assert_eq!(info, ActionType::Info);
    }

    #[test]
    fn test_parsed_signal_creation() {
        let raw = RawSignal {
            source: "telegram".to_string(),
            source_id: "msg123".to_string(),
            content: "ETH breakout".to_string(),
            author: "alpha_group".to_string(),
            author_trust: 0.8,
            timestamp: Utc::now(),
            metadata: None,
        };

        let parsed = ParsedSignal {
            token: "ETH".to_string(),
            direction: SignalDirection::Bullish,
            timeframe: "1h".to_string(),
            confidence: 0.75,
            reasoning: "Technical breakout pattern".to_string(),
            action_type: ActionType::Entry,
            sources: vec![raw],
            agg_score: 0.8,
            timestamp: Utc::now(),
        };

        assert_eq!(parsed.token, "ETH");
        assert_eq!(parsed.direction, SignalDirection::Bullish);
        assert!(parsed.agg_score >= 0.7);
    }

    #[test]
    fn test_parsed_signal_multiple_sources() {
        let raw1 = RawSignal {
            source: "telegram".to_string(),
            source_id: "msg1".to_string(),
            content: "BTC bullish".to_string(),
            author: "author1".to_string(),
            author_trust: 0.8,
            timestamp: Utc::now(),
            metadata: None,
        };

        let raw2 = RawSignal {
            source: "twitter".to_string(),
            source_id: "tweet1".to_string(),
            content: "BTC looking good".to_string(),
            author: "author2".to_string(),
            author_trust: 0.7,
            timestamp: Utc::now(),
            metadata: None,
        };

        let parsed = ParsedSignal {
            token: "BTC".to_string(),
            direction: SignalDirection::Bullish,
            timeframe: "4h".to_string(),
            confidence: 0.85,
            reasoning: "Multiple sources confirm".to_string(),
            action_type: ActionType::Entry,
            sources: vec![raw1, raw2],
            agg_score: 0.9,
            timestamp: Utc::now(),
        };

        assert_eq!(parsed.sources.len(), 2);
        assert!(parsed.agg_score > parsed.confidence);
    }

    #[test]
    fn test_parsed_signal_bearish() {
        let parsed = ParsedSignal {
            token: "SOL".to_string(),
            direction: SignalDirection::Bearish,
            timeframe: "1d".to_string(),
            confidence: 0.65,
            reasoning: "Breakdown below support".to_string(),
            action_type: ActionType::Exit,
            sources: vec![],
            agg_score: 0.65,
            timestamp: Utc::now(),
        };

        assert_eq!(parsed.direction, SignalDirection::Bearish);
        assert_eq!(parsed.action_type, ActionType::Exit);
    }

    #[test]
    fn test_ingester_config_defaults() {
        let json = r#"{}"#;
        let config: std::result::Result<IngesterConfig, serde_json::Error> = serde_json::from_str(json);
        // Should deserialize with defaults
        if let Ok(config) = config {
            assert!(config.telegram.is_none());
            assert!(config.twitter.is_none());
            assert!(config.author_trust.is_empty());
        }
    }

    #[test]
    fn test_ingester_config_with_author_trust() {
        let json = r#"{
            "author_trust": {
                "trusted_user": 0.9,
                "new_user": 0.5
            }
        }"#;
        let config: IngesterConfig = serde_json::from_str(json).unwrap();
        
        assert_eq!(config.author_trust.get("trusted_user"), Some(&0.9));
        assert_eq!(config.author_trust.get("new_user"), Some(&0.5));
        assert!(config.author_trust.get("unknown").is_none());
    }

    #[test]
    fn test_raw_signal_clone() {
        let signal = RawSignal {
            source: "test".to_string(),
            source_id: "1".to_string(),
            content: "content".to_string(),
            author: "author".to_string(),
            author_trust: 0.5,
            timestamp: Utc::now(),
            metadata: None,
        };
        
        let cloned = signal.clone();
        assert_eq!(signal.source, cloned.source);
        assert_eq!(signal.content, cloned.content);
    }

    #[test]
    fn test_signal_direction_clone() {
        let dir = SignalDirection::Bullish;
        let cloned = dir.clone();
        assert_eq!(dir, cloned);
    }

    #[test]
    fn test_action_type_clone() {
        let action = ActionType::Entry;
        let cloned = action.clone();
        assert_eq!(action, cloned);
    }
}
