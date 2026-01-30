//! LLM-powered trading decisions for paper trading
//!
//! Uses DeepSeek API to make dynamic buy/sell/hold decisions

use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

/// Trading decision from LLM
#[derive(Debug, Clone, PartialEq)]
pub enum TradeDecision {
    Hold,
    Sell { position_id: String, reason: String },
    SellAll { reason: String },
    Buy { asset: String, side: String, amount: Decimal, reason: String },
}

/// Position info for LLM context
#[derive(Debug, Serialize)]
pub struct PositionContext {
    pub asset: String,
    pub side: String,  // "UP" or "DOWN"
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl_pct: f64,
    pub shares: f64,
    pub cost_basis: f64,
    pub current_value: f64,
}

/// Market context for LLM
#[derive(Debug, Serialize)]
pub struct MarketContext {
    pub asset: String,
    pub binance_price: f64,
    pub binance_24h_change: f64,
    pub poly_up_price: f64,
    pub poly_down_price: f64,
    pub window_seconds_remaining: u64,
    pub trend_direction: String,
}

/// LLM Trading Advisor
pub struct LlmTrader {
    http: Client,
    api_key: String,
    model: String,
}

impl LlmTrader {
    pub fn new(api_key: String) -> Self {
        Self {
            http: Client::new(),
            api_key,
            model: "deepseek-chat".to_string(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Get trading decision from LLM
    pub async fn decide(
        &self,
        positions: &[PositionContext],
        markets: &[MarketContext],
        cash_balance: f64,
    ) -> Result<Vec<TradeDecision>, String> {
        let prompt = self.build_prompt(positions, markets, cash_balance);
        
        let response = self.call_deepseek(&prompt).await?;
        
        self.parse_decisions(&response)
    }

    fn build_prompt(
        &self,
        positions: &[PositionContext],
        markets: &[MarketContext],
        cash_balance: f64,
    ) -> String {
        let mut prompt = String::from(
r#"你是一个专业的加密货币预测市场交易员。根据以下数据做出交易决策。

## 当前持仓
"#);

        if positions.is_empty() {
            prompt.push_str("无持仓\n");
        } else {
            for pos in positions {
                prompt.push_str(&format!(
                    "- {asset} {side}: 买入价 {entry:.1}%, 现价 {curr:.1}%, 浮盈 {pnl:+.1}%, 价值 ${value:.2}\n",
                    asset = pos.asset,
                    side = pos.side,
                    entry = pos.entry_price * 100.0,
                    curr = pos.current_price * 100.0,
                    pnl = pos.unrealized_pnl_pct,
                    value = pos.current_value,
                ));
            }
        }

        prompt.push_str(&format!("\n现金余额: ${:.2}\n\n## 市场数据\n", cash_balance));

        for mkt in markets {
            prompt.push_str(&format!(
                "- {asset}: Binance ${price:.0} ({change:+.2}% 24h), Poly UP {up:.1}% / DOWN {down:.1}%, 剩余 {secs}s, 趋势 {trend}\n",
                asset = mkt.asset,
                price = mkt.binance_price,
                change = mkt.binance_24h_change,
                up = mkt.poly_up_price * 100.0,
                down = mkt.poly_down_price * 100.0,
                secs = mkt.window_seconds_remaining,
                trend = mkt.trend_direction,
            ));
        }

        prompt.push_str(
r#"
## 决策要求
1. 考虑止盈止损：浮盈 >15% 可考虑止盈，浮亏 >20% 考虑止损
2. 考虑时间：窗口剩余时间越少，越要谨慎
3. 考虑趋势：Binance 趋势是否支持当前持仓
4. 做T机会：如果有波动，可以先卖再买回

## 输出格式 (JSON)
{
  "decisions": [
    {"action": "HOLD"},
    {"action": "SELL", "asset": "BTC", "reason": "止盈+15%"},
    {"action": "BUY", "asset": "ETH", "side": "DOWN", "amount": 50, "reason": "趋势向下"}
  ],
  "reasoning": "简短解释"
}

只输出 JSON，不要其他文字。
"#);

        prompt
    }

    async fn call_deepseek(&self, prompt: &str) -> Result<String, String> {
        let request = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a crypto trading decision system. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        });

        let resp = self.http
            .post("https://api.deepseek.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("API error {}: {}", status, body));
        }

        let json: serde_json::Value = resp.json().await
            .map_err(|e| format!("JSON parse error: {}", e))?;

        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "No content in response".to_string())
    }

    fn parse_decisions(&self, response: &str) -> Result<Vec<TradeDecision>, String> {
        // Try to extract JSON from response
        let json_str = if response.starts_with('{') {
            response.to_string()
        } else if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                response[start..=end].to_string()
            } else {
                return Err("Invalid JSON in response".to_string());
            }
        } else {
            return Err("No JSON found in response".to_string());
        };

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| format!("JSON parse error: {}", e))?;

        let mut decisions = Vec::new();

        if let Some(arr) = parsed["decisions"].as_array() {
            for item in arr {
                let action = item["action"].as_str().unwrap_or("HOLD");
                
                match action.to_uppercase().as_str() {
                    "HOLD" => decisions.push(TradeDecision::Hold),
                    "SELL" => {
                        let asset = item["asset"].as_str().unwrap_or("").to_string();
                        let reason = item["reason"].as_str().unwrap_or("LLM decision").to_string();
                        decisions.push(TradeDecision::Sell {
                            position_id: asset,
                            reason,
                        });
                    }
                    "SELL_ALL" => {
                        let reason = item["reason"].as_str().unwrap_or("LLM decision").to_string();
                        decisions.push(TradeDecision::SellAll { reason });
                    }
                    "BUY" => {
                        let asset = item["asset"].as_str().unwrap_or("").to_string();
                        let side = item["side"].as_str().unwrap_or("UP").to_string();
                        let amount = item["amount"].as_f64().unwrap_or(50.0);
                        let reason = item["reason"].as_str().unwrap_or("LLM decision").to_string();
                        decisions.push(TradeDecision::Buy {
                            asset,
                            side,
                            amount: Decimal::from_f64_retain(amount).unwrap_or(Decimal::ZERO),
                            reason,
                        });
                    }
                    _ => {}
                }
            }
        }

        if let Some(reasoning) = parsed["reasoning"].as_str() {
            info!("🤖 LLM reasoning: {}", reasoning);
        }

        Ok(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_decisions() {
        let trader = LlmTrader::new("test".to_string());
        
        let response = r#"{
            "decisions": [
                {"action": "SELL", "asset": "BTC", "reason": "止盈"},
                {"action": "HOLD"}
            ],
            "reasoning": "BTC有利润，锁定"
        }"#;
        
        let decisions = trader.parse_decisions(response).unwrap();
        assert_eq!(decisions.len(), 2);
    }
}
