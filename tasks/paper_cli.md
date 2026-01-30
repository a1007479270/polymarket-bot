# Feature: Paper Trading CLI

## 目标
命令行工具运行模拟交易，查看持仓和收益

## 任务拆分
- [ ] Task 1: CLI 框架 + help 命令 (~3min)
- [ ] Task 2: `status` 命令显示账户摘要 (~3min)
- [ ] Task 3: `buy` 命令模拟买入 (~4min)
- [ ] Task 4: `sell` 命令模拟卖出 (~3min)
- [ ] Task 5: `positions` 命令显示持仓 (~2min)
- [ ] Task 6: `history` 命令显示交易记录 (~2min)

## 测试用例 (先写)
- test_cli_help: 验证 help 输出
- test_cli_status_empty: 空账户状态
- test_cli_buy_success: 成功买入
- test_cli_buy_insufficient: 余额不足
- test_cli_sell_success: 成功卖出
- test_cli_positions_empty: 无持仓
- test_cli_positions_with_data: 有持仓
- test_cli_history_empty: 无记录
- test_cli_history_with_trades: 有记录

## 验收标准
- [ ] 9 个测试全部通过
- [ ] cargo clippy 无警告
- [ ] README 更新用法

## 进度
Started: 2026-01-30 05:56 UTC
