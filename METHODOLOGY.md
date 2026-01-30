# 开发方法论 (强制执行)

## 核心原则

**每个功能必须经过：Plan → Test → Code → Review**

---

## 1. 任务拆分 (2-5 分钟粒度)

新功能开发前，必须创建 `tasks/FEATURE_NAME.md`：

```markdown
# Feature: [名称]

## 目标
一句话描述

## 任务拆分
- [ ] Task 1: [描述] (~2min)
- [ ] Task 2: [描述] (~3min)
- [ ] Task 3: [描述] (~5min)

## 测试用例 (先写)
- test_xxx_yyy: 验证...
- test_xxx_zzz: 验证...

## 验收标准
- [ ] 所有测试通过
- [ ] cargo clippy 无警告
- [ ] 文档更新
```

---

## 2. 严格 TDD (Red-Green-Refactor)

### Step 1: Red (写失败的测试)
```bash
# 先写测试
cargo test feature_name:: -- --nocapture 2>&1 | head -20
# 必须看到 FAILED
```

### Step 2: Green (最小实现)
```bash
# 写最少代码让测试通过
cargo test feature_name::
# 必须看到 ok
```

### Step 3: Refactor (优化)
```bash
# 重构，测试仍需通过
cargo test feature_name::
cargo clippy -- -D warnings
```

---

## 3. Commit 规范

**格式**: `type: description (task X/N)`

**Types**:
- `test:` 添加测试 (TDD Red)
- `feat:` 实现功能 (TDD Green)
- `refactor:` 重构 (TDD Refactor)
- `fix:` 修复
- `docs:` 文档

**示例**:
```
test: add paper_cli position tests (task 1/4)
feat: implement paper_cli position command (task 1/4)
refactor: simplify position display logic (task 1/4)
```

---

## 4. Pre-commit 检查清单

每次 commit 前自问：
- [ ] 测试先写了吗？
- [ ] 测试先失败了吗？
- [ ] 现在测试通过了吗？
- [ ] clippy 无警告？
- [ ] commit message 符合规范？

---

## 5. Review 标准

功能完成后检查：
- [ ] 覆盖率：新代码有对应测试
- [ ] 边界：测试了边界情况
- [ ] 错误：测试了错误路径
- [ ] 性能：无明显性能问题
- [ ] 文档：README/注释更新

---

## 违规处理

如果跳过步骤：
1. `git reset --soft HEAD~1` 撤销 commit
2. 补齐缺失的步骤
3. 重新提交

**没有例外。**
