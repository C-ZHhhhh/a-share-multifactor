# v6 验收清单

分支 `fix/v6-acceptance`，基于 `18b3528` / `00efe1b`。状态只分三种：**已修复**（有可重复的测试或自动核对且通过）、**部分修复**（工程部分已修，剩余依赖外部数据或未执行的步骤）、**未解决**（需要外部数据，已在报告中标为未通过，没有编造数据）。

测试：`pytest` 共 46 个（核心单测 + 合成端到端 + 数据刷新 + 前瞻时间戳 + 账户 / 基准 / 回撤边界）。Python 3.9 本地通过；CI（3.9 / 3.11）在推送后运行。每次 `run_all.sh` 末尾由 `scripts/verify_report.py` 从 HTML 内嵌的日收益独立复算历史验证期的超额、IR、跟踪误差、年化收益和最大回撤，并核对持仓 CSV（本次通过）。

绩效差异（历史验证期 2022-01 ~ 2026-10，详见 [revision_v6.md](revision_v6.md)）：v5 超额 +1.45% / IR 0.25 → 只换 v6 代码 +1.61% / 0.28 → v6 按同一规则重新校准 λ（5 → 7）+1.66% / 0.32。没有为提高 IR 改策略参数；λ 的变化来自同一规则下的近乎打平（距离 4% 分别为 0.0021 与 0.0019）。

## P0：前瞻验证的前提

| 编号 | 修改文件 | 测试 / 核对 | 结果 | 对历史绩效的影响 | 未解决的限制 |
|---|---|---|---|---|---|
| F01 增量刷新与完整性 | `src/mf/data_fetch.py`, `src/mf/data_fetch_ak.py`, `src/mf/manifest.py`, `scripts/01_fetch_data.py` | `tests/test_data_refresh.py` 11 个（过期重抓、历史成分不改写、失败与空返回即失败、大修订需确认、日期断层、指数不到期末、覆盖检查） | **部分修复**：代码与测试已完成；**尚未用真实网络把数据延长到 11 月再跑一遍**（验收第 2、9 条），当前缓存还没有 manifest，所以前瞻的"数据覆盖检查"显示未通过 | 无 | 新浪对停牌日不返回数据，单只股票缺一天无法区分停牌与缺数，只在超过 5% 的活跃股票同时缺失时才报错 |
| F02 前瞻登记早于执行 | `src/mf/forward.py`, `reports/forward/registry.csv`, `registry_v1_legacy.csv` | `test_registration_after_execution_is_late_and_backfill_cannot_fix_it` | **已修复**：登记记录 `signal_time / execution_bound / registered_at_utc`；2026-10-30 的目标改标 `late_registration`（仅历史 / 演示），原登记表原样保留 | 无 | — |
| F03 前瞻绩效真实性 | `src/mf/forward.py` | `test_strict_forward_days_need_preregistration_hash_and_git_time` | **已修复**：逐日判断驱动收益的目标是否事前登记、哈希一致、有早于执行的 git 提交时间；执行日新旧两期都要有效；0 天显示"尚未开始"，规则一致 / 登记完整 / 数据覆盖分开显示 | 无 | — |
| F04 冻结可复现 | `src/mf/forward.py`, `scripts/freeze_rules.py`, `reports/forward/rules/` | `test_freeze_refuses_dirty_tree` | **部分修复**：新版本必须在干净提交上冻结，记录完整 SHA、Python 与 `pip freeze`、数据内容哈希、全部代码与配置的指纹；旧 v1 原样保留在 `rules/v1.json`。**v2 还没有冻结**：要等本分支提交后在干净工作区执行 | 无 | 本地 CSV 本身可改，独立时间戳依赖推送到 GitHub 的提交时间 |
| F05 历史与前瞻隔离 | `scripts/06_attribution_report.py`, `src/mf/html_report.py`, `src/mf/report_template.html`, `scripts/verify_report.py` | `test_history_window_is_unchanged_by_later_data`；`verify_report.py` 检查 HTML 历史序列不超过 2026-10-31 | **已修复**：历史图表、卡片、年度表固定止于 `split.oos_end`；前瞻单独计算展示 | 无 | 数据修订后的"重述版"对比依赖冻结时记录的历史指标（v2 冻结后生效） |

## P1：研究、基准、风险与成交

| 编号 | 修改文件 | 测试 / 核对 | 结果 | 影响 | 未解决的限制 |
|---|---|---|---|---|---|
| Q01 财报 PIT | `src/mf/facts.py`、报告措辞 | 限制表自动生成 | **未解决**：没有首次披露日与原值数据；页面和报告标为"未严格 PIT"；对照为去掉全部财报因子的方案（IR 0.18） | — | 需要首次披露日历与首发原值 |
| Q02 时点行业 / ST / 退市 | `src/mf/data_audit.py`, `src/mf/facts.py` | ST 分半稳定性检查 | **未解决**：行业仍为当前分类；ST 只做了分半检查（明确说明不是独立持出样本）；退市股缺行情 | — | 需要时点行业、真实 ST 历史、退市行情 |
| Q03 退市与长期停牌 | `src/mf/backtest.py`, `scripts/06_attribution_report.py` | `test_permanently_untraded_holding_is_flagged_and_can_be_haircut` | **已修复**（审计层面）：持仓连续 60 个交易日不可交易即列为审计异常，可选保守减值情景；本次回测 0 例 | 无 | 没有真实的退市终值与处置价格 |
| Q04 基准前一日市值 | `src/mf/panel.py`, `scripts/02_build_panel_audit.py` | `test_benchmark_uses_previous_trading_day_cap_on_reentry`；覆盖率输出 | **已修复**：先按股票全部交易日取前一日市值再连接成分；输出覆盖率（平均 96.5%，最低 92.4%） | 验证期超额 +0.04pp；2016 年基准补上了 01-04 | 成分快照按月生效，6 月 / 12 月调整后约两周不同步 |
| Q05 成交逐笔对账 | `src/mf/backtest.py` | `test_every_execution_reconciles_in_the_ledger`、合成端到端 | **已修复**：每次成交断言 交易前净值 − 成本 = 交易后净值、现金不为负，并输出账本 | 无 | 未建模：100 股整手、券商最低佣金、跳空、分多日成交（报告中列为假设） |
| Q06 换手口径与豁免 | `src/mf/optimizer.py`, `config.yaml` | `test_turnover_limit_applies_unless_book_is_mostly_cash` | **已修复**：计划换手 = 0.5 ×（股票交易额 + 被迫卖出），现金建仓计入；仅当信号日现金 ≥ 50% 时豁免，并逐次记录（本次 1 次：首次建仓） | 无 | — |
| Q07 实际持仓约束 | `scripts/06_attribution_report.py` | results.md "执行与约束审计" | **已修复**：放宽、豁免、卖出受阻、剩余现金分开计数；6 个超约束月份逐条列出原因；逐年对比事前 / 事后跟踪误差（事后约高 20%） | 无 | — |
| Q08 缺失数据不当 0 | `src/mf/metrics.py`, `src/mf/checks.py`, `src/mf/html_report.py`, `scripts/06_attribution_report.py` | `validate_long`（重复键、日历缺口、非正市值即失败） | **已修复**：基准缺失日直接报错；数据检查写入 `data_checks.json` | 无 | 新浪停牌日无记录，停牌与缺数在单只股票层面无法区分 |
| Q09 最大回撤首日 | `src/mf/metrics.py` | `test_max_drawdown_counts_a_loss_on_the_first_day` | **已修复** | 历史数字无变化（首日均为正） | — |
| Q10 风险模型输入 | `src/mf/risk.py` | `test_risk_model_inputs_are_finite_psd_and_new_factors_not_riskless` | **已修复**：历史不足的因子用同类中位方差、协方差半正定投影、特异方差下限、非有限值即失败 | 是 λ 从 5 翻到 7 的原因之一 | — |
| Q11 理论 vs 可成交 IC | `scripts/04_models.py`, results.md | 自动表格 | **已修复**：分开报告全候选与次日可买入样本的 RankIC 及受阻占比；次日信息只用于评估 | 无 | — |

## P1/P2：统计、稳健性与措辞

| 编号 | 修改文件 | 结果 | 未解决的限制 |
|---|---|---|---|
| S01 显著性与多重检验 | `src/mf/factor_test.py`, `scripts/03_factors.py`, results.md | **部分修复**：增加 Newey-West HAC t 与 BH-FDR q 值（仅展示，筛选规则不变）；报告同时展示 Top10% 超额与多空收益（Ridge 为负） | 开发中比较过多个模型和参数，这部分选择偏差无法完全校正 |
| S02 冻结模型选择 | `src/mf/forward.py`, 文档 | **已修复**：2022–2026 统一称"历史验证期（反复查看）"；新规则版本必须显式迁移 | — |
| S03 稳健性一致性 | `scripts/07_robustness.py` | **已修复**：成本情景改称"相同计划目标"；财报扰动注明为信号输入敏感性；缓存按方案内容与主回测内容哈希 | — |
| S04 容量 | `scripts/07_robustness.py`, HTML | **已修复**：冲击系数 0.5 / 1 / 2 并列；参与率超过 100% 的交易金额占比与"一日内不可行"标记；明确只是量级 | 未建模分多日成交 |
| S05 因果 / 统计措辞 | 报告、HTML、`src/mf/facts.py` | **已修复**：删去"t ≥ 2 才显著"的卡片；残差不再称为选股能力；"信号是真的"改为保守表述；Bootstrap 区间跨 0 写入摘要 | — |

## P2：报告与网页

| 编号 | 结果 |
|---|---|
| R01 OOS 相关图 | **已修复**：OOS 图使用 OOS 数据自己的相关矩阵（`factor_corr_oos.csv`），不参与筛选；注明标签成熟的最后信号日 |
| R02 按钮范围与标题 | **已修复**：不随按钮变化的部分标注"固定区间"；首尾年份标注不完整 |
| R03 统计卡片 | **已修复**：删除 IR×√年数 的显著性提示；0 个前瞻日不显示绩效 |
| R04 HTML 与配置一致 | **已修复**：页面参数显示实际选中的 λ；`verify_report.py` 独立复算核对 |
| R05 误导性论述 | **已修复**：限制表每次运行自动生成，写入 results.md 和网页 |
| R06 版本化与导航 | **部分修复**：v1–v5 快照保留在 `data/snapshots/`（本地，不进 git）；合成端到端测试进入 CI；README 已更新导航。完整真实数据回测的日志与哈希要在真实刷新后保存 |

## 一次性验收红线（逐条）

| # | 内容 | 状态 |
|---|---|---|
| 1 | pytest 全部通过，3.9 / 3.11 CI | 本地 3.9 通过（46 个）；CI 待推送后确认 |
| 2 | 完整原始数据跑 `run_all.sh`，各文件覆盖到配置的 end，manifest 一致 | **未完成**：需要联网刷新（约 40 分钟），尚未执行 |
| 3 | 逐日对账、持仓 CSV 与回测一一对应、超约束都有说明 | 通过（账本断言、合成测试、verify_report、审计明细） |
| 4 | 基准退出后回归 / 前一日缺价 / 月初换成分 | 回归情景有测试；前一日缺价按股票自身最后交易日取市值；月初生效规则已说明 |
| 5 | 各报告公式与分段一致，独立复算 | 通过（verify_report） |
| 6 | 保留 v5 快照并列示差异 | 通过（revision_v6.md） |
| 7 | 历史止于冻结前最后交易日；前瞻必须事前登记 | 通过（代码与测试）；当前 0 个严格前瞻日 |
| 8 | 严格 PIT 未通过时明确标注并给出对照 | 通过（限制表、去掉财报因子对照） |
| 9 | 移动样本末端（含 11 月新数据）再跑 | **未完成**：依赖第 2 条的真实刷新 |
| 10 | 本清单 | 本文件 |

**结论：工程类问题已修复并有测试；"严格 PIT 与完全无偏"未通过（缺外部数据）；验收第 2、9 条（真实数据刷新到 11 月后重跑）和 v2 规则冻结尚未执行，因此整体还不能称为"全部完成"。**
