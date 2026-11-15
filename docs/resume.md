# 简历表述

> 所有数字都来自 `reports/research_report.md`，面试时可以逐一溯源。
> **不要改大数字**，也不要只写样本内的结果。面试官问到"样本外怎么样"时，你能当场说清楚，这比数字本身更重要。

## 中文版（选 3–5 条）

**A股多因子选股与动态组合优化**　个人项目　Python / pandas / scikit-learn / cvxpy　 [github.com/C-ZHhhhh/a-share-multifactor](https://github.com/C-ZHhhhh/a-share-multifactor)

- 搭建端到端的量化研究框架（数据 → 因子 → 模型 → 优化 → 回测 → 归因），覆盖时点中证800股票池（1,435 只股票、365 万条日度记录），并编写 单元测试来防止前视偏差。
- 设计数据质量审计并做跨源交叉验证：日收益与独立数据源的一致率为 99.95%，ST 推断精确率为 82%。在审计中发现财报"最新公告日"并不是首次披露日，改用法定截止日估算，使基本面因子的缺失率从 27% 降到 0.3%。
- 构建价值、质量、成长、动量、波动、流动性、量价等 19 个因子，经过 MAD 去极值、行业与市值中性化后做 RankIC 检验，每月只用已实现的标签滚动筛选（平均约 10 个）；低风险类因子的样本外年化 ICIR 稳定在 1.4–1.9。
- 用 walk-forward 方式训练 ICIR 加权、Ridge、GBDT 三种模型；为每个信号日记录标签实现日期，训练、因子筛选、模型与参数选择统一只用已实现的标签；三种模型样本外 RankIC 在 0.057–0.068。
- 实现简化的 Barra 风险模型（行业 + 6 个风格因子，EWMA 协方差），用 cvxpy 求解带主动权重、行业、市值、换手约束的均值-方差优化，并把优化器嵌入回测、以实际持仓为起点。和同一信号的 Top50 等权组合相比，换手下降约 70%，成本拖累从 2.2% 降到 0.7%。
- 日频回测考虑了 T+1 VWAP 成交、一字板与停牌限制、分段印花税；按 4% 跟踪误差的风险预算选择风险厌恶系数，历史回测（2022–2026）相对自定义流通市值基准超额 +1.45%（IR 0.25，不显著）。通过参数、约束、样本切分、数据扰动、Bootstrap 和容量检验确认：各合理方案 IR 在 0.10–0.45、方向一致，集中持仓时行业、风格暴露和成本吃掉了大部分选股收益；Brinson 与因子归因逐项对账到实际超额。

**精简版（空间不够时用这 3 条）**
- 独立搭建 A 股中证800 多因子研究框架，覆盖时点数据审计、19 个因子检验、walk-forward 机器学习选股、Barra 风险模型与 cvxpy 组合优化、含交易限制的回测，以及 Brinson 和因子归因。
- 通过跨源交叉验证（收益一致率 99.95%）和时点财报对齐保证数据可信；所有样本内决策只使用已实现的标签，避免前视。
- 发现并修正了样本内标签泄漏、优化器起点错误和事后挑选参数的问题；在事先定好的风险预算下超额 +1.45%，并用稳健性检验、Bootstrap 和容量分析如实评估其不显著性；冻结规则并登记前瞻样本，与反复检验过的历史结果分开呈现。

## English version

**A-Share Multi-Factor Stock Selection & Portfolio Optimisation**　Personal project　Python, pandas, scikit-learn, cvxpy

- Built an end-to-end quant research pipeline (data → factors → models → optimisation → backtest → attribution) on a point-in-time CSI 800 universe (1,435 stocks, 3.65M stock-days), with unit tests guarding against look-ahead bias.
- Designed a data-quality audit with cross-source validation (99.95% daily-return agreement; 82% precision on inferred ST flags). Found that the vendor's "latest announcement date" is not the first release date and re-estimated it from statutory deadlines, cutting missing fundamentals from 27% to 0.3%.
- Engineered 19 factors (value, quality, growth, momentum, volatility, liquidity, price-volume). After MAD winsorisation and industry/size neutralisation, tested them with RankIC and re-screened every month on realised labels only (about 10 selected); the low-risk factors kept an out-of-sample annualised ICIR of 1.4–1.9.
- Trained walk-forward ICIR-weighted, Ridge and gradient-boosting models. Every signal date carries the date its label is realised, and training, factor screening, model and parameter selection only use realised labels; OOS RankIC was 0.057–0.068 across models.
- Implemented a Barra-style risk model (industries + 6 styles, EWMA covariance) and a cvxpy mean-variance optimiser with active-weight, industry, size and turnover constraints, run inside the backtest from the actual drifted holdings. Against a top-50 equal-weight portfolio on the same signal, it cut turnover by about 70% and cost drag from 2.2% to 0.7%.
- Ran daily backtests with T+1 VWAP execution, limit-up/limit-down and suspension constraints, and the stamp-duty change. With risk aversion set by a 4% tracking-error budget, the portfolio earned +1.45% p.a. over a custom float-cap CSI 800 benchmark in 2022–26 (IR 0.25, not significant); rules were then frozen and every new target portfolio is registered for a genuine forward test. Robustness tests (parameters, constraints, sample splits, data perturbations, block bootstrap, square-root market impact) gave IRs of 0.10–0.45 across reasonable variants, with unintended industry/style exposures and costs eating most of the stock-selection return; attribution was reconciled line by line to the realised excess.

## 面试官可能会对哪个数字追问，你要准备什么

| 数字 | 追问 | 去哪里找答案 |
|---|---|---|
| 99.95% | 跟谁比？为什么不是 100%？ | 报告 1.3：IPO 首日新浪没有 preclose；少量分红日两个数据源的口径不同 |
| 27% → 0.3% | 怎么发现的？会不会引入前视？ | 报告第 6 节第 1 条；对逾期披露的公司有轻微前视，已写进局限 |
| RankIC 0.06 | 为什么样本外反而不差？ | 低风险因子在 A 股长期稳定；但质量和成长类因子在样本外失效 |
| IR 0.25 | 这么低还放简历上？ | 规则事先定好、没有用样本外调参；信号本身显著，问题在 long-only 变现效率，归因能讲清楚钱被谁吃掉了（报告 5.3、5.6、5.8） |
| +1.45% vs +4.42% | 为什么有两个超额？ | 自定义流通市值基准和官方指数的构造差异，不是能力（报告 1.3、5.6） |
| v1 → v5 | 为什么 IR 是 0.28 → 0.91 → 0.20 → 0.24 → 0.25？ | v2 修了泄漏和组合状态，λ=50 碰巧最好；v3 按风险预算选 λ；v4 滚动选因子；v5 账户记账并冻结规则（报告 5.7–5.10） |
| 前瞻样本 | 现在跑得怎么样？ | 规则冻结于 2026-11-15，第一期目标仓位已登记；前瞻收益要等新数据（reports/forward/） |
