# 简历表述

> 所有数字都来自 `reports/research_report.md`，面试时可以逐一溯源。
> **不要改大数字**，也不要只写样本内的结果。面试官问到"样本外怎么样"时，你能当场说清楚，这比数字本身更重要。

## 中文版（选 3–5 条）

**A股多因子选股与动态组合优化**　个人项目　Python / pandas / scikit-learn / cvxpy　 [github.com/C-ZHhhhh/a-share-multifactor](https://github.com/C-ZHhhhh/a-share-multifactor)

- 搭建端到端的量化研究框架（数据 → 因子 → 模型 → 优化 → 回测 → 归因），覆盖时点中证800股票池（1,435 只股票、365 万条日度记录），并编写 9 个单元测试来防止前视偏差。
- 设计数据质量审计并做跨源交叉验证：日收益与独立数据源的一致率为 99.95%，ST 推断精确率为 82%。在审计中发现财报"最新公告日"并不是首次披露日，改用法定截止日估算，使基本面因子的缺失率从 27% 降到 0.3%。
- 构建价值、质量、成长、动量、波动、流动性、量价等 19 个因子，经过 MAD 去极值、行业与市值中性化后做 RankIC 检验，在样本内筛选出 11 个；低风险类因子的样本外年化 ICIR 稳定在 1.4–1.9。
- 用 walk-forward 方式训练 ICIR 加权、Ridge、GBDT 三种模型，训练集与预测期之间留 1 个月 gap 防止标签泄漏；按样本内 ICIR 选出的 Ridge 模型，样本外 RankIC 为 0.057、ICIR 为 1.49。
- 实现简化的 Barra 风险模型（行业 + 6 个风格因子，EWMA 协方差），用 cvxpy 求解带主动权重、行业、市值、换手约束的均值-方差优化。和同一信号的 Top50 等权组合相比，换手下降 67%，成本拖累从 2.2% 降到 0.7%，样本外信息比率从 −0.13 提升到 0.28。
- 日频回测考虑了 T+1 VWAP 成交、一字板与停牌限制、分段印花税；样本外（2022–2026）相对复原的中证800全收益超额 +1.7%，相对官方全收益指数超额 +4.7%。用 Brinson 和风险因子归因定位超额来源（以选股效应和特异收益为主），并量化了 long-only 约束带来的信息损失（转移系数 0.30）。

**精简版（空间不够时用这 3 条）**
- 独立搭建 A 股中证800 多因子研究框架，覆盖时点数据审计、19 个因子检验、walk-forward 机器学习选股、Barra 风险模型与 cvxpy 组合优化、含交易限制的回测，以及 Brinson 和因子归因。
- 通过跨源交叉验证（收益一致率 99.95%）和时点财报对齐保证数据可信；用 1 个月 gap 和严格的样本内/外切分避免前视和过拟合。
- 优化组合样本外（2022–2026）信息比率 0.28、超额 +1.7%（相对官方全收益指数 +4.7%）；相比 Top-N 等权，换手降低 67%，成本拖累降低 1.5 个百分点。

## English version

**A-Share Multi-Factor Stock Selection & Portfolio Optimisation**　Personal project　Python, pandas, scikit-learn, cvxpy

- Built an end-to-end quant research pipeline (data → factors → models → optimisation → backtest → attribution) on a point-in-time CSI 800 universe (1,435 stocks, 3.65M stock-days), with 9 unit tests guarding against look-ahead bias.
- Designed a data-quality audit with cross-source validation (99.95% daily-return agreement; 82% precision on inferred ST flags). Found that the vendor's "latest announcement date" is not the first release date and re-estimated it from statutory deadlines, cutting missing fundamentals from 27% to 0.3%.
- Engineered 19 factors (value, quality, growth, momentum, volatility, liquidity, price-volume). After MAD winsorisation and industry/size neutralisation, tested them with RankIC and selected 11 in-sample; the low-risk factors kept an out-of-sample annualised ICIR of 1.4–1.9.
- Trained walk-forward ICIR-weighted, Ridge and gradient-boosting models, leaving a 1-month gap between training labels and the prediction date to prevent label leakage. The Ridge model, chosen on in-sample ICIR, delivered an OOS RankIC of 0.057 (ICIR 1.49).
- Implemented a Barra-style risk model (industries + 6 styles, EWMA covariance) and a cvxpy mean-variance optimiser with active-weight, industry, size and turnover constraints. Against a top-50 equal-weight portfolio on the same signal, it cut turnover by 67%, reduced cost drag from 2.2% to 0.7%, and raised the OOS information ratio from −0.13 to 0.28.
- Ran daily backtests with T+1 VWAP execution, limit-up/limit-down and suspension constraints, and the stamp-duty change. Over 2022–26 the portfolio earned +1.7% p.a. excess over the reconstructed CSI 800 total-return benchmark (+4.7% vs the official index). Brinson and factor attribution traced the excess mainly to stock selection, and the transfer coefficient (0.30) quantified the information lost to the long-only constraint.

## 面试官可能会对哪个数字追问，你要准备什么

| 数字 | 追问 | 去哪里找答案 |
|---|---|---|
| 99.95% | 跟谁比？为什么不是 100%？ | 报告 1.3：IPO 首日新浪没有 preclose；少量分红日两个数据源的口径不同 |
| 27% → 0.3% | 怎么发现的？会不会引入前视？ | 报告第 6 节第 1 条；对逾期披露的公司有轻微前视，已写进局限 |
| ICIR 1.49 | 为什么样本外反而不差？ | 低风险因子在 A 股长期稳定；但质量和成长类因子在样本外失效 |
| IR 0.28 | 这不算高吧？ | 坦然承认：long-only 的转移系数只有 0.3，样本内基本是 0；用"如何改进"展示你的思考 |
| +1.7% vs +4.7% | 为什么有两个超额？ | 流通市值和自由流通市值加权的差异（报告 1.3） |
| IR 0.28 | 别人能复现出一样的数吗？显著吗？ | 不会逐位相同：独立重新下载后，一个边界因子入选，IR 变成 0.18；固定因子后回到 0.27（报告 5.5）。t 值约 0.6，不显著，要坦然说"方向为正但不显著" |
