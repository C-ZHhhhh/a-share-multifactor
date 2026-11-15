# A股多因子选股与动态组合优化

**A-share multi-factor stock selection & risk-constrained portfolio optimisation (CSI 800, monthly)**

[![tests](https://github.com/C-ZHhhhh/a-share-multifactor/actions/workflows/tests.yml/badge.svg)](https://github.com/C-ZHhhhh/a-share-multifactor/actions/workflows/tests.yml) ![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

这是一个完整的端到端量化研究框架，按顺序包括六步：

1. 时点数据获取与质量审计
2. 19 个因子的构建与检验
3. Walk-forward 选股模型
4. Barra 式风险模型与 cvxpy 组合优化
5. 含 A 股交易限制的日频回测
6. 样本外验证，以及 Brinson 和因子归因

每次运行都会自动生成一份**交互式可视化报告** [reports/report.html](reports/report.html)，用浏览器直接打开即可，不需要联网（[在线预览](https://c-zhhhhh.github.io/a-share-multifactor/reports/report.html)）。报告包含净值、超额收益、分年度表现、选股信号、因子筛选、最新持仓、归因和优化器诊断，并可以在全部、样本内、样本外之间切换。

每个月都会产出一份**具体的股票持仓清单**（平均 44 只），回测收益就是按这些持仓逐日计算的。最新一期见 [reports/holdings/latest_portfolio.md](reports/holdings/latest_portfolio.md)。

![nav](reports/figures/nav.png)

## 核心结果

> **规则已于 2026-11-15 冻结**。下表全部是**历史回测**：2022 年以来的数据在开发中被反复查看，不能当作一次性的样本外检验。真正的检验是冻结之后的前瞻样本，见 [前瞻跟踪](reports/forward/README.md)（第一期目标仓位已登记，尚无前瞻收益）。

调仓频率为月度，回测含交易成本。基准是**自定义流通市值基准**（时点中证800成分股按流通市值加权的全收益），不是官方中证800指数，所以这是"自定义基准增强"的结果。

| 组合优化 (Ridge, 目标跟踪误差 4% → λ=5) | 年化收益 | 超额年化 | 跟踪误差 | 信息比率 | 最大回撤 | 年化换手 | 超额 vs 官方中证800全收益 |
|---|---|---|---|---|---|---|---|
| 样本内 2018-03 ~ 2021-12 | 5.84% | −1.51% | 4.7% | −0.35 | −24.4% | 178% | −1.80% |
| **样本外 2022-01 ~ 2026-10** | **2.52%** | **+1.45%** | 5.2% | **0.25** | −21.6% | 213% | **+4.42%** |

- **信号稳定**：三种模型的样本外 RankIC 在 0.057–0.068。因子每月滚动筛选，只用当时已实现的标签。
- **组合超额为正但不显著**：λ 按事先定好的风险预算（样本内事前跟踪误差 4%）选择，不看收益。样本外超额 +1.45%，Bootstrap 重抽样中主动收益 ≤ 0 的比例约 31%。
- **结论在各种合理方案下方向一致**：换跟踪误差目标、加约束、换样本切分、扰动财报数据，样本外 IR 都在 0.10–0.45；GBDT 模型做成组合后为 −0.25。详见 [稳健性报告](reports/robustness.md)。
- **容量有限**：按平方根冲击模型，1 亿元时额外冲击约 0.7%/年，5 亿元时基本吃掉全部超额。
- **持仓可查、口径分明**：回测是持仓市值 + 现金的账户。每个月的持仓都导出在 [reports/holdings/](reports/holdings/)，分列信号日实际持仓、目标仓位、交易前、成交后和现金，并标出买入受阻、卖出受阻等成交状态。
- **版本**：v2 修正时间泄漏、组合状态、指标与归因对账（[修订记录](reports/revision_log.md)）；v3 按目标跟踪误差选 λ；v4 改为滚动选因子，并修正回测资金处理、容量单位和稳健性脚本（研究报告 5.7–5.9 节）。v5 改为账户记账、统一仓位口径并冻结规则。样本外 IR：v1 0.28 → v2 0.91 → v3 0.20 → v4 0.24 → v5 0.25。

## 方法要点（防偏差设计）

| 环节 | 做法 |
|---|---|
| 幸存者偏差 | 股票池采用每月的沪深300 + 中证500成分股快照 |
| 财报前视 | 财报按公告日的**次日**起使用；首次披露日按 min(最新公告日, 法定截止日) 估计；每股指标随送转股调整（有单元测试） |
| 复权 | 后复权因子推导 preclose，总收益包含分红送转；与 baostock 交叉验证，99.95% 的日收益误差小于 0.1pp |
| 标签泄漏 | 每个信号日记录标签实现日；训练、因子筛选（每月滚动）、模型比较、λ 校准都只用已实现的标签（有单元测试） |
| 过拟合 | 因子每月滚动筛选，模型和 λ 只在样本内按事先定好的规则选择。2022 年以来的数据在开发中被多次查看，所以规则已冻结，只有冻结后的前瞻样本算真正的检验 |
| 交易现实 | T+1 VWAP 成交；一字板和停牌不可交易；持仓市值 + 现金的账户记账，成本从现金支付、不借钱；优化器以实际持仓为起点；佣金、滑点、分段印花税 |
| 基准 | 自定义流通市值基准（含分红）为主口径，官方中证800全收益为副口径；本项目是自定义基准增强，不是官方指数增强 |

## 项目结构

```
config.yaml                 全部参数：样本切分、过滤、模型、约束、成本
src/mf/
  data_fetch_ak.py          新浪日线 + 复权因子、东方财富季报、中证全收益指数（限速、退避、断点续传）
  data_fetch.py             baostock：时点成分股、行业、上市日期、价格指数
  panel.py                  日频面板、ST 推断、涨跌停、时点基本面、复原基准
  data_audit.py             数据质量审计与跨源交叉验证
  factors.py                19 个因子 + 预处理（MAD、标准化、行业/市值中性）
  dataset.py                月度截面、T+1 VWAP 标签、可交易标记
  factor_test.py            RankIC、ICIR、分组、IC 衰减、自相关、因子筛选（含每月滚动筛选）
  models.py                 ICIR 加权 / Ridge / GBDT 的 walk-forward 训练
  risk.py                   行业 + 6 风格因子的 WLS 截面回归，EWMA 协方差
  optimizer.py              cvxpy 均值-方差主动优化
  backtest.py               日频回测引擎
  portfolio.py              组合构建：优化器嵌入回测，每月以实际持仓为起点
  holdings.py               导出每月持仓（四种仓位口径 + 现金）与最新组合报告
  forward.py                冻结规则、登记前瞻目标仓位、前瞻结果
  html_report.py            生成交互式可视化报告（模板 report_template.html，纯 JS 画图，离线可用）
  metrics.py, attribution.py  绩效指标、Brinson 与因子归因
scripts/01 … 08             八个流程步骤（compare_runs.py 用于新旧版本对比）
tests/                      25 个单元测试（前视、标签可用日期、滚动选因子、时点、成本、涨跌停、阻断成交、现金账户、调出指数、归因恒等式、前瞻登记）
reports/                    研究报告、审计、因子检验、回测结果、图表
reports/report.html         每次运行自动生成的交互式可视化报告
reports/holdings/           每月持仓 CSV + 最新一期组合 (latest_portfolio.md)
docs/                       学习指南、面试手册、简历表述
```

## 复现与使用

### 环境
- Python 3.9 及以上（开发环境为 3.9.6），macOS 或 Linux；Windows 请用 WSL，或手动逐条运行 `run_all.sh` 里的命令。
- 需要联网：数据来自免费接口，**不需要账号或 token**。

### 一键复现
```bash
git clone https://github.com/C-ZHhhhh/a-share-multifactor.git
cd a-share-multifactor
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash run_all.sh
```
`run_all.sh` 按顺序执行以下步骤，任何一步出错就停止：

| 步骤 | 命令 | 做什么 | 首次耗时 |
|---|---|---|---|
| 0 | `pytest tests` | 25 个单元测试（前视、标签可用日期、滚动选因子、时点财报、成本、涨跌停、阻断成交、现金账户、调出指数、归因恒等式、前瞻登记） | 几秒 |
| 1 | `scripts/01_fetch_data.py` | 下载数据到 `data/raw/`（限速、失败重试、断点续传） | 约 45 分钟 |
| 2 | `scripts/02_build_panel_audit.py` | 日频面板 + 数据质量审计 → `reports/data_audit.md` | 约 1 分钟 |
| 3 | `scripts/03_factors.py` | 19 个因子、单因子检验、样本内筛选 → `reports/factor_tests.md` | 约 1 分钟 |
| 4 | `scripts/04_models.py` | Walk-forward 训练 ICIR 加权 / Ridge / GBDT | 约 1 分钟 |
| 5 | `scripts/05_portfolio_backtest.py` | 风险模型、λ 校准、组合优化、日频回测 | 约 1 分钟 |
| 6 | `scripts/06_attribution_report.py` | 样本内外评估、归因与对账、约束满足率、**持仓导出** → `reports/results.md`、`reports/holdings/` | 约 1 分钟 |
| 7 | `scripts/07_robustness.py` | 参数、约束、样本切分、财报数据敏感性、成本情景、Bootstrap、容量 → `reports/robustness.md`（结果缓存在 `data/processed/robustness/`） | 约 4 分钟 |
| 8 | `scripts/08_html_report.py` | **交互式可视化报告** → `reports/report.html` | 几秒 |

下载的数据缓存在 `data/` 中（不进 git），第二次运行约 10 分钟。下载中断后重新运行即可从断点继续。

### 产出
| 文件 | 内容 |
|---|---|
| **`reports/report.html`** | **本次运行的交互式可视化报告**（单个文件、离线可用，悬停看数值，可切换样本内/外） |
| `reports/runs/report_<时间>.html` | 每次运行留一份带时间戳的副本，方便对比不同参数或不同下载日期的结果（不进 git） |
| `reports/forward/` | **前瞻跟踪**：冻结的规则、只追加的目标仓位登记表、前瞻结果（与历史回测分开） |
| `reports/robustness.md` | 稳健性、数据敏感性与容量检验 |
| `reports/revision_log.md` | v1 → v2 方法修订的新旧对比（由 `scripts/compare_runs.py` 生成） |
| `reports/results.md` | 绩效表（样本内/外）、分年度、λ 校准、Brinson 与因子归因 |
| `reports/holdings/latest_portfolio.md` | 最新一期全部持仓、调仓、行业主动权重 |
| `reports/holdings/optimized_monthly.csv` | 组合优化的每月持仓（可用 Excel 打开） |
| `reports/holdings/top50_equal_weight_monthly.csv` | Top50 等权对照组的每月持仓 |
| `reports/data_audit.md`、`reports/factor_tests.md` | 数据审计与单因子检验 |
| `reports/figures/*.png` | 全部图表 |

### 修改参数做实验
几乎所有参数都在 [config.yaml](config.yaml)，不用改代码：
- `split`：样本内/外切分；`filters`：ST、次新股、停牌过滤；`model.min_train_months`：最少训练月数；
- `portfolio`：`target_te`（λ 的跟踪误差目标）、`lambda_grid`、`max_active_weight`（单票主动权重上限）、`industry_dev`、`size_dev`、`turnover_limit`、`topn`；
- `costs`：佣金、印花税、滑点。

只改了组合或成本参数时，不用重跑前面的步骤：
```bash
.venv/bin/python scripts/05_portfolio_backtest.py && .venv/bin/python scripts/06_attribution_report.py
```

### 更新到最新数据（前瞻跟踪）
规则已冻结（`reports/forward/frozen_rules.json`）。更新数据后重跑，会继续使用冻结的模型和 λ；新的目标仓位会自动登记，冻结日之后的收益单独计入前瞻结果。如果改动了核心代码或参数，前瞻结果会被标为无效。

把 `config.yaml` 里的 `data.end` 和 `split.oos_end` 改成新的日期，删除旧缓存后重跑全部流程（下载缓存按文件存在与否判断，不删除不会补下新数据）：
```bash
rm -rf data && bash run_all.sh
```

### 注意
- **复现的数字不会逐位相同**：数据源会修订少量历史财报。我们在全新环境中独立复现过一次：价格完全一致，信号的样本外 ICIR 一致（约 1.5），但一个处于筛选门槛边缘的因子入选与否，会让样本外 IR 在 0.18–0.28 之间变动。详见 [研究报告 5.5 节](reports/research_report.md#55-复现稳健性两次独立下载的对比)。本仓库的数字以 2026-11-12 下载的数据为准。
- 数据来自 [AkShare](https://github.com/akfamily/akshare)（新浪、东方财富、中证指数）和 [baostock](http://baostock.com)，仅用于研究，不在仓库中分发。
- 本项目是研究框架，产出的持仓不构成投资建议。

## 报告

- [可视化报告](reports/report.html)（下载后用浏览器打开，或[在线预览](https://c-zhhhhh.github.io/a-share-multifactor/reports/report.html)）
- [研究报告](reports/research_report.md)：方法、结果、归因、局限，以及研究中发现并修正的 6 个问题
- [最新一期持仓](reports/holdings/latest_portfolio.md) · [每月持仓 CSV](reports/holdings/)
- [数据审计](reports/data_audit.md) · [单因子检验](reports/factor_tests.md) · [回测结果明细](reports/results.md)
- [学习指南](docs/learning_guide.md) · [面试手册](docs/interview_guide.md)

## 局限

- 79 只退市股票缺少数据，存在残余的幸存者偏差。
- 行业分类不是时点数据。
- 财报首次披露日是估计值：91.5% 的记录按法定截止日对齐，逾期披露的公司存在前视。
- ST 状态是推断值。
- 官方中证800的历史成分权重无法免费获取，主基准是自定义流通市值基准，每年和官方指数相差 2–5%。
- 冲击成本用平方根模型估算，系数没有用真实成交校准。
- 组合结果对组合构建方式（λ、模型、风格约束）高度敏感，主方案样本外超额统计上不显著（报告 5.6–5.8 节）。

详见报告第 1.4 节和第 7 节。
