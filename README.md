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

调仓频率为月度，回测含交易成本。基准是**自定义流通市值基准**（时点中证800成分股按流通市值加权的全收益），不是官方中证800指数，所以这是"自定义基准增强"的结果。

| 组合优化 (ICIR 加权, λ=50) | 年化收益 | 超额年化 | 跟踪误差 | 信息比率 | 最大回撤 | 年化换手 | 超额 vs 官方中证800全收益 |
|---|---|---|---|---|---|---|---|
| 样本内 2018-03 ~ 2021-12 | 6.81% | −0.60% | 2.2% | −0.29 | −24.9% | 151% | −0.90% |
| **样本外 2022-01 ~ 2026-10** | **3.08%** | **+2.01%** | 2.2% | **0.91** | −21.3% | 164% | **+4.99%** |

- **信号稳定**：三种模型的样本外 RankIC 在 0.057–0.068，选中的 ICIR 加权模型样本外年化 ICIR 为 1.97。
- **结果对组合构建参数敏感**：换成样本内没选中的 λ 或模型，样本外 IR 在 −0.10 到 0.71 之间（λ=2 时为 −0.10）；换样本切分、去掉财报因子后在 0.65–0.71。Bootstrap 下样本外 IR 的 95% 区间为 [−0.11, 1.90]。详见 [稳健性报告](reports/robustness.md)。
- **容量有限**：按平方根冲击模型，约 20 亿元时冲击成本吃掉大约 2/3 的超额，100 亿元时超额转负。
- **持仓可查**：每个月的完整持仓都导出在 [reports/holdings/](reports/holdings/)，每只股票都有组合权重、基准权重、主动权重和模型排名。
- **v2 方法修订**（2026-11-15）：修正了样本内用到样本外才实现的标签、优化器没有以实际持仓为起点、夏普 / IR 定义和归因对账。v1 的样本外 IR 是 0.28，逐项影响见 [修订记录](reports/revision_log.md)。

## 方法要点（防偏差设计）

| 环节 | 做法 |
|---|---|
| 幸存者偏差 | 股票池采用每月的沪深300 + 中证500成分股快照 |
| 财报前视 | 财报按公告日的**次日**起使用；首次披露日按 min(最新公告日, 法定截止日) 估计；每股指标随送转股调整（有单元测试） |
| 复权 | 后复权因子推导 preclose，总收益包含分红送转；与 baostock 交叉验证，99.95% 的日收益误差小于 0.1pp |
| 标签泄漏 | Walk-forward 训练与预测之间留 1 个月 gap（有单元测试） |
| 过拟合 | 因子、模型、λ 只在样本内选择，样本外只评估一次 |
| 交易现实 | T+1 VWAP 成交；一字板和停牌不可交易；佣金、滑点、分段印花税 |
| 基准 | 使用全收益指数（组合收益也含分红）；同时给出复原基准和官方指数两个口径 |

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
  factor_test.py            RankIC、ICIR、分组、IC 衰减、自相关、因子筛选
  models.py                 ICIR 加权 / Ridge / GBDT 的 walk-forward 训练
  risk.py                   行业 + 6 风格因子的 WLS 截面回归，EWMA 协方差
  optimizer.py              cvxpy 均值-方差主动优化
  backtest.py               日频回测引擎
  portfolio.py              组合构建：优化器嵌入回测，每月以实际持仓为起点
  holdings.py               导出每月持仓清单与最新组合报告
  html_report.py            生成交互式可视化报告（模板 report_template.html，纯 JS 画图，离线可用）
  metrics.py, attribution.py  绩效指标、Brinson 与因子归因
scripts/01 … 08             八个流程步骤（compare_runs.py 用于新旧版本对比）
tests/                      13 个单元测试（前视、标签可用日期、时点、成本、涨跌停、实际持仓、报告）
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
| 0 | `pytest tests` | 13 个单元测试（前视、标签可用日期、时点财报、成本、涨跌停、实际持仓、报告） | 几秒 |
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
- `portfolio`：`max_active_weight`（单票主动权重上限）、`industry_dev`、`size_dev`、`turnover_limit`、`topn`；
- `costs`：佣金、印花税、滑点。

只改了组合或成本参数时，不用重跑前面的步骤：
```bash
.venv/bin/python scripts/05_portfolio_backtest.py && .venv/bin/python scripts/06_attribution_report.py
```

### 更新到最新数据
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
- 财报首次披露日是估计值。
- ST 状态是推断值。
- 官方中证800的历史成分权重无法免费获取，主基准是自定义流通市值基准，每年和官方指数相差 2–5%。
- 冲击成本用平方根模型估算，系数没有用真实成交校准。
- 组合结果对样本内的边缘选择（λ、模型、门槛附近的因子）敏感，主方案的 IR 0.91 在合理方案中偏高（报告 5.6、5.7 节）。

详见报告第 1.4 节和第 7 节。
