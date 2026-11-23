"""Facts and limitations bound to every report (R05): generated each run so text cannot drift from reality."""
from __future__ import annotations

import json

import pandas as pd

from .config import PROC, REPORTS


def limitations() -> list:
    """(item, status, detail). status: 'pass' | 'not_strict' | 'limited'."""
    audit = {}
    p = PROC / "audit.json"
    if p.exists():
        audit = json.load(open(p))
    chk = json.load(open(PROC / "data_checks.json")) if (PROC / "data_checks.json").exists() else {}
    pct = lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "-"
    rows = [
        ("财报时点 (PIT)", "not_strict",
         "未严格 PIT: 供应商只给'最新公告日', 91.5% 的记录按法定截止日对齐; 数值可能是修订后的; 逾期披露的公司存在前视。"
         "对照: 只用量价因子的方案见稳健性报告。"),
        ("行业分类", "not_strict", "用的是当前的证监会行业, 不是历史时点数据; 行业中性化、行业约束和行业归因都受影响。"),
        ("ST 状态", "limited", f"由价格推断, 对照真实 ST: 精确率 {pct(audit.get('st_precision'))}, 召回率 {pct(audit.get('st_recall'))}; "
                               "参数在同一样本上选择, 没有独立持出样本。"),
        ("退市股票", "limited", f"部分已退市成分股拿不到行情 (基准覆盖率平均 {pct(chk.get('bench_coverage_mean'))}, 最低 {pct(chk.get('bench_coverage_min'))}); "
                               "组合和基准都无法包含它们。长期不可交易的持仓会被列为审计异常, 不当作零收益资产。"),
        ("基准", "limited", "自定义流通市值基准 (中证800成分, 前一交易日流通市值加权), 不是官方中证800; 官方全收益只作副口径。"),
        ("历史验证期", "limited", "2022-01 ~ 2026-10 在 v1-v6 的开发中被反复查看, 是历史回测, 不是一次性样本外。"),
        ("统计显著性", "limited", "样本外 IR 的 Bootstrap 95% 区间跨 0; 因子筛选存在多重检验 (见 HAC t 与 FDR q 值); 不能据此说'信号真实'或'可实盘赚钱'。"),
        ("成交假设", "limited", "按次日 VWAP 成交, 不建模最小交易单位 (100 股)、券商最低佣金、价格跳空和分多日成交; 冲击成本系数是主观假设。"),
        ("残差归因", "limited", "'个股特异'是风险模型解释不了的残差, 含未建模的风格、现金、交易时点和估计误差, 不等于选股能力。"),
    ]
    return rows


def limitations_md() -> str:
    lab = {"pass": "通过", "not_strict": "**未严格通过**", "limited": "有局限"}
    df = pd.DataFrame([{"项目": a, "状态": lab[b], "说明": c} for a, b, c in limitations()])
    return df.to_markdown(index=False)
