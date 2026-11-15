"""The HTML report: template has the data slot, and the selection reasons mirror select_factors."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.html_report import TEMPLATE, _selection_reasons


def test_template_has_data_slot():
    assert TEMPLATE.read_text(encoding="utf-8").count("/*__DATA__*/null") == 1


def test_selection_reasons():
    tab = pd.DataFrame({"t值": [3.0, 2.5, 1.5, 2.2], "ICIR(年化)": [1.0, 0.9, 0.4, 0.3]}, index=list("abcd"))
    corr = pd.DataFrame(0.0, index=list("abcd"), columns=list("abcd"))
    corr.loc["b", "a"] = corr.loc["a", "b"] = 0.8
    why = _selection_reasons(tab, ["a"], corr)
    assert why["a"] == "入选"
    assert "相关性 0.80" in why["b"]
    assert why["c"].startswith("|t|")
    assert why["d"].startswith("|ICIR|")
