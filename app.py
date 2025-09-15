# -*- coding: utf-8 -*-
"""
💸 Streamlit — DCF Valuation (FCFF / FCFE) — Cash‑Only Focus
هدف التطبيق: تقييم الشركة اعتماداً فقط على التدفقات النقدية المستقبلية
(DCF عبر FCFF أو FCFE) وحساب تكلفة رأس المال (WACC/CAPM) مع حساسية وسيناريوهات.

تشغيل محلي:
    streamlit run dcf_app.py
اعتماديات:
    pip install streamlit yfinance pandas numpy python-dateutil

ملاحظات مهمة:
- نعتمد بيانات Yahoo Finance عبر yfinance؛ قد تختلف الأسماء/التوافر حسب الشركة.
- الحسابات تقريبية تعليمية: يُفضَّل مراجعة التقارير السنوية/الربع سنوية لتدقيق البنود.
- إذا كانت r <= g_T (معدل الخصم ≤ نمو نهائي) سيتم تحذيرك لأن القيمة تصبح غير محدودة.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dateutil.relativedelta import relativedelta

# =============================
# إعداد واجهة RTL + تنسيق بصري بسيط
# =============================
st.set_page_config(page_title="💸 DCF — FCFF/FCFE | Cost of Capital", layout="wide")
CSS = """
<style>
:root, html, body, .stApp { direction: rtl; }
.stApp { text-align: right; font-family: -apple-system, Segoe UI, Tahoma, Arial, sans-serif; }
* { letter-spacing: 0.1px; }
label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, .stTextArea { text-align: right; }
.hero { background: linear-gradient(90deg,#f0f9ff,#ecfeff); border:1px solid #e2e8f0; border-radius: 16px; padding: 14px 18px; margin-bottom: 10px; }
.hero h1{ margin:0; font-size:22px; }
.kpi{ background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:12px; }
.kpi .title{ color:#64748b; font-size:13px; }
.kpi .value{ font-size:20px; font-weight:700; }
.kpi.ok .value{ color:#059669; } .kpi.mid .value{ color:#d97706; } .kpi.bad .value{ color:#dc2626; }
.small{ color:#475569; font-size:12px; }
.table-note{ color:#475569; font-size:12px; margin-top:-8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================
# مفاتيح/دوال مساعدة
# =============================
REV_KEYS = ["Total Revenue","Revenue","TotalRevenue","Sales"]
EBIT_KEYS = ["EBIT","Operating Income","OperatingIncome"]
EBITDA_KEYS = ["EBITDA"]
NI_KEYS = ["Net Income","NetIncome","Net Income Common Stockholders"]
TAX_EXP_KEYS = ["Income Tax Expense","Tax Provision","Provision For Income Taxes"]
PBT_KEYS = ["Income Before Tax","Pretax Income","Earnings Before Tax"]
INT_EXP_KEYS = ["Interest Expense"]

CFO_KEYS = ["Operating Cash Flow","Total Cash From Operating Activities"]
CAPEX_KEYS = ["Capital Expenditure","Capital Expenditures"]
DA_KEYS = ["Depreciation","Depreciation & amortization","Depreciation Amortization"]
WC_CHANGE_KEYS = ["Change In Working Capital","Change in Working Capital"]

TA_KEYS = ["Total Assets","TotalAssets"]
TE_KEYS = ["Total Stockholder Equity","Total Shareholder Equity","Total Stockholders Equity"]
CA_KEYS = ["Total Current Assets","Current Assets","TotalCurrentAssets"]
CL_KEYS = ["Total Current Liabilities","Current Liabilities","TotalCurrentLiabilities"]
CASH_KEYS = ["Cash And Cash Equivalents","Cash And Cash Equivalents, And Short Term Investments","Cash"]
STI_KEYS = ["Short Term Investments"]
CUR_DEBT_KEYS = ["Current Debt"]
LTD_KEYS = ["Long Term Debt"]
SLTD_KEYS = ["Short Long Term Debt"]
TOT_DEBT_KEYS = ["Total Debt"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _index_map(df: pd.DataFrame) -> Dict[str, str]:
    return { _norm(i): i for i in df.index.astype(str) }


def _find(df: pd.DataFrame, keys: List[str], col) -> float:
    if df is None or df.empty or col is None:
        return np.nan
    idx = _index_map(df)
    for k in keys:
        kk = _norm(k)
        if kk in idx:
            try:
                return float(df.loc[idx[kk], col])
            except Exception:
                try:
                    return float(pd.to_numeric(df.loc[idx[kk], col], errors="coerce"))
                except Exception:
                    return np.nan
    return np.nan


def _cols(df: pd.DataFrame, reverse: bool=True):
    try:
        return sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=reverse)
    except Exception:
        cols = list(df.columns)
        return cols[::-1] if reverse else cols


def _safe_div(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def _fmt_num(x, d: int = 2):
    if x is None or pd.isna(x):
        return "—"
    ax = abs(float(x))
    if ax >= 1_000_000_000_000: return f"{x/1_000_000_000_000:.{d}f}T"
    if ax >= 1_000_000_000: return f"{x/1_000_000_000:.{d}f}B"
    if ax >= 1_000_000: return f"{x/1_000_000:.{d}f}M"
    if ax >= 1_000: return f"{x/1_000:.{d}f}K"
    return f"{x:.{d}f}"


def _fmt_pct(x, d: int = 2):
    return "—" if x is None or pd.isna(x) else f"{100*float(x):.{d}f}%"


# مساعد تنسيق إضافي: نسبة على شكل x-times

def _fmt_x(x, d: int = 2):
    return "—" if x is None or pd.isna(x) else f"{float(x):.{d}f}x"

# =============================
# تحميل البيانات (كاش)
# =============================
@st.cache_data(ttl=1800)
def load_data(symbol: str) -> Dict[str, pd.DataFrame]:
    t = yf.Ticker(symbol)

    def get_df(fn):
        try:
            df = fn()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    inc_a = get_df(lambda: t.financials)
    inc_q = get_df(lambda: t.quarterly_financials)
    bal_a = get_df(lambda: t.balance_sheet)
    bal_q = get_df(lambda: t.quarterly_balance_sheet)
    cf_a  = get_df(lambda: t.cashflow)
    cf_q  = get_df(lambda: t.quarterly_cashflow)

    info = {}
    try:
        # get_info() قد يُرمي تحذيرات؛ نستخدمها بحذر
        data_info = {}
        try:
            data_info = t.get_info()
        except Exception:
            data_info = getattr(t, "info", {}) or {}
        if isinstance(data_info, dict):
            for f in [
                "longName","industry","sector","country","financialCurrency","beta","beta3Year",
                "website","longBusinessSummary","fullTimeEmployees"
            ]:
                info[f] = data_info.get(f)
    except Exception:
        pass

    price = shares = mcap = np.nan
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        shares = float(fi.get("shares", np.nan))
        mcap = float(fi.get("market_cap", np.nan))
    except Exception:
        pass

    return {
        "inc_a": inc_a, "inc_q": inc_q,
        "bal_a": bal_a, "bal_q": bal_q,
        "cf_a": cf_a,   "cf_q": cf_q,
        "info": info, "price": price, "shares": shares, "mcap": mcap,
    }


# =============================
# استخراج TTM ومشتقاتها
# =============================

def _sum_last4(df: pd.DataFrame, keys: List[str]) -> float:
    if df is None or df.empty:
        return np.nan
    cols = _cols(df)[:4]
    vals = [ _find(df, keys, c) for c in cols ]
    vals = [ v for v in vals if not pd.isna(v) ]
    return sum(vals) if vals else np.nan


def compute_ttm_blocks(d: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    inc_q, cf_q = d["inc_q"], d["cf_q"]
    ebit = _sum_last4(inc_q, EBIT_KEYS)
    ni   = _sum_last4(inc_q, NI_KEYS)
    pbt  = _sum_last4(inc_q, PBT_KEYS)
    tax  = _sum_last4(inc_q, TAX_EXP_KEYS)
    cfo  = _sum_last4(cf_q, CFO_KEYS)
    capex= _sum_last4(cf_q, CAPEX_KEYS)
    da   = _sum_last4(cf_q, DA_KEYS)
    wc_chg = _sum_last4(cf_q, WC_CHANGE_KEYS)
    return {
        "EBIT_TTM": ebit, "NI_TTM": ni, "PBT_TTM": pbt, "TAX_TTM": tax,
        "CFO_TTM": cfo, "CAPEX_TTM": capex, "DA_TTM": da, "WC_CHG_TTM": wc_chg
    }


def compute_latest_balance(d: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    bal_q, bal_a = d["bal_q"], d["bal_a"]
    bal = bal_q if not bal_q.empty else bal_a
    cols = _cols(bal)
    cur = cols[0] if cols else None
    prev= cols[1] if len(cols) > 1 else None

    total_assets = _find(bal, TA_KEYS, cur)
    total_equity = _find(bal, TE_KEYS, cur)
    cash = _find(bal, CASH_KEYS, cur)
    sti  = _find(bal, STI_KEYS, cur)

    # الدين الإجمالي
    total_debt = _find(bal, TOT_DEBT_KEYS, cur)
    if pd.isna(total_debt):
        components = [ _find(bal, LTD_KEYS, cur), _find(bal, SLTD_KEYS, cur), _find(bal, CUR_DEBT_KEYS, cur) ]
        components = [x for x in components if not pd.isna(x)]
        total_debt = sum(components) if components else np.nan

    # NWC (Working Capital) = (CA - Cash - STI) - (CL - CurrentDebt)
    ca = _find(bal, CA_KEYS, cur)
    cl = _find(bal, CL_KEYS, cur)
    cur_debt = _find(bal, CUR_DEBT_KEYS, cur)
    nwc = np.nan
    if not any(pd.isna(x) for x in [ca, cl]):
        nwc = (ca - (0 if pd.isna(sti) else sti) - (0 if pd.isna(cash) else cash)) - (cl - (0 if pd.isna(cur_debt) else cur_debt))

    # التغير في NWC عبر الميزانية (بديل إذا لم يتوفر من CF)
    ca_prev = _find(bal, CA_KEYS, prev)
    cl_prev = _find(bal, CL_KEYS, prev)
    cash_prev = _find(bal, CASH_KEYS, prev)
    sti_prev = _find(bal, STI_KEYS, prev)
    cur_debt_prev = _find(bal, CUR_DEBT_KEYS, prev)
    nwc_prev = np.nan
    if not any(pd.isna(x) for x in [ca_prev, cl_prev]):
        nwc_prev = (ca_prev - (0 if pd.isna(sti_prev) else sti_prev) - (0 if pd.isna(cash_prev) else cash_prev)) - (cl_prev - (0 if pd.isna(cur_debt_prev) else cur_debt_prev))
    dNWC_bs = np.nan if any(pd.isna(x) for x in [nwc, nwc_prev]) else (nwc - nwc_prev)

    # تغير الدين (Net Borrowings تقريباً)
    total_debt_prev = _find(bal, TOT_DEBT_KEYS, prev)
    if pd.isna(total_debt_prev):
        comps_prev = [ _find(bal, LTD_KEYS, prev), _find(bal, SLTD_KEYS, prev), _find(bal, CUR_DEBT_KEYS, prev) ]
        comps_prev = [x for x in comps_prev if not pd.isna(x)]
        total_debt_prev = sum(comps_prev) if comps_prev else np.nan
    net_borrow = np.nan if any(pd.isna(x) for x in [total_debt, total_debt_prev]) else (total_debt - total_debt_prev)

    return {
        "TA": total_assets, "TE": total_equity, "Cash": cash, "STI": sti,
        "TotalDebt": total_debt, "NWC": nwc, "dNWC_BS": dNWC_bs, "NetBorrowings_BS": net_borrow
    }


# =============================
# تكلفة رأس المال (WACC/CAPM) و FCFs
# =============================
@dataclass
class CapitalCosts:
    re: float  # Cost of Equity
    rd: float  # Pre-tax Cost of Debt
    tax_rate: float
    we: float  # Equity weight
    wd: float  # Debt weight
    wacc: float


def estimate_tax_rate(ttm: Dict[str, float]) -> float:
    pbt, tax = ttm.get("PBT_TTM"), ttm.get("TAX_TTM")
    if pbt is None or tax is None or pd.isna(pbt) or pbt == 0 or pd.isna(tax):
        return 0.25
    est = float(tax) / float(pbt)
    return float(np.clip(est, 0.0, 0.5))


def estimate_rd(inc_q: pd.DataFrame, bal_q: pd.DataFrame, tax_rate: float) -> float:
    cols_i = _cols(inc_q)
    cols_b = _cols(bal_q)
    if not cols_i or not cols_b:
        return 0.08  # بديل محافظ
    int_exp = _find(inc_q, INT_EXP_KEYS, cols_i[0])
    debt_cur = _find(bal_q, TOT_DEBT_KEYS, cols_b[0])
    debt_prev= _find(bal_q, TOT_DEBT_KEYS, cols_b[1]) if len(cols_b)>1 else np.nan
    avg_debt = np.nanmean([debt_cur, debt_prev])
    rd = _safe_div(abs(int_exp) if not pd.isna(int_exp) else np.nan, avg_debt)
    if pd.isna(rd) or rd<=0 or rd>0.25:
        rd = 0.08  # بديل معقول
    return float(rd)


def build_capital_costs(d: Dict[str, pd.DataFrame], ttm: Dict[str, float], mcap: float, debt: float,
                        rf: float, erp: float, beta_opt: Optional[float], tax_override: Optional[float],
                        rd_override: Optional[float]) -> CapitalCosts:
    info = d.get("info", {}) or {}
    beta = beta_opt if (beta_opt is not None and not pd.isna(beta_opt) and beta_opt>0) else None
    if beta is None:
        beta = info.get("beta3Year") or info.get("beta")
    try:
        beta = float(beta)
        if pd.isna(beta) or beta<=0: beta = 1.0
    except Exception:
        beta = 1.0

    re = rf + beta * erp

    tax_rate = tax_override if (tax_override is not None) else estimate_tax_rate(ttm)
    rd = rd_override if (rd_override is not None) else estimate_rd(d["inc_q"], d["bal_q"], tax_rate)

    E = mcap if (mcap is not None and not pd.isna(mcap) and mcap>0) else np.nan
    D = debt if (debt is not None and not pd.isna(debt) and debt>0) else 0.0
    V = (0 if pd.isna(E) else E) + D
    we = _safe_div(E, V)
    wd = _safe_div(D, V)
    if pd.isna(we) or pd.isna(wd):
        # fallback: 80/20
        we, wd = 0.8, 0.2

    wacc = we*re + wd*rd*(1 - tax_rate)
    return CapitalCosts(re=re, rd=rd, tax_rate=tax_rate, we=we, wd=wd, wacc=wacc)


@dataclass
class FCFPack:
    fcff_ttm: Optional[float]
    fcfe_ttm: Optional[float]
    cfo_ttm: Optional[float]
    capex_ttm: Optional[float]
    dNWC_ttm: Optional[float]
    net_borrow: Optional[float]


def compute_fcf(d: Dict[str, pd.DataFrame]) -> FCFPack:
    ttm = compute_ttm_blocks(d)
    bal = compute_latest_balance(d)

    # dNWC: من التدفق النقدي إن وُجد وإلا من الميزانية
    dNWC_cf = ttm.get("WC_CHG_TTM")
    dNWC_bs = bal.get("dNWC_BS")
    dNWC = dNWC_cf if (dNWC_cf is not None and not pd.isna(dNWC_cf)) else dNWC_bs

    ebit = ttm.get("EBIT_TTM")
    tax_rate = estimate_tax_rate(ttm)
    nopat = None if ebit is None or pd.isna(ebit) else ebit * (1 - tax_rate)

    da = ttm.get("DA_TTM")
    capex = ttm.get("CAPEX_TTM")

    # FCFF = NOPAT + DA – CAPEX – ΔNWC
    fcff = np.nan
    if not any(pd.isna(x) for x in [nopat, da, capex, dNWC]):
        fcff = nopat + da - capex - dNWC

    # بديل بسيط: FCFF ≈ CFO – CAPEX + Interest*(1-T) (إذا رغبت)
    cfo = ttm.get("CFO_TTM")
    int_exp = _find(d["inc_q"], INT_EXP_KEYS, _cols(d["inc_q"])[0]) if not d["inc_q"].empty else np.nan
    alt_fcff = None if any(pd.isna(x) for x in [cfo, capex, int_exp]) else (cfo - capex + abs(int_exp)*(1 - tax_rate))
    if pd.isna(fcff) and alt_fcff is not None:
        fcff = alt_fcff

    # Net Borrowings
    net_borrow = bal.get("NetBorrowings_BS")

    # FCFE = CFO – CAPEX + NetBorrowings
    fcfe = None if any(pd.isna(x) for x in [cfo, capex, net_borrow]) else (cfo - capex + net_borrow)

    return FCFPack(
        fcff_ttm=fcff,
        fcfe_ttm=fcfe,
        cfo_ttm=cfo,
        capex_ttm=capex,
        dNWC_ttm=dNWC,
        net_borrow=net_borrow,
    )


# =============================
# DCF — إسقاطات وتقييم
# =============================
@dataclass
class DCFInputs:
    mode: str  # "FCFF" أو "FCFE"
    base_fcf: float
    discount_rate: float
    growth_years: int
    growth_rate: float
    terminal_method: str  # "Perpetuity" أو "Exit Multiple"
    terminal_growth: float
    exit_multiple: Optional[float]


def project_and_value(inputs: DCFInputs) -> Tuple[float, pd.DataFrame]:
    if inputs.base_fcf is None or pd.isna(inputs.base_fcf):
        return np.nan, pd.DataFrame()

    if inputs.terminal_method == "Perpetuity" and inputs.discount_rate <= inputs.terminal_growth:
        return np.nan, pd.DataFrame()

    flows = []
    pv_sum = 0.0
    fcf = float(inputs.base_fcf)

    for t in range(1, inputs.growth_years + 1):
        fcf = fcf * (1 + inputs.growth_rate)
        pv = fcf / ((1 + inputs.discount_rate) ** t)
        pv_sum += pv
        flows.append({"السنة": t, "FCF": fcf, "القيمة الحالية": pv})

    # القيمة النهائية
    if inputs.terminal_method == "Perpetuity":
        tv = fcf * (1 + inputs.terminal_growth) / (inputs.discount_rate - inputs.terminal_growth)
    else:
        # Exit multiple على FCF (افتراضي محافظ)
        mult = inputs.exit_multiple if (inputs.exit_multiple and inputs.exit_multiple>0) else 12.0
        tv = fcf * mult

    pv_tv = tv / ((1 + inputs.discount_rate) ** inputs.growth_years)
    flows.append({"السنة": "القيمة النهائية", "FCF": tv, "القيمة الحالية": pv_tv})

    total_pv = pv_sum + pv_tv
    return total_pv, pd.DataFrame(flows)


# =============================
# حساسية 5×5 للقيمة/السهم
# =============================

def build_sensitivity(base_fcf: float, 
                      disc_grid: List[float], 
                      term_growth_grid: List[float], 
                      years: int, growth: float, 
                      mode: str,
                      shares: Optional[float],
                      debt: Optional[float], cash: Optional[float]) -> pd.DataFrame:
    rows = []
    for r in disc_grid:
        row = {"r\\g": f"{100*r:.1f}%"}
        for g in term_growth_grid:
            if r <= g:
                row[f"g={100*g:.1f}%"] = np.nan
                continue
            total, _ = project_and_value(DCFInputs(mode=mode, base_fcf=base_fcf, discount_rate=r,
                                                   growth_years=years, growth_rate=growth,
                                                   terminal_method="Perpetuity", terminal_growth=g, exit_multiple=None))
            equity = np.nan
            if mode == "FCFF":
                if total is not None and not pd.isna(total):
                    enterprise = total
                    net_debt = (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash)
                    equity = enterprise - net_debt
            else:
                equity = total
            ps = np.nan if (shares is None or pd.isna(shares) or shares<=0 or equity is None or pd.isna(equity)) else (equity / shares)
            row[f"g={100*g:.1f}%"] = ps
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# =============================
# لوحة نسب شبيهة بالصورة — دوال المشتقات السنوية + بناء الجدول
# =============================

def compute_annual_blocks(d: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    inc_a, cf_a, bal_a = d["inc_a"], d["cf_a"], d["bal_a"]
    cols = _cols(inc_a) if not inc_a.empty else []
    if not cols:
        cols = _cols(cf_a) if not cf_a.empty else []
    if not cols:
        cols = _cols(bal_a) if not bal_a.empty else []
    cols = cols[:2]
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        rev   = _find(inc_a, REV_KEYS, c)
        ebit  = _find(inc_a, EBIT_KEYS, c)
        ni    = _find(inc_a, NI_KEYS, c)
        intex = _find(inc_a, INT_EXP_KEYS, c)
        cfo   = _find(cf_a, CFO_KEYS, c)
        capex = _find(cf_a, CAPEX_KEYS, c)
        da    = _find(cf_a, DA_KEYS, c)
        dNWC  = _find(cf_a, WC_CHANGE_KEYS, c)
        ca    = _find(bal_a, CA_KEYS, c)
        cl    = _find(bal_a, CL_KEYS, c)
        te    = _find(bal_a, TE_KEYS, c)
        cash  = _find(bal_a, CASH_KEYS, c)
        debt  = _find(bal_a, TOT_DEBT_KEYS, c)
        if pd.isna(debt):
            parts = [_find(bal_a, LTD_KEYS, c), _find(bal_a, SLTD_KEYS, c), _find(bal_a, CUR_DEBT_KEYS, c)]
            parts = [x for x in parts if not pd.isna(x)]
            debt = sum(parts) if parts else np.nan
        out[str(c)] = {
            "REV": rev, "EBIT": ebit, "NI": ni, "INTEXP": intex,
            "CFO": cfo, "CAPEX": capex, "DA": da, "dNWC": dNWC,
            "CA": ca, "CL": cl, "TE": te, "Cash": cash, "Debt": debt
        }
    return out


def build_cash_ratios_table(annual: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    years = list(annual.keys())
    y0 = years[0] if years else None
    y1 = years[1] if len(years) > 1 else None

    def metrics_for_year(y: Optional[str]) -> Dict[str, Optional[float]]:
        if not y or y not in annual: return {}
        a = annual[y]
        fcff = np.nan
        if not any(pd.isna(x) for x in [a.get("CFO"), a.get("CAPEX")]):
            fcff = a.get("CFO") - a.get("CAPEX")
        cur = _safe_div(a.get("CA"), a.get("CL"))
        de  = _safe_div(a.get("Debt"), a.get("TE"))
        cov = _safe_div(a.get("EBIT"), abs(a.get("INTEXP")) if a.get("INTEXP") is not None else np.nan)
        cfo_ni = _safe_div(a.get("CFO"), a.get("NI"))
        capex_ocf = _safe_div(a.get("CAPEX"), a.get("CFO"))
        fcff_margin = _safe_div(fcff, a.get("REV"))
        netdebt = np.nan if a.get("Debt") is None else (a.get("Debt") - (0 if pd.isna(a.get("Cash")) else a.get("Cash")))
        nd_fcff = _safe_div(netdebt, fcff)
        return {
            "CurrentRatio": cur,
            "D_to_E": de,
            "IntCoverage": cov,
            "CFO_to_NI": cfo_ni,
            "Capex_to_OCF": capex_ocf,
            "FCFF_Margin": fcff_margin,
            "NetDebt_to_FCFF": nd_fcff,
            "FCFF": fcff
        }

    m0, m1 = metrics_for_year(y0), metrics_for_year(y1)

    def judge(val: Optional[float], kind: str) -> str:
        if val is None or pd.isna(val):
            return "—"
        v = float(val)
        if kind == "CurrentRatio":
            return "✅" if v >= 1.5 else ("⚠️" if v >= 1.0 else "❌")
        if kind == "D_to_E":
            return "✅" if v <= 0.5 else ("⚠️" if v <= 1.0 else "❌")
        if kind == "IntCoverage":
            return "✅" if v >= 10 else ("⚠️" if v >= 6 else "❌")
        if kind == "CFO_to_NI":
            return "✅" if v >= 1.0 else ("⚠️" if v >= 0.8 else "❌")
        if kind == "Capex_to_OCF":
            return "✅" if v <= 0.4 else ("⚠️" if v <= 0.6 else "❌")
        if kind == "FCFF_Margin":
            return "✅" if v >= 0.08 else ("⚠️" if v >= 0.05 else "❌")
        if kind == "NetDebt_to_FCFF":
            return "✅" if v <= 2.0 else ("⚠️" if v <= 3.0 else "❌")
        return "—"

    rows = []
    def add_row(name, explain, key, target, fmt="pct"):
        v0 = m0.get(key) if m0 else np.nan
        v1 = m1.get(key) if m1 else np.nan
        if fmt == "pct":
            a0 = _fmt_pct(v0)
            a1 = _fmt_pct(v1)
        elif fmt == "x":
            a0 = _fmt_x(v0)
            a1 = _fmt_x(v1)
        else:
            a0 = _fmt_num(v0)
            a1 = _fmt_num(v1)
        verdict = judge(v0, key)
        rows.append({
            "البند": name,
            "شرح النسبة": explain,
            str(y0 or "أحدث"): a0,
            (str(y1) if y1 else "السنة السابقة"): a1,
            "المعيار/المستهدف": target,
            "رأي فني": verdict
        })

    add_row("السيولة الجارية", "الأصول المتداولة ÷ الخصوم المتداولة", "CurrentRatio", "≥ 1.5", fmt="x")
    add_row("المديونية D/E", "إجمالي الدين ÷ حقوق الملكية", "D_to_E", "≤ 0.5 (≤1.0 مقبول)", fmt="x")
    add_row("تغطية الفوائد", "EBIT ÷ مصروف الفائدة", "IntCoverage", "≥ 10x (≥6x مقبول)", fmt="x")
    add_row("جودة الأرباح", "CFO ÷ صافي الربح", "CFO_to_NI", "≥ 1.0", fmt="x")
    add_row("كثافة الاستثمار", "Capex ÷ OCF", "Capex_to_OCF", "≤ 40%", fmt="pct")
    add_row("هامش FCFF", "(CFO−Capex) ÷ الإيراد", "FCFF_Margin", "≥ 8%", fmt="pct")
    add_row("صافي الدين/FCFF", "(الدين−النقد) ÷ FCFF", "NetDebt_to_FCFF", "≤ 2.0x", fmt="x")

    df = pd.DataFrame(rows)
    return df

# =============================
# واجهة المستخدم
# =============================
st.markdown("""
<div class="hero">
  <h1>💸 تقييم DCF (FCFF/FCFE) — فقط كاش، بلا ضوضاء</h1>
  <div class="small">خلّنا واقعيين: بدون تدفّق حر مقنع ما في قيمة مستدامة. هذا النموذج يشكّك أولًا، ثم يحسب.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي","السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"

    symbol_in = st.text_input("أدخل الرمز (واحد)", "AAPL" if market=="السوق الأمريكي" else "1120")
    if suffix and symbol_in and symbol_in.isalnum() and not symbol_in.endswith(".SR"):
        symbol = symbol_in.upper() + suffix
    else:
        symbol = (symbol_in or "").upper()

    st.markdown("---")
    st.markdown("#### إعدادات DCF")
    dcf_mode = st.radio("نموذج التدفق", ["FCFF","FCFE"], index=0, help="FCFF للمنشأة ← استخدم WACC. FCFE للمساهم ← استخدم Cost of Equity.")
    years = st.number_input("سنوات الإسقاط", 3, 10, 5, 1)
    growth = st.number_input("نمو FCF السنوي (المرحلة 1)", 0.00, 0.30, 0.05, 0.01)
    terminal_method = st.selectbox("طريقة القيمة النهائية", ["Perpetuity","Exit Multiple"], index=0)
    terminal_growth = st.number_input("نمو نهائي gₜ", 0.00, 0.05, 0.02, 0.005)
    exit_mult = None
    if terminal_method == "Exit Multiple":
        exit_mult = st.number_input("مضاعف الخروج على FCF", 5.0, 25.0, 12.0, 0.5)

    st.markdown("---")
    st.markdown("#### تكلفة رأس المال")
    rf = st.number_input("Risk‑Free (Rf)", 0.00, 0.15, 0.04 if market=="السوق الأمريكي" else 0.05, 0.005)
    erp = st.number_input("Equity Risk Premium (ERP)", 0.02, 0.15, 0.055 if market=="السوق الأمريكي" else 0.07, 0.005)
    beta_override = st.number_input("Beta (اختياري)", 0.00, 3.00, 0.00, 0.05, help="اتركه 0 لاستخدام بيتا من Yahoo إن توفرت.")
    tax_override_pct = st.number_input("معدل الضريبة (اختياري)", 0.00, 0.50, 0.00, 0.01)
    rd_override = st.number_input("تكلفة الدين قبل الضريبة (اختياري)", 0.00, 0.30, 0.00, 0.005)

    st.caption("ملاحظة: إن تركت الحقول الاختيارية = 0 سيُستنتج الحساب تلقائيًا.")

# زر التنفيذ
if st.button("🚀 قيّم الشركة"):
    if not symbol:
        st.warning("يرجى إدخال رمز صحيح.")
        st.stop()

    with st.spinner("جاري تحميل البيانات وحساب التدفقات وتكلفة رأس المال…"):
        data = load_data(symbol)
        ttm = compute_ttm_blocks(data)
        bal = compute_latest_balance(data)
        fcf = compute_fcf(data)

        # تكلفة رأس المال
        tax_override = tax_override_pct if tax_override_pct>0 else None
        beta_opt = beta_override if beta_override>0 else None
        rd_opt = rd_override if rd_override>0 else None
        cc = build_capital_costs(
            d=data, ttm=ttm, mcap=data.get("mcap", np.nan), debt=bal.get("TotalDebt"),
            rf=rf, erp=erp, beta_opt=beta_opt, tax_override=tax_override, rd_override=rd_opt
        )

        # اختيار FCF الأساسي و معدل الخصم المناسب
        if dcf_mode == "FCFF":
            base_fcf = fcf.fcff_ttm
            disc = cc.wacc
        else:
            base_fcf = fcf.fcfe_ttm
            disc = cc.re

        d_inputs = DCFInputs(
            mode=dcf_mode, base_fcf=base_fcf, discount_rate=disc,
            growth_years=int(years), growth_rate=growth,
            terminal_method=terminal_method, terminal_growth=terminal_growth,
            exit_multiple=exit_mult
        )
        total_pv, flows_df = project_and_value(d_inputs)

        # اشتقاق قيمة حقوق الملكية/السهم
        price = data.get("price", np.nan)
        shares = data.get("shares", np.nan)
        cash = bal.get("Cash")
        debt = bal.get("TotalDebt")

        equity_value = np.nan
        if dcf_mode == "FCFF":
            enterprise_value = total_pv
            if enterprise_value is not None and not pd.isna(enterprise_value):
                net_debt = (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash)
                equity_value = enterprise_value - net_debt
        else:
            equity_value = total_pv

        fair_ps = np.nan if (shares is None or pd.isna(shares) or shares<=0 or equity_value is None or pd.isna(equity_value)) else (equity_value / shares)
        upside = None if (price is None or pd.isna(price) or price<=0 or fair_ps is None or pd.isna(fair_ps)) else (fair_ps/price - 1)

    # ===== KPIs =====
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi ok'><div class='title'>نموذج التدفق</div><div class='value'>{dcf_mode}</div><div class='small'>FCFF=للمنشأة • FCFE=للمساهم</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi {'ok' if (base_fcf is not None and not pd.isna(base_fcf) and base_fcf>0) else 'bad'}'><div class='title'>الـ FCF (TTM)</div><div class='value'>{_fmt_num(base_fcf)}</div><div class='small'>قاعدة الإسقاط</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi mid'><div class='title'>معدل الخصم</div><div class='value'>{_fmt_pct(disc)}</div><div class='small'>{'WACC' if dcf_mode=='FCFF' else 'Cost of Equity (CAPM)'}</div></div>", unsafe_allow_html=True)
    with c4:
        color = 'ok' if (upside is not None and upside>0.15) else ('mid' if upside and upside>0 else 'bad')
        st.markdown(f"<div class='kpi {color}'><div class='title'>القيمة العادلة/سهم</div><div class='value'>{_fmt_num(fair_ps)}</div><div class='small'>مقابل السعر الحالي {_fmt_num(price)}</div></div>", unsafe_allow_html=True)

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.markdown(f"<div class='kpi'><div class='title'>Cost of Equity (Re)</div><div class='value'>{_fmt_pct(cc.re)}</div><div class='small'>Rf + β×ERP</div></div>", unsafe_allow_html=True)
    with cc2: st.markdown(f"<div class='kpi'><div class='title'>Cost of Debt (Rd)</div><div class='value'>{_fmt_pct(cc.rd)}</div><div class='small'>قبل الضريبة</div></div>", unsafe_allow_html=True)
    with cc3: st.markdown(f"<div class='kpi'><div class='title'>Tax Rate</div><div class='value'>{_fmt_pct(cc.tax_rate)}</div><div class='small'>مُستنتج/اختياري</div></div>", unsafe_allow_html=True)
    with cc4: st.markdown(f"<div class='kpi'><div class='title'>الأوزان E/D</div><div class='value'>{_fmt_pct(cc.we)} / {_fmt_pct(cc.wd)}</div><div class='small'>WACC={_fmt_pct(cc.wacc)}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 تدفقات السنوات + القيمة الحالية")
    if flows_df is not None and not flows_df.empty:
        st.dataframe(flows_df, use_container_width=True)
    else:
        st.info("لم نستطع احتساب الجداول (تحقق من المدخلات).")

    st.markdown("---")
    st.subheader("🧪 حساسية القيمة العادلة/سهم — شبكة r × gₜ")
    r_grid = [max(0.03, disc + x) for x in [-0.02,-0.01,0.0,0.01,0.02]]
    g_grid = [max(0.00, terminal_growth + x) for x in [-0.01,-0.005,0.0,0.005,0.01]]
    sens = build_sensitivity(base_fcf=base_fcf, disc_grid=r_grid, term_growth_grid=g_grid,
                             years=int(years), growth=growth, mode=dcf_mode,
                             shares=shares, debt=debt, cash=cash)
    st.dataframe(sens, use_container_width=True)
    st.caption("الخلايا الفارغة تشير إلى حالة غير صالحة (r ≤ gₜ).")

    st.markdown("---")
    st.subheader("🧾 تفصيل TTM وملخص الميزانية")
    ttm_rows = [{
        "EBIT_TTM": _fmt_num(ttm.get("EBIT_TTM")),
        "NI_TTM": _fmt_num(ttm.get("NI_TTM")),
        "CFO_TTM": _fmt_num(ttm.get("CFO_TTM")),
        "CAPEX_TTM": _fmt_num(ttm.get("CAPEX_TTM")),
        "DA_TTM": _fmt_num(ttm.get("DA_TTM")),
        "ΔNWC_TTM": _fmt_num(fcf.dNWC_ttm),
        "NetBorrowings": _fmt_num(fcf.net_borrow),
        "FCFF_TTM": _fmt_num(fcf.fcff_ttm),
        "FCFE_TTM": _fmt_num(fcf.fcfe_ttm),
    }]
    st.dataframe(pd.DataFrame(ttm_rows), use_container_width=True)

    bal_rows = [{
        "Cash": _fmt_num(bal.get("Cash")),
        "TotalDebt": _fmt_num(bal.get("TotalDebt")),
        "NetDebt": _fmt_num((0 if pd.isna(bal.get('TotalDebt')) else bal.get('TotalDebt')) - (0 if pd.isna(bal.get('Cash')) else bal.get('Cash'))),
        "Equity(TE)": _fmt_num(bal.get("TE")),
        "NWC": _fmt_num(bal.get("NWC")),
    }]
    st.dataframe(pd.DataFrame(bal_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 لوحة نسب نقدية/تقييم (شكل مشابه للصورة)")
    try:
        annual = compute_annual_blocks(data)
        ratios_df = build_cash_ratios_table(annual)
        st.dataframe(ratios_df, use_container_width=True)
        st.caption("ملاحظة: الأعوام تمثل آخر سنتين سنويتين متاحتين في Yahoo؛ قد تختلف تواريخ الإقفال بين الشركات. القيم '—' تعني عدم توفر بيانات.")
    except Exception as e:
        st.info(f"تعذر بناء اللوحة لعدم كفاية البيانات: {e}")

    st.markdown("---")
    st.subheader("📜 مذكرة تقييم مختصرة")
    info = data.get("info", {})
    lines = []
    lines.append(f"**الشركة/الرمز:** {info.get('longName') or symbol} ({symbol}) — القطاع: {info.get('sector') or '—'} | الصناعة: {info.get('industry') or '—'}")
    lines.append(f"**منهجية:** {dcf_mode} • سنوات الإسقاط {int(years)} • نمو {_fmt_pct(growth)} • طريقة القيمة النهائية {terminal_method}{(' (gₜ='+_fmt_pct(terminal_growth)+')' if terminal_method=='Perpetuity' else ' (Mult='+str(exit_mult)+')')}.")
    lines.append(f"**تكلفة رأس المال:** Re={_fmt_pct(cc.re)} | Rd={_fmt_pct(cc.rd)} | Tax={_fmt_pct(cc.tax_rate)} | WACC={_fmt_pct(cc.wacc)} | We={_fmt_pct(cc.we)} | Wd={_fmt_pct(cc.wd)}.")
    lines.append(f"**TTM FCF:** FCFF={_fmt_num(fcf.fcff_ttm)} | FCFE={_fmt_num(fcf.fcfe_ttm)} | CFO={_fmt_num(fcf.cfo_ttm)} | Capex={_fmt_num(fcf.capex_ttm)} | ΔNWC={_fmt_num(fcf.dNWC_ttm)} | NetBorrowings={_fmt_num(fcf.net_borrow)}.")
    lines.append(f"**القيمة العادلة/سهم (تقريبية):** {_fmt_num(fair_ps)} مقابل السعر الحالي {_fmt_num(price)} → {_fmt_pct(upside) if upside is not None else '—'} عائد ضمني.")
    st.markdown("\n\n".join(lines))

    st.caption("هذا النموذج لأغراض تعليمية/بحثية — لا يعتبر توصية استثمارية. دقّق الفرضيات قبل اتخاذ قرار.")

else:
    st.info("ادخل الرمز واضبط الفرضيات ثم اضغط \"🚀 قيّم الشركة\".")
