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
# إضافات للمخزون/الذمم/تكلفة المبيعات
COGS_KEYS = ["Cost Of Revenue","Cost of Revenue","CostOfRevenue","COGS"]
AR_KEYS   = ["Net Receivables","Accounts Receivable","Receivables"]
AP_KEYS   = ["Accounts Payable","Payables"]
INV_KEYS  = ["Inventory","Inventory Net"]


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
    # نحدد الأعمدة السنوية المتاحة (الأحدث أولاً)
    cols = _cols(inc_a) if not inc_a.empty else []
    if not cols:
        cols = _cols(cf_a) if not cf_a.empty else []
    if not cols:
        cols = _cols(bal_a) if not bal_a.empty else []
    cols = cols[:2]  # آخر سنتين فقط
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        rev    = _find(inc_a, REV_KEYS, c)
        cogs   = _find(inc_a, COGS_KEYS, c)
        ebit   = _find(inc_a, EBIT_KEYS, c)
        ebitda = _find(inc_a, EBITDA_KEYS, c)
        if pd.isna(ebitda):
            da_temp = _find(cf_a, DA_KEYS, c)
            ebitda = (ebit + da_temp) if not any(pd.isna(x) for x in [ebit, da_temp]) else np.nan
        ni     = _find(inc_a, NI_KEYS, c)
        pbt    = _find(inc_a, PBT_KEYS, c)
        tax    = _find(inc_a, TAX_EXP_KEYS, c)
        intex  = _find(inc_a, INT_EXP_KEYS, c)

        cfo   = _find(cf_a, CFO_KEYS, c)
        capex = _find(cf_a, CAPEX_KEYS, c)
        da    = _find(cf_a, DA_KEYS, c)
        dNWC  = _find(cf_a, WC_CHANGE_KEYS, c)

        ta    = _find(bal_a, TA_KEYS, c)
        te    = _find(bal_a, TE_KEYS, c)
        ca    = _find(bal_a, CA_KEYS, c)
        cl    = _find(bal_a, CL_KEYS, c)
        cash  = _find(bal_a, CASH_KEYS, c)
        debt  = _find(bal_a, TOT_DEBT_KEYS, c)
        if pd.isna(debt):
            parts = [_find(bal_a, LTD_KEYS, c), _find(bal_a, SLTD_KEYS, c), _find(bal_a, CUR_DEBT_KEYS, c)]
            parts = [x for x in parts if not pd.isna(x)]
            debt = sum(parts) if parts else np.nan

        ar    = _find(bal_a, AR_KEYS, c)
        inv   = _find(bal_a, INV_KEYS, c)
        ap    = _find(bal_a, AP_KEYS, c)

        out[str(c)] = {
            "REV": rev, "COGS": cogs, "EBIT": ebit, "EBITDA": ebitda, "NI": ni, "PBT": pbt, "TAX": tax, "INTEXP": intex,
            "CFO": cfo, "CAPEX": capex, "DA": da, "dNWC": dNWC,
            "TA": ta, "TE": te, "CA": ca, "CL": cl, "Cash": cash, "Debt": debt,
            "AR": ar, "INV": inv, "AP": ap
        }
    return out


def build_cash_ratios_table(annual: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    years = list(annual.keys())
    y0 = years[0] if years else None  # الأحدث
    y1 = years[1] if len(years) > 1 else None  # السابق
    a0 = annual.get(y0, {}) if y0 else {}
    a1 = annual.get(y1, {}) if y1 else {}

    def nmean(*vals):
        vv = [float(v) for v in vals if v is not None and not pd.isna(v)]
        return float(np.mean(vv)) if vv else np.nan

    def compute(a_cur: Dict[str, float], a_prev: Dict[str, float]):
        rev = a_cur.get("REV"); cogs = a_cur.get("COGS"); ebit = a_cur.get("EBIT"); ebitda = a_cur.get("EBITDA")
        ni = a_cur.get("NI"); cfo = a_cur.get("CFO"); capex = a_cur.get("CAPEX"); intex = a_cur.get("INTEXP")
        ta = a_cur.get("TA"); te = a_cur.get("TE"); ca = a_cur.get("CA"); cl = a_cur.get("CL"); cash = a_cur.get("Cash"); debt = a_cur.get("Debt")
        ar = a_cur.get("AR"); inv = a_cur.get("INV"); ap = a_cur.get("AP")
        pbt = a_cur.get("PBT"); tax = a_cur.get("TAX")
        tax_rate = _safe_div(tax, pbt)
        if pd.isna(tax_rate) or tax_rate < 0 or tax_rate > 0.6:
            tax_rate = 0.25
        fcff = np.nan if any(pd.isna(x) for x in [cfo, capex]) else (cfo - capex)
        gross = np.nan if any(pd.isna(x) for x in [rev, cogs]) else (rev - cogs)

        # متوسطات لأغراض العوائد/الدوران
        ta_avg = nmean(ta, a_prev.get("TA") if a_prev else np.nan)
        te_avg = nmean(te, a_prev.get("TE") if a_prev else np.nan)
        ar_avg = nmean(ar, a_prev.get("AR") if a_prev else np.nan)
        inv_avg = nmean(inv, a_prev.get("INV") if a_prev else np.nan)
        ap_avg = nmean(ap, a_prev.get("AP") if a_prev else np.nan)

        invested = np.nan if any(pd.isna(x) for x in [debt, te, cash]) else (debt + te - cash)
        nopat = np.nan if pd.isna(ebit) else (ebit * (1 - tax_rate))

        # نسب
        cur = _safe_div(ca, cl)
        quick = _safe_div((None if pd.isna(ca) else ca) - (0 if pd.isna(inv) else inv), cl)
        cash_ratio = _safe_div(cash, cl)
        de = _safe_div(debt, te)
        icov = _safe_div(ebit, abs(intex) if intex is not None else np.nan)
        netdebt = np.nan if pd.isna(debt) else (debt - (0 if pd.isna(cash) else cash))
        nd_fcff = _safe_div(netdebt, fcff)
        nd_ebitda = _safe_div(netdebt, ebitda)

        gm = _safe_div(gross, rev)
        ebitda_m = _safe_div(ebitda, rev)
        ebit_m = _safe_div(ebit, rev)
        net_m = _safe_div(ni, rev)
        ocf_m = _safe_div(cfo, rev)
        fcff_m = _safe_div(fcff, rev)

        roa = _safe_div(ni, ta_avg)
        roe = _safe_div(ni, te_avg)
        roic = _safe_div(nopat, invested)

        at = _safe_div(rev, ta_avg)
        rt = _safe_div(rev, ar_avg)
        it = _safe_div(cogs, inv_avg)
        pt = _safe_div(cogs, ap_avg)
        dso = _safe_div(365.0, rt)
        dio = _safe_div(365.0, it)
        dpo = _safe_div(365.0, pt)
        ccc = (dso + dio - dpo) if not any(pd.isna(x) for x in [dso, dio, dpo]) else np.nan

        return {
            # سيولة/ملاءة
            "CurrentRatio": cur, "QuickRatio": quick, "CashRatio": cash_ratio,
            # مديونية
            "D_to_E": de, "IntCoverage": icov, "NetDebt_to_FCFF": nd_fcff, "NetDebt_to_EBITDA": nd_ebitda,
            # ربحية وهوامش
            "GM": gm, "EBITDA_Margin": ebitda_m, "EBIT_Margin": ebit_m, "Net_Margin": net_m, "OCF_Margin": ocf_m, "FCFF_Margin": fcff_m,
            # عوائد
            "ROA": roa, "ROE": roe, "ROIC": roic,
            # جودة وتحويل
            "CFO_to_NI": _safe_div(cfo, ni), "FCFF_to_NI": _safe_div(fcff, ni), "Capex_to_OCF": _safe_div(capex, cfo), "Capex_to_Rev": _safe_div(capex, rev),
            # كفاءة تشغيل
            "AssetTurnover": at, "ReceivablesTurnover": rt, "InventoryTurnover": it, "PayablesTurnover": pt, "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
            # مستويات
            "EBITDA": ebitda, "FCFF": fcff, "CFO": cfo, "CAPEX": capex
        }

    m0 = compute(a0, a1)
    m1 = compute(a1, {}) if y1 else {}

    # قواعد تقييم مبسّطة
    def judge(val: Optional[float], kind: str) -> str:
        if val is None or pd.isna(val):
            return "—"
        v = float(val)
        if kind == "CurrentRatio":      return "✅" if v >= 1.5 else ("⚠️" if v >= 1.0 else "❌")
        if kind == "QuickRatio":        return "✅" if v >= 1.0 else ("⚠️" if v >= 0.7 else "❌")
        if kind == "CashRatio":         return "✅" if v >= 0.2 else ("⚠️" if v >= 0.1 else "❌")
        if kind == "D_to_E":            return "✅" if v <= 0.5 else ("⚠️" if v <= 1.0 else "❌")
        if kind == "IntCoverage":       return "✅" if v >= 10 else ("⚠️" if v >= 6 else "❌")
        if kind == "NetDebt_to_FCFF":   return "✅" if v <= 2.0 else ("⚠️" if v <= 3.0 else "❌")
        if kind == "NetDebt_to_EBITDA": return "✅" if v <= 2.0 else ("⚠️" if v <= 3.0 else "❌")
        if kind == "GM":                return "✅" if v >= 0.25 else ("⚠️" if v >= 0.18 else "❌")
        if kind == "EBITDA_Margin":     return "✅" if v >= 0.20 else ("⚠️" if v >= 0.10 else "❌")
        if kind == "EBIT_Margin":       return "✅" if v >= 0.15 else ("⚠️" if v >= 0.10 else "❌")
        if kind == "Net_Margin":        return "✅" if v >= 0.10 else ("⚠️" if v >= 0.06 else "❌")
        if kind == "OCF_Margin":        return "✅" if v >= 0.12 else ("⚠️" if v >= 0.08 else "❌")
        if kind == "FCFF_Margin":       return "✅" if v >= 0.08 else ("⚠️" if v >= 0.05 else "❌")
        if kind == "ROA":               return "✅" if v >= 0.08 else ("⚠️" if v >= 0.05 else "❌")
        if kind == "ROE":               return "✅" if v >= 0.15 else ("⚠️" if v >= 0.10 else "❌")
        if kind == "ROIC":              return "✅" if v >= 0.15 else ("⚠️" if v >= 0.10 else "❌")
        if kind == "CFO_to_NI":         return "✅" if v >= 1.0 else ("⚠️" if v >= 0.8 else "❌")
        if kind == "FCFF_to_NI":        return "✅" if v >= 0.8 else ("⚠️" if v >= 0.5 else "❌")
        if kind == "Capex_to_OCF":      return "✅" if v <= 0.40 else ("⚠️" if v <= 0.60 else "❌")
        if kind == "Capex_to_Rev":      return "✅" if v <= 0.10 else ("⚠️" if v <= 0.15 else "❌")
        if kind == "AssetTurnover":     return "✅" if v >= 0.8 else ("⚠️" if v >= 0.5 else "❌")
        if kind == "ReceivablesTurnover": return "✅" if v >= 8 else ("⚠️" if v >= 5 else "❌")
        if kind == "InventoryTurnover": return "✅" if v >= 5 else ("⚠️" if v >= 3 else "❌")
        if kind == "DSO":               return "✅" if v <= 45 else ("⚠️" if v <= 60 else "❌")
        if kind == "DIO":               return "✅" if v <= 60 else ("⚠️" if v <= 90 else "❌")
        if kind == "CCC":               return "✅" if v <= 0 else ("⚠️" if v <= 30 else "❌")
        return "—"

    rows = []
    def add_row(category, name, explain, key, target, fmt="pct"):
        v0 = m0.get(key) if m0 else np.nan
        v1 = m1.get(key) if m1 else np.nan
        if fmt == "pct":
            a0 = _fmt_pct(v0); a1 = _fmt_pct(v1)
        elif fmt == "x":
            a0 = _fmt_x(v0); a1 = _fmt_x(v1)
        elif fmt == "num":
            a0 = _fmt_num(v0); a1 = _fmt_num(v1)
        else:
            a0 = _fmt_num(v0); a1 = _fmt_num(v1)
        verdict = judge(v0, key)
        rows.append({
            "الفئة": category,
            "البند": name,
            "شرح النسبة": explain,
            str(y0 or "أحدث"): a0,
            (str(y1) if y1 else "السنة السابقة"): a1,
            "المعيار/المستهدف": target,
            "رأي فني": verdict
        })

    # سيولة
    add_row("السيولة", "Current Ratio", "الأصول المتداولة ÷ الخصوم المتداولة", "CurrentRatio", "≥ 1.5", fmt="x")
    add_row("السيولة", "Quick Ratio", "(الأصول المتداولة − المخزون) ÷ الخصوم المتداولة", "QuickRatio", "≥ 1.0", fmt="x")
    add_row("السيولة", "Cash Ratio", "النقد ÷ الخصوم المتداولة", "CashRatio", "≥ 0.2", fmt="x")

    # مديونية
    add_row("المديونية", "D/E", "إجمالي الدين ÷ حقوق الملكية", "D_to_E", "≤ 0.5 (≤1.0 مقبول)", fmt="x")
    add_row("المديونية", "تغطية الفوائد", "EBIT ÷ مصروف الفائدة", "IntCoverage", "≥ 10x (≥6x مقبول)", fmt="x")
    add_row("المديونية", "صافي الدين/EBITDA", "(الدين − النقد) ÷ EBITDA", "NetDebt_to_EBITDA", "≤ 2.0x", fmt="x")
    add_row("المديونية", "صافي الدين/FCFF", "(الدين − النقد) ÷ FCFF", "NetDebt_to_FCFF", "≤ 2.0x", fmt="x")

    # هوامش ربحية
    add_row("الربحية", "الهامش الإجمالي", "(الإيراد − تكلفة المبيعات) ÷ الإيراد", "GM", "≥ 25%", fmt="pct")
    add_row("الربحية", "هامش EBITDA", "EBITDA ÷ الإيراد", "EBITDA_Margin", "≥ 20%", fmt="pct")
    add_row("الربحية", "هامش التشغيل (EBIT)", "EBIT ÷ الإيراد", "EBIT_Margin", "≥ 15%", fmt="pct")
    add_row("الربحية", "الهامش الصافي", "صافي الربح ÷ الإيراد", "Net_Margin", "≥ 10%", fmt="pct")
    add_row("الربحية", "هامش OCF", "OCF ÷ الإيراد", "OCF_Margin", "≥ 12%", fmt="pct")
    add_row("الربحية", "هامش FCFF", "FCFF ÷ الإيراد", "FCFF_Margin", "≥ 8%", fmt="pct")

    # عوائد
    add_row("العوائد", "ROA", "صافي الربح ÷ متوسط الأصول", "ROA", "≥ 8%", fmt="pct")
    add_row("العوائد", "ROE", "صافي الربح ÷ متوسط حقوق الملكية", "ROE", "≥ 15%", fmt="pct")
    add_row("العوائد", "ROIC", "NOPAT ÷ (الدين + حقوق − النقد)", "ROIC", "≥ 15%", fmt="pct")

    # جودة وتحويل
    add_row("الجودة/التحويل", "CFO/NI", "التدفق التشغيلي ÷ صافي الربح", "CFO_to_NI", "≥ 1.0", fmt="x")
    add_row("الجودة/التحويل", "FCFF/NI", "التدفق الحر للمنشأة ÷ صافي الربح", "FCFF_to_NI", "≥ 0.8", fmt="x")
    add_row("الجودة/التحويل", "Capex/OCF", "النفقات الرأسمالية ÷ OCF", "Capex_to_OCF", "≤ 40%", fmt="pct")
    add_row("الجودة/التحويل", "Capex/Revenue", "النفقات الرأسمالية ÷ الإيراد", "Capex_to_Rev", "≤ 10%", fmt="pct")

    # كفاءة تشغيل
    add_row("الكفاءة", "دوران الأصول", "الإيراد ÷ متوسط الأصول", "AssetTurnover", "≥ 0.8x", fmt="x")
    add_row("الكفاءة", "دوران الذمم", "الإيراد ÷ متوسط الذمم المدينة", "ReceivablesTurnover", "≥ 8x", fmt="x")
    add_row("الكفاءة", "دوران المخزون", "COGS ÷ متوسط المخزون", "InventoryTurnover", "≥ 5x", fmt="x")
    add_row("الكفاءة", "DSO", "أيام التحصيل = 365 ÷ دوران الذمم", "DSO", "≤ 45 يوم", fmt="num")
    add_row("الكفاءة", "DIO", "أيام المخزون = 365 ÷ دوران المخزون", "DIO", "≤ 60 يوم", fmt="num")
    add_row("الكفاءة", "DPO", "أيام السداد = 365 ÷ دوران الموردين", "DPO", "—", fmt="num")
    add_row("الكفاءة", "CCC", "دورة التحويل النقدي = DSO + DIO − DPO", "CCC", "≤ 0 يوم (≤30 مقبول)", fmt="num")

    # مستويات مطلقة (عرض فقط)
    add_row("المستويات", "EBITDA", "الأرباح قبل الفائدة والضريبة والإهلاك", "EBITDA", "—", fmt="num")
    add_row("المستويات", "CFO", "التدفق التشغيلي", "CFO", "—", fmt="num")
    add_row("المستويات", "Capex", "النفقات الرأسمالية", "CAPEX", "—", fmt="num")
    add_row("المستويات", "FCFF", "التدفق الحر للمنشأة (CFO − Capex)", "FCFF", "—", fmt="num")

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
    st.caption("التركيز الآن على جدول نسب موسّع قائم على التدفقات النقدية والسيولة فقط (بدون تفاصيل DCF).")

# زر التنفيذ
if st.button("🚀 قيّم الشركة"):
    if not symbol:
        st.warning("يرجى إدخال رمز صحيح.")
        st.stop()

    with st.spinner("جاري تحميل البيانات وبناء اللوحات…"):
        data = load_data(symbol)
        ttm = compute_ttm_blocks(data)
        bal = compute_latest_balance(data)
        fcf = compute_fcf(data)
        annual = compute_annual_blocks(data)
        ratios_df = build_cash_ratios_table(annual)

    # ===== KPIs رئيسية (كاش فقط) =====
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi {'ok' if (ttm.get('CFO_TTM') is not None and not pd.isna(ttm.get('CFO_TTM')) and ttm.get('CFO_TTM')>0) else 'bad'}'>"
            "<div class='title'>OCF (TTM)</div>"
            f"<div class='value'>{_fmt_num(ttm.get('CFO_TTM'))}</div>"
            "<div class='small'>تشغيلي</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='kpi {'mid' if (ttm.get('CAPEX_TTM') is not None and not pd.isna(ttm.get('CAPEX_TTM')) and ttm.get('CAPEX_TTM')>0) else 'ok'}'>"
            "<div class='title'>Capex (TTM)</div>"
            f"<div class='value'>{_fmt_num(ttm.get('CAPEX_TTM'))}</div>"
            "<div class='small'>استثمار</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='kpi {'ok' if (fcf.fcff_ttm is not None and not pd.isna(fcf.fcff_ttm) and fcf.fcff_ttm>0) else 'bad'}'>"
            "<div class='title'>FCFF (TTM)</div>"
            f"<div class='value'>{_fmt_num(fcf.fcff_ttm)}</div>"
            "<div class='small'>≈ OCF − Capex</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        try:
            ykeys = list(annual.keys())
            cr = _safe_div(annual[ykeys[0]].get('CA'), annual[ykeys[0]].get('CL')) if ykeys else np.nan
        except Exception:
            cr = np.nan
        st.markdown(
            f"<div class='kpi {'ok' if (cr is not None and not pd.isna(cr) and cr>=1.5) else ('mid' if (cr is not None and not pd.isna(cr) and cr>=1.0) else 'bad') }'>"
            "<div class='title'>Current Ratio</div>"
            f"<div class='value'>{_fmt_x(cr)}</div>"
            "<div class='small'>سيولة قصيرة الأجل</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🧾 تفصيل TTM وملخص الميزانية")
    ttm_rows = [{
        "الإيراد (TTM)": _fmt_num(ttm.get('REV_TTM')),
        "EBIT (TTM)": _fmt_num(ttm.get('EBIT_TTM')),
        "EBITDA (TTM)": _fmt_num(ttm.get('EBITDA_TTM')),
        "صافي الربح (TTM)": _fmt_num(ttm.get('NI_TTM')),
        "OCF (TTM)": _fmt_num(ttm.get('CFO_TTM')),
        "Capex (TTM)": _fmt_num(ttm.get('CAPEX_TTM')),
        "FCFF (TTM)": _fmt_num(fcf.fcff_ttm)
    }]
    st.dataframe(pd.DataFrame(ttm_rows), use_container_width=True)

    bal_rows = [{
        "إجمالي الأصول": _fmt_num(bal.get('TA')),
        "حقوق الملكية": _fmt_num(bal.get('TE')),
        "النقد": _fmt_num(bal.get('Cash')),
        "الاستثمارات القصيرة": _fmt_num(bal.get('STI')),
        "إجمالي الدين": _fmt_num(bal.get('TotalDebt')),
        "صافي الدين": _fmt_num(bal.get('NetDebt')),
        "NWC (تقريبي)": _fmt_num(bal.get('NWC'))
    }]
    st.dataframe(pd.DataFrame(bal_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 لوحة نسب نقدية/تشغيلية موسّعة")
    try:
        st.dataframe(ratios_df, use_container_width=True)
        st.caption("المعايير عامة وقد تختلف حسب القطاع/الدورة. القيم '—' تعني عدم توفر بيانات كافية.")
    except Exception as e:
        st.info(f"تعذر بناء اللوحة: {e}")

    st.caption("هذا النموذج لأغراض تعليمية/بحثية، وليس توصية استثمارية.")
else:
    st.info("ادخل الرمز ثم اضغط \"🚀 قيّم الشركة\".")

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
