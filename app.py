# -*- coding: utf-8 -*- 
"""
📊 Financial Analysis Model (Buffett Principles) — v3 (Detailed Report) + WS
ملف واحد — تحليل مالي شامل + تقرير Markdown مفصل + نظام إنذارات تلاعب/جودة أرباح (WS).
تشغيل: streamlit run app.py
اعتماديات: streamlit, yfinance, pandas, numpy, dataclasses
"""

import re
from html import escape
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# تهيئة الصفحة + RTL + ثيم بصري
# =============================
st.set_page_config(page_title="📊 نموذج التحليل المالي | Buffett Principles + WS", layout="wide")
THEME_CSS = """
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; font-family: -apple-system, Segoe UI, Tahoma, Arial, sans-serif; }
  input, textarea, select { direction: rtl; text-align: right; }
  .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
  .stNumberInput input, .stDateInput input, .stMultiSelect [data-baseweb],
  label, .stButton button { text-align: right; }

  .hero { background: linear-gradient(90deg,#e0f2fe,#ecfeff);
    padding: 14px 18px; border:1px solid #e2e8f0; border-radius: 14px; margin-bottom: 10px; }
  .hero h1 { margin: 0; font-size: 22px; }
  .muted { color:#475569; font-size:13px; }

  .kpi { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:14px; height:100%; }
  .kpi .title { color:#64748b; font-size:13px; margin-bottom:4px;}
  .kpi .value { font-size:20px; font-weight:700; }
  .kpi .sub { color:#64748b; font-size:12px; margin-top:4px; }
  .kpi.ok .value { color:#059669; } .kpi.mid .value{ color:#d97706; } .kpi.bad .value{ color:#dc2626; }

  .buffett-table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
  .buffett-table th, .buffett-table td {border: 1px solid #e5e7eb; padding: 8px; text-align: center;}
  .buffett-table th {background-color: #0ea5e9; color: white;}
  .buffett-table tr:nth-child(even){background-color: #f8fafc;}
  .buffett-table tr:hover {background-color: #eef2ff;}
  .buffett-table td.green { color: #059669; font-weight: bold; }
  .buffett-table td.yellow { color: #d97706; font-weight: bold; }
  .buffett-table td.red { color: #dc2626; font-weight: bold; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# =============================
# أدوات مساعدة (أرقام/فورمات)
# =============================
def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def build_index_map(df: pd.DataFrame):
    return {normalize_idx(raw): raw for raw in df.index.astype(str)}

def find_any(df: pd.DataFrame, keys: List[str], col):
    if df is None or df.empty or col is None:
        return np.nan
    idx = build_index_map(df)
    for k in keys:
        kk = normalize_idx(k)
        if kk in idx:
            try:
                return float(df.loc[idx[kk], col])
            except Exception:
                try:
                    return float(pd.to_numeric(df.loc[idx[kk], col], errors="coerce"))
                except Exception:
                    return np.nan
    return np.nan

def sorted_cols(df: pd.DataFrame, reverse=True):
    try:
        return sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=reverse)
    except Exception:
        cols = list(df.columns)
        return cols[::-1] if reverse else cols

def safe_div(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan

def to_percent(x, digits=2):
    return "—" if x is None or pd.isna(x) else f"{x*100:.{digits}f}%"

def to_ratio(x, digits=2):
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}x"

def to_days(x):
    return "—" if x is None or pd.isna(x) else f"{x:.1f} يوم"

def to_num(x, digits=2):
    if x is None or pd.isna(x): return "—"
    ax = abs(float(x))
    if ax >= 1_000_000_000_000: return f"{x/1_000_000_000_000:.{digits}f}T"
    if ax >= 1_000_000_000:     return f"{x/1_000_000_000:.{digits}f}B"
    if ax >= 1_000_000:         return f"{x/1_000_000:.{digits}f}M"
    if ax >= 1_000:             return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

def status_word(sym):
    return "متوافق" if sym == "✅" else ("مقبول" if sym == "⚠️" else "غير متوافق")

def classify(value, ok=None, mid=None, reverse=False):
    if value is None or pd.isna(value): return "bad"
    v = float(value)
    if reverse:
        if ok is not None and v <= ok:   return "ok"
        if mid is not None and v <= mid: return "mid"
        return "bad"
    else:
        if ok is not None and v >= ok:   return "ok"
        if mid is not None and v >= mid: return "mid"
        return "bad"

def kpi_card(title, value_str, sub=None, status="ok"):
    cls = f"kpi {status}"
    sub_html = f"<div class='sub'>{escape(sub)}</div>" if sub else ""
    return f"""
    <div class="{cls}">
      <div class="title">{escape(title)}</div>
      <div class="value">{escape(value_str)}</div>
      {sub_html}
    </div>
    """

def md_table(headers, rows):
    """يبني جدول Markdown بسيط."""
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"]*len(headers)) + " |"
    lines = [line1, line2]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)

# مفاتيح Yahoo
REV_KEYS = ["Total Revenue","Revenue","TotalRevenue","Sales"]
COGS_KEYS = ["Cost Of Revenue","Cost of Revenue","CostOfRevenue","COGS"]
GP_KEYS   = ["Gross Profit","GrossProfit"]
OPINC_KEYS= ["Operating Income","OperatingIncome","EBIT"]
EBIT_KEYS = ["EBIT","Operating Income","OperatingIncome"]
NI_KEYS   = ["Net Income","NetIncome","Net Income Common Stockholders","Net Income Applicable To Common Shares"]
PBT_KEYS  = ["Income Before Tax","Pretax Income","Earnings Before Tax"]
TAX_KEYS  = ["Income Tax Expense","Tax Provision","Provision For Income Taxes"]
TA_KEYS   = ["Total Assets","TotalAssets"]
TE_KEYS   = ["Total Stockholder Equity","Total Shareholder Equity","Total Equity Gross Minority Interest","Total Stockholders Equity"]
CA_KEYS   = ["Total Current Assets","Current Assets","TotalCurrentAssets"]
CL_KEYS   = ["Total Current Liabilities","Current Liabilities","TotalCurrentLiabilities"]
INV_KEYS  = ["Inventory","Inventory Net"]
AR_KEYS   = ["Net Receivables","Accounts Receivable","Receivables"]
AP_KEYS   = ["Accounts Payable","Payables"]
CASH_KEYS = ["Cash And Cash Equivalents","Cash And Cash Equivalents, And Short Term Investments","Cash"]
STI_KEYS  = ["Short Term Investments"]
LTD_KEYS  = ["Long Term Debt"]
SLTD_KEYS = ["Short Long Term Debt"]
CUR_DEBT_KEYS = ["Current Debt"]
TOT_DEBT_KEYS = ["Total Debt"]
INT_EXP_KEYS = ["Interest Expense"]
OCF_KEYS  = ["Operating Cash Flow","Total Cash From Operating Activities"]
CAPEX_KEYS = ["Capital Expenditure","Capital Expenditures"]

# مفاتيح إضافية للـ WS
OTHER_INC_KEYS = ["Other Income Expense","Total Other Income/Expenses Net","Other Non Operating Income (Expense)"]
GOODWILL_KEYS  = ["Goodwill"]
INTANG_KEYS    = ["Intangible Assets","Other Intangible Assets"]
ACQ_KEYS       = ["Acquisitions Net","Net Income From Continuing Ops Net Minority Interest","Net Acquisition"]  # الأول هو الصحيح غالباً

# =============================
# التحميل — كاش قابلة للتسلسل
# =============================
@st.cache_data(ttl=1800)
def load_company_data(ticker: str):
    t = yf.Ticker(ticker)

    def get_df(callable_df):
        try:
            df = callable_df()
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
        data_info = {}
        try:
            data_info = t.get_info()
        except Exception:
            data_info = getattr(t, "info", {}) or {}
        if isinstance(data_info, dict):
            for f in ["longName","industry","sector","country","city","fullTimeEmployees","website","longBusinessSummary","currency","financialCurrency"]:
                val = data_info.get(f, None)
                if isinstance(val, (str, int, float)) or val is None:
                    info[f] = val
    except Exception:
        pass

    price = shares = market_cap = np.nan
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        shares = float(fi.get("shares", np.nan))
        market_cap = float(fi.get("market_cap", np.nan))
    except Exception:
        pass

    return {
        "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q, "cf_a": cf_a, "cf_q": cf_q,
        "price": price, "shares": shares, "market_cap": market_cap, "info": info
    }

# =============================
# TTM / تجميع
# =============================
def sum_last4(df: pd.DataFrame, keys):
    if df is None or df.empty: return np.nan
    cols = sorted_cols(df)[:4]
    vals = [find_any(df, keys, c) for c in cols]
    vals = [v for v in vals if not pd.isna(v)]
    return sum(vals) if vals else np.nan

def compute_ttm(inc_q, cf_q):
    rev = sum_last4(inc_q, REV_KEYS)
    ebit = sum_last4(inc_q, EBIT_KEYS)
    if pd.isna(ebit): ebit = sum_last4(inc_q, OPINC_KEYS)
    ni  = sum_last4(inc_q, NI_KEYS)
    ocf = sum_last4(cf_q, OCF_KEYS)
    cap = sum_last4(cf_q, CAPEX_KEYS)
    cogs = sum_last4(inc_q, COGS_KEYS)
    return rev, ebit, ni, ocf, cap, cogs

# =============================
# حساب المؤشرات الجوهرية
# =============================
def compute_core_metrics(data: dict, mode: str):
    inc_a, inc_q = data["inc_a"], data["inc_q"]
    bal_a, bal_q = data["bal_a"], data["bal_q"]
    cf_a,  cf_q  = data["cf_a"],  data["cf_q"]

    if mode == "TTM" and not inc_q.empty:
        rev, ebit, ni, ocf, capex, cogs = compute_ttm(inc_q, cf_q)
        inc_used = inc_q
        bal = bal_q if not bal_q.empty else bal_a
        col_income = sorted_cols(inc_q)[0] if not inc_q.empty else None
    else:
        col_i = sorted_cols(inc_a)[0] if not inc_a.empty else None
        col_c = sorted_cols(cf_a)[0] if not cf_a.empty else None
        rev  = find_any(inc_a, REV_KEYS, col_i)
        ebit = find_any(inc_a, EBIT_KEYS, col_i)
        if pd.isna(ebit): ebit = find_any(inc_a, OPINC_KEYS, col_i)
        ni   = find_any(inc_a, NI_KEYS, col_i)
        ocf  = find_any(cf_a, OCF_KEYS, col_c)
        capex= find_any(cf_a, CAPEX_KEYS, col_c)
        cogs = find_any(inc_a, COGS_KEYS, col_i)
        inc_used = inc_a
        bal = bal_a
        col_income = col_i

    # هوامش
    gp = (rev - cogs) if (not pd.isna(rev) and not pd.isna(cogs)) else np.nan
    gross_margin = safe_div(gp, rev)
    op_margin    = safe_div(ebit, rev)
    net_margin   = safe_div(ni, rev)

    # الميزانية
    bal_cols = sorted_cols(bal)
    cur = bal_cols[0] if bal_cols else None
    prev = bal_cols[1] if len(bal_cols) > 1 else None
    ta = find_any(bal, TA_KEYS, cur)
    te = find_any(bal, TE_KEYS, cur)
    ca = find_any(bal, CA_KEYS, cur)
    cl = find_any(bal, CL_KEYS, cur)
    inv = find_any(bal, INV_KEYS, cur)
    cash = find_any(bal, CASH_KEYS, cur)
    sti  = find_any(bal, STI_KEYS, cur)
    ar = find_any(bal, AR_KEYS, cur);  ar_prev = find_any(bal, AR_KEYS, prev)
    ap = find_any(bal, AP_KEYS, cur);  ap_prev = find_any(bal, AP_KEYS, prev)
    inv_prev = find_any(bal, INV_KEYS, prev)

    # رأس المال المستثمر و ROIC
    total_debt = find_any(bal, TOT_DEBT_KEYS, cur)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, cur) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan
    invested = np.nan if (pd.isna(total_debt) or pd.isna(te)) else total_debt + te - (0 if pd.isna(cash) else cash)

    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = tax / pbt if (not pd.isna(pbt) and pbt != 0 and not pd.isna(tax)) else 0.25
    eff_tax = float(np.clip(eff_tax, 0.0, 0.6))
    nopat = ebit * (1 - eff_tax) if not pd.isna(ebit) else np.nan
    roic = safe_div(nopat, invested)

    # أرباح المالك + الجودة
    owner_earnings = np.nan if (pd.isna(ocf) or pd.isna(capex)) else (ocf - capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(owner_earnings, rev)

    # السيولة/الملاءة
    current_ratio = safe_div(ca, cl)
    quick_ratio   = safe_div((ca - (inv if not pd.isna(inv) else 0)), cl)
    debt_to_equity = safe_div(total_debt, te)
    roa = safe_div(ni, ta)
    roe = safe_div(ni, te)

    # الفوائد
    int_exp = find_any(inc_used, INT_EXP_KEYS, col_income)
    if not pd.isna(int_exp): int_exp = abs(int_exp)
    interest_cov = safe_div(ebit, int_exp)

    # الكفاءة + CCC
    ta_prev = find_any(bal, TA_KEYS, prev)
    avg_assets = np.nanmean([ta, ta_prev])
    asset_turn = safe_div(rev, avg_assets)
    ar_avg  = np.nanmean([ar, ar_prev])
    ap_avg  = np.nanmean([ap, ap_prev])
    inv_avg = np.nanmean([inv, inv_prev])
    cogs_eff = cogs if not pd.isna(cogs) else rev
    rec_turn = safe_div(rev, ar_avg)
    pay_turn = safe_div(cogs_eff, ap_avg)
    inv_turn = safe_div(cogs_eff, inv_avg)
    dso = safe_div(365, rec_turn)
    dpo = safe_div(365, pay_turn)
    dio = safe_div(365, inv_turn)
    ccc = dso + dio - dpo if not any(pd.isna(x) for x in [dso, dio, dpo]) else np.nan

    # السوق/التقييم
    price = data.get("price", np.nan)
    shares = data.get("shares", np.nan)
    market_cap = data.get("market_cap", np.nan)
    pe = pb = ps = bvps = np.nan
    if not (pd.isna(price) or pd.isna(shares) or shares == 0):
        eps = safe_div(ni, shares)
        pe = safe_div(price, eps)
        sales_ps = safe_div(rev, shares)
        ps = safe_div(price, sales_ps)
        bvps = safe_div(te, shares)
        pb = safe_div(price, bvps)
    if (pd.isna(market_cap) or market_cap == 0) and (not pd.isna(price) and not pd.isna(shares)):
        market_cap = price * shares
    oe_yield = safe_div(owner_earnings, market_cap)
    p_to_oe  = safe_div(market_cap, owner_earnings)

    # تواريخ آخر فترات
    meta = {
        "income_period": (str(sorted_cols(inc_q)[0]) if (mode=="TTM" and not inc_q.empty) else (str(sorted_cols(inc_a)[0]) if not inc_a.empty else "—"))),
        "balance_period": (str(sorted_cols(bal_q)[0]) if (mode=="TTM" and not bal_q.empty) else (str(sorted_cols(bal_a)[0]) if not bal_a.empty else "—"))),
        "cashflow_period": (str(sorted_cols(cf_q)[0]) if (mode=="TTM" and not cf_q.empty) else (str(sorted_cols(cf_a)[0]) if not cf_a.empty else "—"))),
    }

    return {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "EBIT": ebit, "NetIncome": ni,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "Cash": cash, "STInvest": sti, "TotalDebt": total_debt,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": owner_earnings,
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "ROA": roa, "ROE": roe, "ROIC": roic, "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "CurrentRatio": current_ratio, "QuickRatio": quick_ratio, "InterestCoverage": interest_cov,
        "AssetTurnover": asset_turn, "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        "Price": price, "Shares": shares, "MarketCap": market_cap,
        "PE": pe, "PB": pb, "PS": ps, "BVPS": bvps,
        "OwnerEarningsYield": oe_yield, "P/OwnerEarnings": p_to_oe,
        "_meta": meta
    }

# =============================
# قائمة تحقق بافيت + الأسباب + النقاط
# =============================
def buffett_scorecard(r):
    score = 0
    flags = {}
    reasons = []
    components = []  # جديد: تفصيل النقاط

    def set_flag(name, ok, mid=False, points_ok=10, points_mid=5, points_bad=0, explain=""):
        nonlocal score
        sym = "✅" if ok else ("⚠️" if mid else "❌")
        flags[name] = sym
        pts = points_ok if ok else (points_mid if mid else points_bad)
        score += pts
        components.append({"البند": name, "الرمز": sym, "النقاط": pts, "التفسير": explain})
        return sym

    # 1) ROIC ≥ 15%
    roic = r["ROIC"]
    ok = (not pd.isna(roic) and roic >= 0.15)
    mid = (not pd.isna(roic) and 0.10 <= roic < 0.15)
    sym = set_flag("ROIC ≥15%", ok, mid, points_ok=20, points_mid=8,
                   explain=("غير متوفر" if pd.isna(roic) else f"ROIC = {to_percent(roic)} (≥15% ممتاز، ≥10% مقبول)."))
    reasons.append({"البند":"ROIC ≥15%","الحالة":status_word(sym),
                    "السبب": "غير متوفر" if pd.isna(roic) else f"ROIC = {to_percent(roic)}."})

    # 2) الهامش الإجمالي ≥25%
    gm = r["GrossMargin"]; ok=(not pd.isna(gm) and gm>=0.25); mid=(not pd.isna(gm) and 0.18<=gm<0.25)
    sym=set_flag("هامش إجمالي قوي", ok, mid,
                 explain=("غير متوفر" if pd.isna(gm) else f"Gross Margin = {to_percent(gm)}."))
    reasons.append({"البند":"هامش إجمالي قوي","الحالة":status_word(sym),"السبب": "غير متوفر" if pd.isna(gm) else f"{to_percent(gm)}."})

    # 3) جودة الأرباح OCF/NI ≥1
    q=r["OCF/NI"]; ok=(not pd.isna(q) and q>=1.0); mid=(not pd.isna(q) and 0.8<=q<1.0)
    sym=set_flag("جودة الأرباح OCF/NI ≥1", ok, mid, explain=("غير متوفر" if pd.isna(q) else f"OCF/NI = {to_ratio(q)}."))
    reasons.append({"البند":"جودة الأرباح OCF/NI","الحالة":status_word(sym),"السبب": "غير متوفر" if pd.isna(q) else f"{to_ratio(q)}."})

    # 4) هامش أرباح المالك ≥8%
    f=r["FCF_Margin"]; ok=(not pd.isna(f) and f>=0.08); mid=(not pd.isna(f) and 0.05<=f<0.08)
    sym=set_flag("هامش أرباح المالك ≥8%", ok, mid, explain=("غير متوفر" if pd.isna(f) else f"OE Margin = {to_percent(f)}."))
    reasons.append({"البند":"هامش التدفق الحر","الحالة":status_word(sym),"السبب":"غير متوفر" if pd.isna(f) else f"{to_percent(f)}."})

    # 5) هيكل دين متحفظ
    td, cash = r["TotalDebt"], r["Cash"]; oe = r["OwnerEarnings"]
    net_debt = np.nan if pd.isna(td) else td - (0 if pd.isna(cash) else cash)
    ratio_debt_oe = (td/oe) if (not any(pd.isna(x) for x in [td, oe]) and oe>0) else np.nan
    crit = (not pd.isna(net_debt) and net_debt<=0) or (not pd.isna(ratio_debt_oe) and ratio_debt_oe<=2.0)
    mid  = (not pd.isna(ratio_debt_oe) and ratio_debt_oe<=3.0)
    sym=set_flag("هيكل دين متحفظ", crit, mid,
                 explain=("بيانات غير متوفرة" if (pd.isna(td) and pd.isna(cash)) else
                          (f"صافي الدين: {to_num(net_debt)}" + (f"، الدين/أرباح المالك: {to_ratio(ratio_debt_oe)}" if not pd.isna(ratio_debt_oe) else ""))))
    reasons.append({"البند":"هيكل دين متحفظ","الحالة":status_word(sym),
                    "السبب": "بيانات غير متوفرة" if (pd.isna(td) and pd.isna(cash)) else
                    (f"صافي الدين: {to_num(net_debt)}" + (f"، الدين/أرباح المالك: {to_ratio(ratio_debt_oe)}" if not pd.isna(ratio_debt_oe) else ""))})

    # 6) تغطية الفوائد ≥10x
    ic=r["InterestCoverage"]; ok=(not pd.isna(ic) and ic>=10.0); mid=(not pd.isna(ic) and 6.0<=ic<10.0)
    sym=set_flag("تغطية الفوائد ≥10x", ok, mid, explain=("غير متوفر" if pd.isna(ic) else f"Interest Coverage = {to_ratio(ic)}."))
    reasons.append({"البند":"تغطية الفوائد","الحالة":status_word(sym),"السبب":"غير متوفر" if pd.isna(ic) else f"{to_ratio(ic)}."})

    # 7) CCC ≤ 0 يوم (≤30 مقبول)
    ccc=r["CCC"]; ok=(not pd.isna(ccc) and ccc<=0); mid=(not pd.isna(ccc) and ccc<=30)
    sym=set_flag("دورة التحويل النقدي ≤0", ok, mid, points_ok=5, points_mid=2, explain=("غير متوفر" if pd.isna(ccc) else f"CCC = {to_days(ccc)}."))
    reasons.append({"البند":"دورة التحويل النقدي","الحالة":status_word(sym),"السبب":"غير متوفر" if pd.isna(ccc) else f"{to_days(ccc)}."})

    # 8) تقييم معقول
    oey=r["OwnerEarningsYield"]; pto=r["P/OwnerEarnings"]
    ok = (not pd.isna(oey) and oey>=0.06) or (not pd.isna(pto) and pto<=20)
    mid= (not pd.isna(oey) and oey>=0.04) or (not pd.isna(pto) and pto<=25)
    sym=set_flag("تقييم معقول (OE Yield ≥6% أو P/OE ≤20)", ok, mid,
                 explain=((f"OE Yield = {to_percent(oey)}" if not pd.isna(oey) else "")+
                          ("؛ " if (not pd.isna(oey) and not pd.isna(pto)) else "")+
                          (f"P/OE = {to_ratio(pto)}" if not pd.isna(pto) else "")))
    reasons.append({"البند":"تقييم معقول","الحالة":status_word(sym),
                    "السبب": (f"OE Yield {to_percent(oey)} | " if not pd.isna(oey) else "") +
                             (f"P/OE {to_ratio(pto)}" if not pd.isna(pto) else "")})

    verdict = "✅ جذّابة مع هامش أمان" if score >= 75 else ("🟧 جيدة لكن انتظر سعرًا أفضل" if score >= 55 else "🕒 راقِب")
    return float(score), flags, verdict, (np.nan if pd.isna(td) else net_debt), reasons, components

# =============================
# اتجاهات تاريخية (مبسّطة للرسوم)
# =============================
def historical_trends(inc_a: pd.DataFrame, cf_a: pd.DataFrame, years: int = 5):
    def take_series(df, keys):
        if df is None or df.empty: return pd.Series(dtype=float)
        cols = sorted_cols(df)[:years] if years>0 else sorted_cols(df)
        data = {str(c): find_any(df, keys, c) for c in cols}
        return pd.Series(data)

    rev_s = take_series(inc_a, REV_KEYS)
    ni_s  = take_series(inc_a, NI_KEYS)
    ocf_s = take_series(cf_a, OCF_KEYS)
    cap_s = take_series(cf_a, CAPEX_KEYS)
    oe_s  = ocf_s - cap_s

    df = pd.DataFrame({"Revenue": rev_s, "NetIncome": ni_s, "OwnerEarnings": oe_s}).T
    df = df.replace([np.inf,-np.inf], np.nan)
    return df

# =============================
# DCF مبسّط
# =============================
def simple_dcf(oe_base, discount_rate=0.12, growth_rate=0.05, years=5, terminal_growth=0.02):
    if pd.isna(oe_base) or oe_base<=0 or discount_rate<=terminal_growth:
        return np.nan, pd.DataFrame()
    flows = []; pv = 0.0
    for t in range(1, years+1):
        cf = oe_base * ((1+growth_rate) ** t)
        pv_cf = cf / ((1+discount_rate) ** t)
        flows.append({"السنة": t, "التدفق المتوقع": cf, "القيمة الحالية": pv_cf})
        pv += pv_cf
    tv = (oe_base * ((1+growth_rate) ** years) * (1+terminal_growth)) / (discount_rate - terminal_growth)
    pv_tv = tv / ((1+discount_rate) ** years)
    flows.append({"السنة": "قيمة نهائية", "التدفق المتوقع": tv, "القيمة الحالية": pv_tv})
    total_pv = pv + pv_tv
    return total_pv, pd.DataFrame(flows)

# =============================
# ======== WS: Warning Signs ========
# =============================
@dataclass
class WSRuleResult:
    rule_id: str
    title: str
    severity: str  # 'high' | 'medium' | 'low'
    flagged: bool
    metric_value: Optional[float]
    threshold: Optional[str]
    rationale: str

@dataclass
class WSPeerStats:
    revenue_cagr_median: float = np.nan
    revenue_cagr_std: float = np.nan
    gross_margin_median: float = np.nan
    gross_margin_std: float = np.nan
    op_margin_median: float = np.nan
    op_margin_std: float = np.nan
    asset_turnover_median: float = np.nan
    asset_turnover_std: float = np.nan

_WS_SEVERITY_WEIGHT = {'high': 3, 'medium': 2, 'low': 1}

_WS_DISCLOSURE_PATTERNS = {
    'rev_change_methods': r'(change|revis\w+)\s+(in\s+)?(revenue recognition|accounting polic\w+)',
    'bill_and_hold': r'\bbill[- ]?and[- ]?hold\b',
    'barter': r'\bbarter\b|\bnon[- ]monetary exchange\b',
    'rebates': r'\brebate(s)?\b|\bcontra[- ]?revenue\b',
    'related_party': r'\brelated\s+part(y|ies)\b',
    'non_gaap_push': r'\bnon[- ]gaap\b|\badjusted\b\s+(earnings|ebitda|profit)',
    'nonrecurring': r'\bnon[- ]?recurring\b|\bone[- ]?time\b|\bspecial charge\b',
    'lifo_reserve': r'\bLIFO reserve\b|\bLIFO liquidation\b',
    'capitalized_dev': r'\bcapitali[sz]ed (software|development)\b|\bintangible additions\b'
}

def _ws_trend_slope(series: pd.Series) -> Optional[float]:
    y = series.dropna().astype(float).values
    if len(y) < 4:
        return None
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])

def _ws_z(value: float, mean: float, std: float) -> Optional[float]:
    if std is None or std == 0 or pd.isna(std) or value is None or pd.isna(value):
        return None
    return (value - mean) / std

def _ws_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = numer.astype(float) / denom.replace(0, np.nan).astype(float)
    return ratio

def ws_search_disclosures(text: str, key: str) -> bool:
    if not isinstance(text, str) or key not in _WS_DISCLOSURE_PATTERNS:
        return False
    return re.search(_WS_DISCLOSURE_PATTERNS[key], text, flags=re.IGNORECASE) is not None

def ws_build_financials_ts(data: dict, max_points: int = 12) -> pd.DataFrame:
    """يبني DataFrame زمني لفصول حديثة (حتى 12 ربع) مع الحقول اللازمة لقواعد WS."""
    inc_q, bal_q, cf_q = data["inc_q"], data["bal_q"], data["cf_q"]
    # fallback سنوي لو مافيه ربعي
    inc_src = inc_q if not inc_q.empty else data["inc_a"]
    bal_src = bal_q if not bal_q.empty else data["bal_a"]
    cf_src  = cf_q  if not cf_q.empty  else data["cf_a"]

    cols = sorted_cols(inc_src)[:max_points] if not inc_src.empty else []
    rows = []
    for c in cols:
        revenue = find_any(inc_src, REV_KEYS, c)
        cogs    = find_any(inc_src, COGS_KEYS, c)
        opinc   = find_any(inc_src, OPINC_KEYS, c)
        ni      = find_any(inc_src, NI_KEYS, c)
        other_i = find_any(inc_src, OTHER_INC_KEYS, c)

        receiv  = find_any(bal_src, AR_KEYS, c)
        inv     = find_any(bal_src, INV_KEYS, c)
        assets  = find_any(bal_src, TA_KEYS, c)
        goodw   = find_any(bal_src, GOODWILL_KEYS, c)
        intang  = find_any(bal_src, INTANG_KEYS, c)

        cfo     = find_any(cf_src, OCF_KEYS, c)
        acq     = find_any(cf_src, ACQ_KEYS, c)  # غالباً سالبة عند الاستحواذ

        rows.append({
            "date": str(c),
            "revenue": revenue,
            "cogs": cogs,
            "operating_income": opinc,
            "net_income": ni,
            "cfo": cfo,
            "receivables": receiv,
            "inventory": inv,
            "total_assets": assets,
            "goodwill": goodw,
            "intangibles": intang,
            "other_income": other_i,
            "acq_cash_outflow": acq
        })
    return pd.DataFrame(rows)

# قواعد WS الرقمية
def ws_rule_revenue_outlier_vs_peers(df: pd.DataFrame, peers: WSPeerStats, years: float = 3.0) -> WSRuleResult:
    title = "نمو الإيراد خارج نطاق الأقران"
    severity = 'high'
    if df['revenue'].dropna().empty:
        return WSRuleResult("REV_PEER_OUTLIER", title, severity, False, None, None, "لا توجد بيانات كافية")
    # تقدير CAGR على آخر ~3 سنوات (12 ربع ≈ 3 سنوات إن توفر)
    series = df['revenue'].dropna().astype(float)
    if len(series) < 5:
        return WSRuleResult("REV_PEER_OUTLIER", title, severity, False, None, None, "بيانات قليلة لحساب CAGR")
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    yrs = max(1.0, len(series)/4.0)
    try:
        cagr = (last/first)**(1.0/yrs) - 1.0 if first>0 else np.nan
    except Exception:
        cagr = np.nan
    if pd.isna(cagr) or pd.isna(peers.revenue_cagr_median):
        return WSRuleResult("REV_PEER_OUTLIER", title, severity, False, cagr, None, "بدون أقران/إنحراف معياري")
    z = _ws_z(cagr, peers.revenue_cagr_median, peers.revenue_cagr_std)
    flagged = (z is not None) and (abs(z) >= 2.0)
    return WSRuleResult("REV_PEER_OUTLIER", title, severity, flagged, cagr, "|z| ≥ 2", f"CAGR={cagr:.2%}, z={z:.2f}")

def ws_rule_receivables_turnover_decline(df: pd.DataFrame) -> WSRuleResult:
    title = "تدهور دوران الذمم المدينة عبر فترات"
    severity = 'medium'
    ar = df['receivables'].astype(float)
    ar_avg = (ar + ar.shift(1)) / 2.0
    rt = _ws_ratio(df['revenue'], ar_avg)
    slope = _ws_trend_slope(rt.dropna())
    flagged = (slope is not None) and (slope < 0) and (abs(slope) > 0.01)
    rationale = f"slope={slope:.4f}" if slope is not None else "بيانات غير كافية"
    return WSRuleResult("AR_TURN_DECLINE", title, severity, bool(flagged), float(slope) if slope is not None else None, "slope < -0.01", rationale)

def ws_rule_asset_turnover_down_with_acq(df: pd.DataFrame) -> WSRuleResult:
    title = "هبوط دوران الأصول مع نشاط استحواذ"
    severity = 'high'
    ta = df['total_assets'].astype(float)
    ta_avg = (ta + ta.shift(1)) / 2.0
    at = df['revenue'].astype(float) / ta_avg.replace(0, np.nan)
    slope = _ws_trend_slope(at.dropna())
    acq_spike = df['acq_cash_outflow'].fillna(0).astype(float).tail(8).sum() < -1e-6
    flagged = (slope is not None) and (slope < 0) and acq_spike
    rationale = f"AT_slope={slope:.4f}, استحواذات حديثة={acq_spike}"
    return WSRuleResult("ASSET_TURN_ACQ", title, severity, bool(flagged), float(slope) if slope is not None else None, "slope<0 مع استحواذات", rationale)

def ws_rule_other_income_in_revenue(df: pd.DataFrame) -> WSRuleResult:
    title = "بنود غير تشغيلية/لمرة واحدة ضمن الإيراد"
    severity = 'medium'
    rev = df['revenue'].astype(float)
    oi = df['other_income'].fillna(0).astype(float)
    ratio = (oi.rolling(4, min_periods=1).sum()) / (rev.rolling(4, min_periods=1).sum().replace(0, np.nan))
    val = float(ratio.iloc[-1]) if not ratio.dropna().empty else np.nan
    flagged = (not np.isnan(val)) and (val >= 0.05)
    rationale = f"other_income/ revenue (TTM) = {val:.2%}" if not np.isnan(val) else "غير متاح"
    return WSRuleResult("OTHER_IN_REV", title, severity, flagged, val, "≥ 5%", rationale)

def ws_rule_inventory_turnover_decline(df: pd.DataFrame) -> WSRuleResult:
    title = "انخفاض دوران المخزون عبر فترات"
    severity = 'medium'
    inv = df['inventory'].astype(float)
    inv_avg = (inv + inv.shift(1)) / 2.0
    cogs = df['cogs'].astype(float)
    it = _ws_ratio(cogs, inv_avg)
    slope = _ws_trend_slope(it.dropna())
    flagged = (slope is not None) and (slope < 0) and (abs(slope) > 0.01)
    rationale = f"slope={slope:.4f}" if slope is not None else "بيانات غير كافية"
    return WSRuleResult("INV_TURN_DECLINE", title, severity, bool(flagged), float(slope) if slope is not None else None, "slope < -0.01", rationale)

def ws_rule_cfo_ni_ratio(df: pd.DataFrame) -> WSRuleResult:
    title = "CFO/NI أقل من 1 أو يتدهور"
    severity = 'high'
    ni = df['net_income'].astype(float)
    cfo = df['cfo'].astype(float)
    ratio = (cfo.rolling(8, min_periods=4).sum()) / (ni.rolling(8, min_periods=4).sum().replace(0, np.nan))
    val = float(ratio.iloc[-1]) if not ratio.dropna().empty else np.nan
    trend = _ws_trend_slope(ratio.dropna())
    flagged = (not np.isnan(val) and val < 1.0) or (trend is not None and trend < 0)
    rationale = f"CFO/NI (rolling)={val:.2f}, trend={(f'{trend:.4f}' if trend is not None else 'NA')}"
    return WSRuleResult("CFO_NI", title, severity, flagged, val, "<1 أو اتجاه تنازلي", rationale)

def ws_rule_margins_outlier_vs_peers(df: pd.DataFrame, peers: WSPeerStats) -> List[WSRuleResult]:
    res=[]
    gm = None; om=None
    try:
        gp = (df['revenue'].astype(float) - df['cogs'].astype(float))
        gm_series = gp / df['revenue'].replace(0, np.nan).astype(float)
        gm = float(gm_series.dropna().iloc[-1]) if not gm_series.dropna().empty else None
    except Exception:
        pass
    try:
        om_series = df['operating_income'].astype(float) / df['revenue'].replace(0, np.nan).astype(float)
        om = float(om_series.dropna().iloc[-1]) if not om_series.dropna().empty else None
    except Exception:
        pass

    # GM
    gm_z = _ws_z(gm, peers.gross_margin_median, peers.gross_margin_std) if gm is not None else None
    gm_flag = (gm_z is not None) and (gm_z >= 2.0)
    res.append(WSRuleResult("GM_OUTLIER","هامش إجمالي مرتفع بشكل غير اعتيادي","medium",gm_flag,gm,"z ≥ 2", f"GM={to_percent(gm)} z={('NA' if gm_z is None else f'{gm_z:.2f}')}"))

    # OM
    om_z = _ws_z(om, peers.op_margin_median, peers.op_margin_std) if om is not None else None
    om_flag = (om_z is not None) and (om_z >= 2.0)
    res.append(WSRuleResult("OM_OUTLIER","هامش تشغيلي مرتفع بشكل غير اعتيادي","medium",om_flag,om,"z ≥ 2", f"OM={to_percent(om)} z={('NA' if om_z is None else f'{om_z:.2f}')}"))
    return res

def ws_rule_q4_anomaly(df: pd.DataFrame) -> WSRuleResult:
    title = "نمط ربع رابع غير اعتيادي"
    severity = 'low'
    if 'date' not in df.columns or df['date'].empty:
        return WSRuleResult("Q4_ANOM", title, severity, False, None, None, "لا يوجد تاريخ")
    rev = df[['date','revenue']].dropna()
    if rev.empty:
        return WSRuleResult("Q4_ANOM", title, severity, False, None, None, "لا توجد إيرادات")
    rev = rev.copy()
    rev['quarter'] = pd.to_datetime(rev['date']).dt.quarter
    q4s = rev[rev['quarter'] == 4]['revenue'].astype(float)
    others = rev[rev['quarter'] != 4]['revenue'].astype(float)
    if len(q4s) < 2 or len(others) < 4:
        return WSRuleResult("Q4_ANOM", title, severity, False, None, None, "بيانات غير كافية")
    q4_mean, others_mean, others_std = q4s.mean(), others.mean(), others.std()
    z = (q4_mean - others_mean) / (others_std if others_std != 0 else np.nan)
    flagged = (not np.isnan(z)) and (abs(z) >= 2.0)
    rationale = f"z={z:.2f} (Q4 مقابل بقية الفصول)"
    return WSRuleResult("Q4_ANOM", title, severity, flagged, float(z) if not np.isnan(z) else None, "|z| ≥ 2", rationale)

def ws_rule_disclosures(text: str) -> List[WSRuleResult]:
    items = [
        ("REV_METHODS","تغيّر طرق الاعتراف بالإيراد","high",'rev_change_methods'),
        ("BILL_HOLD","استخدام bill-and-hold","high",'bill_and_hold'),
        ("BARTER","معاملات مقايضة","high",'barter'),
        ("REBATES","برامج حسومات معقّدة","medium",'rebates'),
        ("RELATED","تعاملات مع أطراف ذات علاقة","high",'related_party'),
        ("NON_GAAP","تركيز مبالغ على Non-GAAP","medium",'non_gaap_push'),
        ("NONREC","بنود غير متكررة تتكرر","medium",'nonrecurring'),
        ("LIFO_LIQ","مؤشرات LIFO liquidation","medium",'lifo_reserve'),
        ("CAP_DEV","رسملة تكاليف تطوير/برمجيات","medium",'capitalized_dev'),
    ]
    out=[]
    for rid, title, severity, key in items:
        found = ws_search_disclosures(text, key)
        out.append(WSRuleResult(rid, title, severity, bool(found), None, "وجود إشارة بالإفصاح", f"match={found}"))
    return out

def ws_aggregate(results: List[WSRuleResult]) -> Tuple[int, pd.DataFrame]:
    penalty = 0
    rows=[]
    for r in results:
        w = _WS_SEVERITY_WEIGHT.get(r.severity,1)
        pen = (10 * w) if r.flagged else 0
        penalty += pen
        rows.append({
            "rule_id": r.rule_id,
            "العنوان": r.title,
            "الخطورة": r.severity,
            "Flag": r.flagged,
            "القيمة": r.metric_value,
            "الحد": r.threshold,
            "التبرير": r.rationale,
            "Penalty": pen
        })
    score = int(max(0, 100 - penalty))
    return score, pd.DataFrame(rows)

def ws_run_all_checks(df_financials: pd.DataFrame, peers: WSPeerStats, disclosures_text: str = "") -> Dict[str, Any]:
    results: List[WSRuleResult] = []
    results.append(ws_rule_revenue_outlier_vs_peers(df_financials, peers))
    results.append(ws_rule_receivables_turnover_decline(df_financials))
    results.append(ws_rule_asset_turnover_down_with_acq(df_financials))
    results.append(ws_rule_other_income_in_revenue(df_financials))
    results.append(ws_rule_inventory_turnover_decline(df_financials))
    results.append(ws_rule_cfo_ni_ratio(df_financials))
    results.extend(ws_rule_margins_outlier_vs_peers(df_financials, peers))
    results.append(ws_rule_q4_anomaly(df_financials))
    if disclosures_text:
        results.extend(ws_rule_disclosures(disclosures_text))

    score, df = ws_aggregate(results)
    summary = {
        'ManipulationRiskScore': score,  # 0 (خطر عالي) ← 100 (مطمئن)
        'flags_count': int(df['Flag'].sum()),
        'high_flags': int(((df['الخطورة']=='high') & (df['Flag'])).sum()),
        'medium_flags': int(((df['الخطورة']=='medium') & (df['Flag'])).sum()),
        'low_flags': int(((df['الخطورة']=='low') & (df['Flag'])).sum()),
    }
    return {'summary': summary, 'details': df.sort_values(['Flag','الخطورة'], ascending=[False, True])}

def ws_build_peers_from_symbols(symbols: List[str], suffix: str, mode: str) -> WSPeerStats:
    """نستخرج إحصائيات أقران بسيطة من الرموز المُدخلة (median/std)."""
    cagr_list=[]; gm_list=[]; om_list=[]; at_list=[]
    for c in symbols[:10]:
        try:
            cc = c if (suffix=="" or c.endswith(".SR")) else c+suffix
            d = load_company_data(cc)
            # CAGR إيرادات من السنوي
            inc = d["inc_a"]
            cols = sorted_cols(inc)
            if len(cols)>=2:
                first = find_any(inc, REV_KEYS, cols[-2])
                last  = find_any(inc, REV_KEYS, cols[-1])
                years = 1.0
                if len(cols)>=4:
                    first = find_any(inc, REV_KEYS, cols[3])
                    last  = find_any(inc, REV_KEYS, cols[0])
                    years = 3.0
                if first and first>0 and last and last>0:
                    cagr_list.append((last/first)**(1/years)-1)
            # هوامش + دوران أصول من core metrics
            rr = compute_core_metrics(d, mode)
            if not pd.isna(rr["GrossMargin"]): gm_list.append(float(rr["GrossMargin"]))
            if not pd.isna(rr["OperatingMargin"]): om_list.append(float(rr["OperatingMargin"]))
            if not pd.isna(rr["AssetTurnover"]): at_list.append(float(rr["AssetTurnover"]))
        except Exception:
            pass
    def medstd(v):
        v = [x for x in v if not pd.isna(x)]
        if not v: return (np.nan, np.nan)
        return (float(np.nanmedian(v)), float(np.nanstd(v, ddof=0)))
    c_med, c_std = medstd(cagr_list)
    gm_med, gm_std = medstd(gm_list)
    om_med, om_std = medstd(om_list)
    at_med, at_std = medstd(at_list)
    return WSPeerStats(
        revenue_cagr_median=c_med, revenue_cagr_std=c_std,
        gross_margin_median=gm_med, gross_margin_std=gm_std,
        op_margin_median=om_med, op_margin_std=om_std,
        asset_turnover_median=at_med, asset_turnover_std=at_std
    )

def ws_section_md(ws_out: Dict[str,Any]) -> str:
    if not ws_out or "summary" not in ws_out: return ""
    s=ws_out["summary"]; df=ws_out["details"]
    top_flags = df[df["Flag"]==True][["العنوان","الخطورة","التبرير"]].head(6).values.tolist() if isinstance(df,pd.DataFrame) else []
    tbl = md_table(["المؤشر","الخطورة","التبرير"], top_flags) if top_flags else "_لا توجد أعلام بارزة._"
    lines = [
        "## 12) مؤشرات إنذار/تلاعب (WS)",
        f"- **درجة المخاطر (0=خطر، 100=مطمئن):** {s['ManipulationRiskScore']}",
        f"- High/Med/Low: {s['high_flags']}/{s['medium_flags']}/{s['low_flags']}",
        "", tbl
    ]
    return "\n".join(lines)

# =============================
# نصوص مساعدة للتقرير
# =============================
def executive_summary(sym, info, r, score, verdict, dcf_value_ps, price):
    sector = info.get("sector") or "—"; industry = info.get("industry") or "—"
    lines = [
        f"**الشركة/الرمز:** {info.get('longName') or sym} ({sym}) — القطاع: {sector} | الصناعة: {industry}",
        f"- **أبرز النتائج:** هامش إجمالي {to_percent(r['GrossMargin'])}، ROIC {to_percent(r['ROIC'])}، OCF/NI {to_ratio(r['OCF/NI'])}، CCC {to_days(r['CCC'])}.",
        f"- **السيولة/الملاءة:** Current {to_ratio(r['CurrentRatio'])}، Quick {to_ratio(r['QuickRatio'])}، D/E {to_ratio(safe_div(r['TotalDebt'], r['TotalEquity']))}.",
        f"- **القوة المالية:** صافي الدين {to_num((r['TotalDebt']-(0 if pd.isna(r['Cash']) else r['Cash'])) if not pd.isna(r['TotalDebt']) else np.nan)}، تغطية الفوائد {to_ratio(r['InterestCoverage'])}."
    ]
    if not pd.isna(dcf_value_ps) and not pd.isna(price) and price>0:
        disc = (dcf_value_ps/price)-1
        lines.append(f"- **التقييم (DCF مبسّط):** القيمة الجوهرية/سهم ≈ {to_num(dcf_value_ps)} مقابل السعر {to_num(price)} → هامش أمان {to_percent(disc)}.")
    lines.append(f"**الخلاصة:** درجة بافيت {score:.0f}/100 — {verdict}.")
    return "\n".join(lines)

def company_overview(info):
    nm = info.get("longName") or "—"
    parts = [
        f"**الاسم:** {nm}",
        f"**القطاع/الصناعة:** {info.get('sector') or '—'} / {info.get('industry') or '—'}",
        f"**الموظفون:** {info.get('fullTimeEmployees') or '—'} | **الموقع:** {info.get('city') or '—'}, {info.get('country') or '—'}"
    ]
    if info.get("website"): parts.append(f"**الموقع:** {info.get('website')}")
    if info.get("longBusinessSummary"):
        s = info.get("longBusinessSummary"); parts.append(f"**وصف مختصر:** {s[:1200]}{'…' if len(s)>1200 else ''}")
    return "\n".join(parts)

def build_report_md(sym, info, r, score, verdict, dcf_ps, price, reasons, components, mode, trend_df,
                    dcf_table, base_ps, best_ps, worst_ps, comps_rows, data, ws_md_section: str = ""):
    # ترويسة ومعلومات عامة
    currency = info.get("financialCurrency") or info.get("currency") or "—"
    meta = r.get("_meta", {})
    header = [
        f"# تقرير تحليل مالي مفصل — {info.get('longName') or sym}",
        f"**الرمز:** {sym} | **العملة:** {currency} | **وضع الفترة:** {mode}",
        f"**فترات البيانات:** دخل: {meta.get('income_period','—')} | ميزانية: {meta.get('balance_period','—')} | تدفق نقدي: {meta.get('cashflow_period','—')}",
        "",
        "## 1) ملخص تنفيذي",
        executive_summary(sym, info, r, score, verdict, dcf_ps, price),
        "",
        "## 2) نظرة عامة على الشركة",
        company_overview(info),
    ]

    # 3) تحليل القوائم
    bs_tbl = md_table(
        ["البند","القيمة"],
        [
            ["الأصول المتداولة", to_num(r["CurrentAssets"])],
            ["الخصوم المتداولة", to_num(r["CurrentLiabilities"])],
            ["حقوق الملكية", to_num(r["TotalEquity"])],
            ["النقد", to_num(r["Cash"])],
            ["الاستثمارات القصيرة", to_num(r["STInvest"])],
            ["إجمالي الأصول", to_num(r["TotalAssets"])],
            ["إجمالي الدين", to_num(r["TotalDebt"])],
        ]
    )
    is_tbl = md_table(
        ["البند","القيمة","شرح"],
        [
            ["الإيرادات", to_num(r["Revenue"]), "مبيعات/دخل تشغيلي قبل خصم التكاليف."],
            ["الربح الإجمالي", to_num(r["GrossProfit"]), "الإيرادات – تكلفة المبيعات."],
            ["EBIT", to_num(r["EBIT"]), "ربح قبل الفوائد والضرائب (مقياس تشغيل)."],
            ["صافي الربح", to_num(r["NetIncome"]), "الربح بعد كل المصروفات والضرائب."],
            ["الهامش الإجمالي", to_percent(r["GrossMargin"]), "قوة التسعير/الخندق."],
            ["هامش التشغيل", to_percent(r["OperatingMargin"]), "كفاءة العمليات."],
            ["هامش صافي", to_percent(r["NetMargin"]), "ربحية شاملة."],
        ]
    )
    cf_tbl = md_table(
        ["البند","القيمة","شرح"],
        [
            ["التشغيلي (OCF)", to_num(r["OCF"]), "نقد متولد من الأعمال الأساسية."],
            ["Capex", to_num(r["Capex"]), "استثمارات رأسمالية."],
            ["أرباح المالك (OE)", to_num(r["OwnerEarnings"]), "≈ OCF - Capex (تبسيط)."],
        ]
    )

    # 4) نسب مالية مع شرح
    ratios_tbl = md_table(
        ["الفئة","النسبة","تفسير سريع"],
        [
            ["الربحية: Gross", to_percent(r["GrossMargin"]), "≥25% قوي؛ يعكس تسعير/كلفة."],
            ["الربحية: Net", to_percent(r["NetMargin"]), "صافي هامش الربح النهائي."],
            ["ROA", to_percent(r["ROA"]), "عائد على الأصول."],
            ["ROE", to_percent(r["ROE"]), "عائد على حقوق الملكية."],
            ["ROIC", to_percent(r["ROIC"]), "عائد على رأس المال المستثمر (مفتاح بافيت)."],
            ["السيولة: Current", to_ratio(r["CurrentRatio"]), "قدرة تغطية الخصوم الجارية."],
            ["السيولة: Quick", to_ratio(r["QuickRatio"]), "سيولة أكثر تحفظاً."],
            ["المديونية: D/E", to_ratio(safe_div(r["TotalDebt"], r["TotalEquity"])), "كلما أقل كان أفضل."],
            ["تغطية الفوائد", to_ratio(r["InterestCoverage"]), "≥10x مريح."],
            ["الكفاءة: دوران الأصول", to_ratio(r["AssetTurnover"]), "ناتج/أصل."],
            ["DSO", to_days(r["DSO"]), "أيام التحصيل."],
            ["DIO", to_days(r["DIO"]), "أيام المخزون."],
            ["DPO", to_days(r["DPO"]), "أيام السداد."],
            ["CCC", to_days(r["CCC"]), "أقل أفضل (≤0 مثالي)."],
            ["السوق: P/E", ("—" if pd.isna(r["PE"]) else f"{r['PE']:.2f}x"), "تقييم نسبي."],
            ["السوق: P/B", ("—" if pd.isna(r["PB"]) else f"{r['PB']:.2f}x"), "تقييم مقابل الدفترية."],
            ["BVPS", to_num(r["BVPS"]), "القيمة الدفترية للسهم."],
        ]
    )

    # 5) اتجاهات (ملخص نصي)
    trend_lines = []
    if isinstance(trend_df, pd.DataFrame) and not trend_df.empty:
        def dir_txt(series):
            vals = series.dropna().values
            if len(vals)<2: return "—"
            change = safe_div(vals[-1]-vals[0], abs(vals[0]) if vals[0]!=0 else np.nan)
            return to_percent(change)
        trend_lines.append(f"- الإيرادات (تغير من أول سنة لأحدث): {dir_txt(trend_df.loc['Revenue'])}")
        trend_lines.append(f"- صافي الربح: {dir_txt(trend_df.loc['NetIncome'])}")
        trend_lines.append(f"- أرباح المالك: {dir_txt(trend_df.loc['OwnerEarnings'])}")
    trends_section = "\n".join(trend_lines) if trend_lines else "لا تتوفر بيانات تاريخية كافية."

    # 6) قائمة تحقق بافيت — جدول نقاط + تبرير
    comp_rows = [[c["البند"], c["الرمز"], c["النقاط"], c["التفسير"]] for c in components]
    buffett_tbl = md_table(["البند","التقييم","النقاط","المبررات"], comp_rows)

    # 7) تقييم (DCF) + حساسية
    dcf_rows = dcf_table.to_dict("records") if isinstance(dcf_table, pd.DataFrame) and not dcf_table.empty else []
    dcf_md = md_table(["السنة","التدفق المتوقع","القيمة الحالية"],
                      [[str(rw["السنة"]), to_num(rw["التدفق المتوقع"]), to_num(rw["القيمة الحالية"])] for rw in dcf_rows]) if dcf_rows else "لا تتوفر تفاصيل التدفقات (تحقق من المدخلات)."

    sens_tbl = md_table(
        ["السيناريو","القيمة الجوهرية/سهم"],
        [
            ["أساسي (r,g)", to_num(dcf_ps)],
            ["أفضل (r-2%, g+2%)", to_num(best_ps)],
            ["أسوأ (r+2%, g-2%)", to_num(worst_ps)],
        ]
    )

    # 8) مقارنات (إن وُجدت)
    comps_md = ""
    if comps_rows:
        comps_md = md_table(
            ["الرمز","P/E","P/B","ROE","ROIC","هامش صافي"],
            [[row["الرمز"], row["P/E"], row["P/B"], row["ROE"], row["ROIC"], row["هامش صافي"]] for row in comps_rows]
        )

    # 9) مخاطر + 10) توصيات
    risks = []
    if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0: risks.append("سيولة جارية ضعيفة (<1.0).")
    if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0: risks.append("تغطية فوائد منخفضة (<6x).")
    if not pd.isna(r["CCC"]) and r["CCC"]>30: risks.append("دورة تحويل نقدي بطيئة (>30 يوم).")
    if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]<=0: risks.append("تدفقات حرة سلبية/ضعيفة.")
    risks_md = "\n".join([f"- {x}" for x in risks]) if risks else "- لا توجد مخاطر بارزة من البيانات المتاحة."

    recs=[]
    if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]>0 and (score>=75):
        recs.append("السهم جذّاب وفق مبادئ القيمة مع هامش أمان معقول.")
    if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0:
        recs.append("تعزيز السيولة (زيادة النقد/خفض الالتزامات الجارية).")
    if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0:
        recs.append("خفض الدين أو رفع الربحية لزيادة تغطية الفوائد.")
    if not pd.isna(r["CCC"]) and r["CCC"]>30:
        recs.append("تحسين رأس المال العامل (تحصيل أسرع، مخزون أخف، شروط دفع أطول).")
    if not recs:
        recs.append("لا توجد توصيات تشغيلية مُلحة استنادًا للبيانات المتاحة.")
    recs_md = "\n".join([f"- {x}" for x in recs])

    appendix = """
**المنهجية (مختصر):**
- تم الاعتماد على Yahoo Finance عبر yfinance وقد تختلف تسمية البنود بين الشركات.
- **TTM** = مجموع آخر 4 أرباع لقائمة الدخل/التدفق النقدي، وأحدث ميزانية متاحة.
- **ROIC (تقريبي)** = NOPAT / (الدين + حقوق – النقد) حيث NOPAT ≈ EBIT×(1–الضريبة).
- **أرباح المالك (Owner Earnings)** ≈ OCF – Capex (تبسيط لا يفرّق Capex الصيانة/النمو).
- حدود بافيت المستخدمَة عامة؛ قد تُعدّل حسب القطاع وطبيعة الأعمال.

**مسرد مختصر:**
- **OCF**: التدفق النقدي التشغيلي.
- **Capex**: الإنفاق الرأسمالي.
- **OE**: أرباح المالك = OCF – Capex.
- **CCC**: دورة التحويل النقدي = DSO + DIO – DPO (أقل أفضل).
- **OE Yield**: OE / القيمة السوقية (كلما أعلى كان أفضل).

**المعاملات بالمقايضة (Barter):** تبادل سلع/خدمات بدون نقد؛ قد تضخّم الإيرادات المحاسبية إن ذُكرت دون تدفق نقدي مقابل.
"""

    # تجميع كل الأقسام
    sections = []
    sections += header
    sections += [
        "",
        "## 3) تحليل القوائم المالية",
        "### الميزانية العمومية (مختصر)", bs_tbl,
        "", "### قائمة الدخل (مختصر)", is_tbl,
        "", "### قائمة التدفقات النقدية (مختصر)", cf_tbl,
        "",
        "## 4) النسب المالية (مشروحة)", ratios_tbl,
        "",
        "## 5) تحليل الاتجاهات (ملخص)",
        trends_section,
        "",
        "## 6) قائمة تحقق بافيت — نقاط وتبريرات",
        buffett_tbl,
        "",
        "## 7) التقييم بطريقة التدفقات المخصومة (DCF)",
        f"- **السعر الحالي:** {to_num(price)} | **القيمة الجوهرية/سهم (أساسي):** {to_num(dcf_ps)}",
        "", "### جدول التدفقات والقيمة الحالية", dcf_md,
        "", "### حساسية مبسّطة", sens_tbl,
        "",
        "## 8) مقارنة شركات (إن وُجدت)",
        (comps_md if comps_md else "_لم تُحدَّد مقارنات._"),
        "",
        "## 9) تقييم المخاطر",
        risks_md,
        "",
        "## 10) التوصيات والاستنتاج",
        recs_md,
        "",
        "## 11) الملاحق",
        appendix.strip(),
    ]
    if ws_md_section:
        sections += ["", ws_md_section]
    sections += ["", "_المصدر: Yahoo Finance عبر yfinance. هذا التقرير لأغراض تعليمية/بحثية وليس توصية استثمارية._"]
    return "\n".join(sections)

# =============================
# واجهة المستخدم
# =============================
st.markdown("<div class='hero'><h1>📊 نموذج التحليل المالي (مستلهَم من مبادئ بافيت) + مؤشرات إنذار (WS)</h1><div class='muted'>واجهة محسّنة + تقرير Markdown مفصل للتحميل + نظام كشف تحذيرات جودة الأرباح/التلاعب</div></div>", unsafe_allow_html=True)

with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1)
    simple_mode = st.toggle("وضع مبسّط (يناسب غير المتخصص)", value=True)
    st.markdown("---")
    st.markdown("#### إعدادات DCF")
    disc_rate = st.number_input("معدل الخصم (r)", 0.05, 0.30, 0.12, 0.01)
    growth_rate = st.number_input("نمو السنوات (g)", 0.00, 0.30, 0.05, 0.01)
    years = st.number_input("عدد السنوات", 3, 10, 5, 1)
    term_growth = st.number_input("نمو نهائي (gₜ)", 0.00, 0.05, 0.02, 0.005)
    st.caption("تلميح: r > gₜ وإلا يفشل التقييم.")
    st.markdown("---")
    comps_input = st.text_input("مقارنات (اختياري، رموز بمسافة/سطر)", "")
    st.markdown("---")
    st.markdown("#### أمثلة")
    if st.button("USA: AAPL"): st.session_state.syms = "AAPL"
    if st.button("KSA: 1120"): st.session_state.syms = "1120"

symbols_input = st.text_input("أدخل رمزًا واحدًا:", st.session_state.get("syms","")).strip()

# تجهيز الرمز
if symbols_input:
    sym = symbols_input.upper()
    if suffix and sym.isalnum() and not sym.endswith(".SR"): sym = sym + suffix
else:
    sym = ""

if st.button("🚀 تحليل الشركة"):
    if not sym:
        st.warning("يرجى إدخال رمز واحد.")
        st.stop()

    with st.spinner("جاري تحميل البيانات وتحليلها..."):
        data = load_company_data(sym)
        r = compute_core_metrics(data, mode)
        score, flags, verdict, net_debt, reasons, components = buffett_scorecard(r)
        trend_df = historical_trends(data["inc_a"], data["cf_a"], years=5)
        dcf_total, dcf_table = simple_dcf(r["OwnerEarnings"], disc_rate, growth_rate, int(years), term_growth)
        dcf_per_share = (dcf_total / r["Shares"]) if (not pd.isna(dcf_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan

        # حساسية
        best_total,_ = simple_dcf(r["OwnerEarnings"], disc_rate-0.02, growth_rate+0.02, int(years), term_growth)
        worst_total,_= simple_dcf(r["OwnerEarnings"], disc_rate+0.02, growth_rate-0.02, int(years), term_growth)
        best_ps = (best_total/r["Shares"]) if (not pd.isna(best_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan
        worst_ps= (worst_total/r["Shares"]) if (not pd.isna(worst_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan

        # === WS: peers + timeseries + run checks ===
        comps_raw = [c.strip().upper() for c in comps_input.replace("\n"," ").split() if c.strip()]
        ws_peers = ws_build_peers_from_symbols(comps_raw, suffix, mode) if comps_raw else WSPeerStats()
        df_fin_ts = ws_build_financials_ts(data, max_points=12)
        disclosures_text = (data.get("info", {}) or {}).get("longBusinessSummary","") or ""
        ws_out = ws_run_all_checks(df_fin_ts, ws_peers, disclosures_text)

    # ======== KPIs ========
    st.markdown("### المؤشرات الرئيسية")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.markdown(kpi_card("ROIC", to_percent(r["ROIC"]), "≥15% أفضلية", classify(r["ROIC"], ok=0.15, mid=0.10)), unsafe_allow_html=True)
    with cc2: st.markdown(kpi_card("الهامش الإجمالي", to_percent(r["GrossMargin"]), "≥25% قوي", classify(r["GrossMargin"], ok=0.25, mid=0.18)), unsafe_allow_html=True)
    with cc3: st.markdown(kpi_card("OCF/NI", to_ratio(r["OCF/NI"]), "≥1.0 جودة أرباح", classify(r["OCF/NI"], ok=1.0, mid=0.8)), unsafe_allow_html=True)
    with cc4: st.markdown(kpi_card("CCC", to_days(r["CCC"]), "أقل أفضل", classify(r["CCC"], ok=0, mid=30, reverse=True)), unsafe_allow_html=True)

    dd1, dd2, dd3, dd4 = st.columns(4)
    with dd1:
        de_class = classify(safe_div(r["TotalDebt"], r["TotalEquity"]), ok=0.5, mid=1.0, reverse=True)
        st.markdown(kpi_card("D/E", to_ratio(safe_div(r["TotalDebt"], r["TotalEquity"])), "≤0.5 مريح", de_class), unsafe_allow_html=True)
    with dd2: st.markdown(kpi_card("هامش OE", to_percent(r["FCF_Margin"]), "≥8% جيد", classify(r["FCF_Margin"], ok=0.08, mid=0.05)), unsafe_allow_html=True)
    with dd3: st.markdown(kpi_card("تغطية الفوائد", to_ratio(r["InterestCoverage"]), "≥10x آمن", classify(r["InterestCoverage"], ok=10, mid=6)), unsafe_allow_html=True)
    with dd4: st.markdown(kpi_card("OE Yield", to_percent(r["OwnerEarningsYield"]), "≥6% معقول", classify(r["OwnerEarningsYield"], ok=0.06, mid=0.04)), unsafe_allow_html=True)

    # WS KPI
    st.markdown("### مؤشرات الإنذار (WS)")
    w1, w2, w3, w4 = st.columns(4)
    ws_sum = ws_out["summary"]
    with w1: st.markdown(kpi_card("WS Risk Score", str(ws_sum["ManipulationRiskScore"]), "0=خطر عالي • 100=اطمئنان", ("ok" if ws_sum["ManipulationRiskScore"]>=80 else ("mid" if ws_sum["ManipulationRiskScore"]>=60 else "bad"))), unsafe_allow_html=True)
    with w2: st.markdown(kpi_card("High Flags", str(ws_sum["high_flags"]), "أعلام خطرة", ("bad" if ws_sum["high_flags"]>0 else "ok")), unsafe_allow_html=True)
    with w3: st.markdown(kpi_card("Medium Flags", str(ws_sum["medium_flags"]), "أعلام متوسطة", ("mid" if ws_sum["medium_flags"]>0 else "ok")), unsafe_allow_html=True)
    with w4: st.markdown(kpi_card("Low Flags", str(ws_sum["low_flags"]), "أعلام منخفضة", "ok"), unsafe_allow_html=True)

    st.write("**درجة بافيت:** ", f"{score:.0f}/100 — {verdict}")
    st.progress(min(max(int(score), 0), 100)/100)

    # ======== تبويبات ========
    tabs = st.tabs([
        "1) ملخص تنفيذي", "2) نظرة عامة", "3) القوائم (BS/IS/CF)",
        "4) النسب", "5) الاتجاهات", "6) المخاطر", "7) التقييم", "8) مقارنات",
        "9) الأسباب/التحقق", "10) تقرير للتنزيل", "11) مؤشرات إنذار/تلاعب"
    ])

    with tabs[0]:
        st.markdown(executive_summary(sym, data.get("info", {}), r, score, verdict, dcf_per_share, r["Price"]))
        if simple_mode: st.caption("تم تبسيط العرض. عطّل 'وضع مبسّط' لعرض مزيد من التفاصيل.")

    with tabs[1]:
        st.markdown(company_overview(data.get("info", {})))

    with tabs[2]]:
        cA, cB = st.columns(2)
        with cA:
            st.markdown("### قائمة الدخل")
            is_rows = [{
                "الإيرادات": to_num(r["Revenue"]),
                "الربح الإجمالي": to_num(r["GrossProfit"]),
                "EBIT": to_num(r["EBIT"]),
                "صافي الربح": to_num(r["NetIncome"]),
                "الهامش الإجمالي": to_percent(r["GrossMargin"]),
                "هامش التشغيل": to_percent(r["OperatingMargin"]),
                "هامش صافي": to_percent(r["NetMargin"])
            }]
            st.dataframe(pd.DataFrame(is_rows), use_container_width=True)
        with cB:
            st.markdown("### الميزانية العمومية")
            bs_rows = [{
                "الأصول المتداولة": to_num(r["CurrentAssets"]),
                "الخصوم المتداولة": to_num(r["CurrentLiabilities"]),
                "حقوق الملكية": to_num(r["TotalEquity"]),
                "النقد": to_num(r["Cash"]),
                "الاستثمارات القصيرة": to_num(r["STInvest"]),
                "إجمالي الأصول": to_num(r["TotalAssets"]),
                "إجمالي الدين": to_num(r["TotalDebt"]),
            }]
            st.dataframe(pd.DataFrame(bs_rows), use_container_width=True)
        st.markdown("### التدفقات النقدية")
        cf_rows = [{
            "تشغيلي OCF": to_num(r["OCF"]),
            "Capex": to_num(r["Capex"]),
            "أرباح المالك (OE)": to_num(r["OwnerEarnings"])
        }]
        st.dataframe(pd.DataFrame(cf_rows), use_container_width=True)
        st.caption("(*) تبسيطات بسبب اختلاف تفصيل البنود في Yahoo Finance.")

    with tabs[3]:
        ratios_tbl = [{
            "Gross": to_percent(r["GrossMargin"]), "Net": to_percent(r["NetMargin"]),
            "ROA": to_percent(r["ROA"]), "ROE": to_percent(r["ROE"]), "ROIC": to_percent(r["ROIC"]),
            "Current": to_ratio(r["CurrentRatio"]), "Quick": to_ratio(r["QuickRatio"]),
            "D/E": to_ratio(safe_div(r["TotalDebt"], r["TotalEquity"])),
            "Interest Cov": to_ratio(r["InterestCoverage"]),
            "Asset Turn": to_ratio(r["AssetTurnover"]),
            "DSO": to_days(r["DSO"]), "DIO": to_days(r["DIO"]), "DPO": to_days(r["DPO"]),
            "P/E": "—" if pd.isna(r["PE"]) else f"{r['PE']:.2f}x",
            "P/B": "—" if pd.isna(r["PB"]) else f"{r['PB']:.2f}x",
            "BVPS": to_num(r["BVPS"])
        }]
        st.dataframe(pd.DataFrame(ratios_tbl), use_container_width=True)

    with tabs[4]:
        st.caption("خطوط بسيطة توضح الاتجاه العام (إيرادات/صافي ربح/أرباح المالك).")
        try:
            st.line_chart(trend_df.T)
        except Exception:
            st.info("لا تتوفر بيانات تاريخية كافية للرسم.")

    with tabs[5]:
        risks = []
        if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0: risks.append("سيولة جارية دون 1.0 قد تضغط على السداد القصير.")
        if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0: risks.append("تغطية فوائد منخفضة تُعلي حساسية الفائدة/الأرباح.")
        if not pd.isna(r["CCC"]) and r["CCC"]>30: risks.append("دورة تحويل نقدي بطيئة (>30 يوم).")
        if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]<=0: risks.append("أرباح مالك ضعيفة/سلبية تحدّ من المرونة الاستثمارية.")
        st.write("- " + "\n- ".join(risks) if risks else "لا توجد مخاطر جوهرية ظاهرة من البيانات المتاحة.")

    with tabs[6]:
        st.markdown("### DCF مبسّط (على أرباح المالك)")
        if not pd.isna(dcf_total):
            st.dataframe(dcf_table, use_container_width=True)
            st.write("**القيمة الحالية الإجمالية (للشركة):**", to_num(dcf_total))
            if not pd.isna(dcf_per_share): st.write("**القيمة الجوهرية/سهم:**", to_num(dcf_per_share))
            st.write("**حساسية:** أفضل:", to_num(best_ps), " | أساسي:", to_num(dcf_per_share), " | أسوأ:", to_num(worst_ps))
        else:
            st.info("لا يمكن حساب DCF (تحقق من r>gₜ و OE>0).")

    with tabs[7]:
        comps_rows = []
        if comps_raw:
            with st.spinner("جاري جلب المقارنات..."):
                for c in comps_raw[:8]:
                    try:
                        cc = c if (suffix=="" or c.endswith(".SR")) else c+suffix
                        d = load_company_data(cc)
                        rr = compute_core_metrics(d, mode)
                        comps_rows.append({
                            "الرمز": cc,
                            "P/E": "—" if pd.isna(rr["PE"]) else f"{rr['PE']:.2f}",
                            "P/B": "—" if pd.isna(rr["PB"]) else f"{rr['PB']:.2f}",
                            "ROE": to_percent(rr["ROE"]),
                            "ROIC": to_percent(rr["ROIC"]),
                            "هامش صافي": to_percent(rr["NetMargin"])
                        })
                    except Exception as e:
                        comps_rows.append({"الرمز": c, "P/E":"—","P/B":"—","ROE":"—","ROIC":"—","هامش صافي":f"خطأ: {e}"})
            st.dataframe(pd.DataFrame(comps_rows), use_container_width=True)
        else:
            st.caption("أدخل رموزًا للمقارنة في الشريط الجانبي.")
        st.session_state._comps_rows = comps_rows  # لاستخدامها في التقرير

    with tabs[8]:
        df_flags = pd.DataFrame([{"البند":k, "التقييم":v} for k,v in dict(sorted(flags.items())).items()])
        st.markdown("**قائمة تحقق بافيت:**")
        st.dataframe(df_flags, use_container_width=True)
        st.markdown("**الأسباب التفصيلية:**")
        st.dataframe(pd.DataFrame(reasons), use_container_width=True)

    with tabs[9]:
        comps_rows = st.session_state.get("_comps_rows", [])
        ws_md = ws_section_md(ws_out)
        report_md = build_report_md(
            sym, data.get("info", {}), r, score, verdict, dcf_per_share, r["Price"], reasons, components,
            mode, trend_df, dcf_table, dcf_per_share, best_ps, worst_ps, comps_rows, data, ws_md_section=ws_md
        )
        st.download_button("📥 تنزيل التقرير المفصل (Markdown)", report_md.encode("utf-8"),
                           file_name=f"Detailed_Financial_Report_{sym}.md", mime="text/markdown")
        st.caption("يشمل: ملخص تنفيذي، نظرة عامة، تحليل القوائم، نسب مشروحة، بافيت (نقاط/مبررات)، اتجاهات، تقييم DCF، حساسية، مخاطر، توصيات، WS إنذارات.")

    with tabs[10]:
        st.markdown("#### ملخص مؤشرات الإنذار (WS)")
        st.write(f"**WS Risk Score:** {ws_sum['ManipulationRiskScore']} — High/Med/Low: {ws_sum['high_flags']}/{ws_sum['medium_flags']}/{ws_sum['low_flags']}")
        st.dataframe(ws_out["details"], use_container_width=True)
        with st.expander("ما هي المعاملات بالمقايضة؟"):
            st.markdown("- **المقايضة (Barter):** تبادل سلع/خدمات بدون نقد. قد ترفع الإيرادات المحاسبية دون دعم نقدي موازٍ.")
        st.caption("ملاحظة: قواعد WS لا تعني وجود تلاعب بالضرورة؛ هي إشارات تحتاج تدقيق إضافي (DD).")
# دليل مبسّط
with st.expander("ℹ️ ماذا تعني المؤشرات؟"):
    st.markdown("""
- **ROIC**: كفاءة تحويل رأس المال إلى أرباح تشغيلية بعد الضرائب.
- **OCF/NI**: جودة الأرباح؛ ≥1.0 يعني أن النقد يدعم الربح المحاسبي.
- **CCC**: زمن دورة النقد؛ أقل أفضل.
- **OE Yield**: أرباح المالك/القيمة السوقية؛ مقياس لعائد ضمني.
- **WS Risk Score**: درجة مخاطر التلاعب/الجودة (0=خطر عالي، 100=اطمئنان). تزداد المخاطر بوجود Bill-and-Hold، Barter، تضخم Other Income، تدهور دوران الذمم/المخزون، إلخ.
- **DCF مبسّط**: تقدير أولي للقيمة الجوهرية اعتمادًا على OE وافتراضات محافظة.
""")
