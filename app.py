# -*- coding: utf-8 -*-
"""
📊 Buffett Principles Analyzer — Streamlit
مستند إلى مبادئ وارن بافيت: جودة الأعمال، القوة المالية، تحويل النقد، وإدارة رأس المال + تحليل نصي.
المكتبات: streamlit, yfinance, pandas, numpy (مسموح بها ضمن سياق المشروع).
تشغيل: streamlit run app.py
"""

import os
import re
import math
from datetime import datetime
from html import escape

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="📊 Buffett Principles Analyzer", layout="wide")
RTL_CSS = """
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; }
  input, textarea, select { direction: rtl; text-align: right; }
  .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
  .stNumberInput input, .stDateInput input, .stMultiSelect [data-baseweb],
  label, .stButton button { text-align: right; }
  table { direction: rtl; }
  .stAlert { direction: rtl; }
  .metric-card { background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px; }
  .buffett-table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
  .buffett-table th, .buffett-table td {border: 1px solid #ddd; padding: 8px; text-align: center;}
  .buffett-table th {background-color: #0ea5e9; color: white;}
  .buffett-table tr:nth-child(even){background-color: #f8fafc;}
  .buffett-table tr:hover {background-color: #eef2ff;}
  .buffett-table td.green { color: green; font-weight: bold; }
  .buffett-table td.yellow { color: #d97706; font-weight: bold; }
  .buffett-table td.red { color: red; font-weight: bold; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# =============================
# أدوات مساعدة
# =============================
def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def build_index_map(df: pd.DataFrame):
    idx = {}
    for raw in df.index.astype(str):
        idx[normalize_idx(raw)] = raw
    return idx

def find_any(df: pd.DataFrame, keys: list[str], col):
    if df is None or df.empty or col is None:
        return np.nan
    idx_map = build_index_map(df)
    for k in keys:
        key = normalize_idx(k)
        if key in idx_map:
            try:
                return float(df.loc[idx_map[key], col])
            except Exception:
                try:
                    return float(pd.to_numeric(df.loc[idx_map[key], col], errors="coerce"))
                except Exception:
                    return np.nan
    return np.nan

def sorted_cols(df: pd.DataFrame, reverse=True):
    try:
        return sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=reverse)
    except Exception:
        return list(df.columns)[::-1] if reverse else list(df.columns)

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
    if x is None or pd.isna(x):
        return "—"
    absx = abs(x)
    if absx >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.{digits}f}T"
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.{digits}f}B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.{digits}f}M"
    if absx >= 1_000:
        return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

# مرادفات شائعة لأسماء البنود في Yahoo
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

# =============================
# تحميل البيانات
# =============================
@st.cache_data(ttl=1800)
def load_company_data(ticker: str):
    t = yf.Ticker(ticker)
    def safe_df(getter, fallback=None):
        try:
            df = getter()
            return df if isinstance(df, pd.DataFrame) else (fallback if fallback is not None else pd.DataFrame())
        except Exception:
            return fallback if fallback is not None else pd.DataFrame()

    inc_a = safe_df(lambda: t.financials)
    inc_q = safe_df(lambda: t.quarterly_financials)
    bal_a = safe_df(lambda: t.balance_sheet)
    bal_q = safe_df(lambda: t.quarterly_balance_sheet)
    cf_a  = safe_df(lambda: t.cashflow)
    cf_q  = safe_df(lambda: t.quarterly_cashflow)

    # السعر والقيمة السوقية وعدد الأسهم
    price = np.nan
    shares_latest = np.nan
    market_cap = np.nan
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        shares_latest = float(fi.get("shares", np.nan))
        market_cap = float(fi.get("market_cap", np.nan))
    except Exception:
        pass

    # تاريخ عدد الأسهم إن توفر
    shares_hist = None
    try:
        shares_hist = t.get_shares_full(start="1995-01-01")  # قد لا تتوفر دائمًا
    except Exception:
        shares_hist = None

    return {
        "ticker": t, "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q, "cf_a": cf_a, "cf_q": cf_q,
        "price": price, "shares": shares_latest, "market_cap": market_cap, "shares_hist": shares_hist
    }

# =============================
# حساب النِّسب وفق مبادئ بافيت
# =============================
def compute_ttm(inc_q: pd.DataFrame, cf_q: pd.DataFrame):
    """تجميع 4 أرباع أخيرة لبنود الدخل والتدفقات النقدية."""
    def sum_last4(df, keys):
        if df is None or df.empty: return np.nan
        cols = sorted_cols(df)[:4]
        return sum([find_any(df, keys, c) for c in cols])
    rev_ttm = sum_last4(inc_q, REV_KEYS)
    ebit_ttm = sum_last4(inc_q, EBIT_KEYS) if not pd.isna(sum_last4(inc_q, EBIT_KEYS)) else sum_last4(inc_q, OPINC_KEYS)
    ni_ttm = sum_last4(inc_q, NI_KEYS)
    ocf_ttm = sum_last4(cf_q, OCF_KEYS)
    capex_ttm = sum_last4(cf_q, CAPEX_KEYS)
    return rev_ttm, ebit_ttm, ni_ttm, ocf_ttm, capex_ttm

def estimate_invested_capital(bal_df: pd.DataFrame, col):
    total_debt = find_any(bal_df, TOT_DEBT_KEYS, col)
    if pd.isna(total_debt):
        parts = []
        for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS):
            parts.append(find_any(bal_df, ks, col))
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan
    te = find_any(bal_df, TE_KEYS, col)
    cash = find_any(bal_df, CASH_KEYS, col)
    if pd.isna(total_debt) or pd.isna(te):
        return np.nan, np.nan, np.nan
    invested = total_debt + te - (cash if not pd.isna(cash) else 0.0)
    return invested, total_debt, cash

def owner_earnings(ocf, capex):
    if pd.isna(ocf) or pd.isna(capex): return np.nan
    return ocf - capex  # تقريب: OCF - Capex ~ Owner Earnings

def pct_stdev(series):
    series = [x for x in series if not pd.isna(x)]
    if len(series) < 2: return np.nan
    return float(np.std(series, ddof=1))

def cagr(first, last, years):
    try:
        if any(pd.isna(x) for x in [first, last]) or years <= 0 or first <= 0:
            return np.nan
        return (last/first) ** (1/years) - 1
    except Exception:
        return np.nan

def compute_buffett_ratios(data: dict, mode: str = "TTM"):
    inc_a, bal_a, cf_a = data["inc_a"], data["bal_a"], data["cf_a"]
    inc_q, bal_q, cf_q = data["inc_q"], data["bal_q"], data["cf_q"]

    if mode == "TTM" and not inc_q.empty:
        rev, ebit, ni, ocf, capex = compute_ttm(inc_q, cf_q)
        bal = bal_q if not bal_q.empty else bal_a
    else:
        # آخر سنة متاحة
        inc_cols = sorted_cols(inc_a)
        cf_cols = sorted_cols(cf_a)
        col = inc_cols[0] if inc_cols else None
        col_cf = cf_cols[0] if cf_cols else None
        rev = find_any(inc_a, REV_KEYS, col)
        ebit = find_any(inc_a, EBIT_KEYS, col)
        if pd.isna(ebit): ebit = find_any(inc_a, OPINC_KEYS, col)
        ni = find_any(inc_a, NI_KEYS, col)
        ocf = find_any(cf_a, OCF_KEYS, col_cf)
        capex = find_any(cf_a, CAPEX_KEYS, col_cf)
        bal = bal_a

    bal_cols = sorted_cols(bal)
    bal_curr = bal_cols[0] if bal_cols else None
    invested, total_debt, cash = estimate_invested_capital(bal, bal_curr)
    pbt = find_any(inc_a if mode!="TTM" else inc_q, PBT_KEYS, sorted_cols(inc_a)[0] if mode!="TTM" else sorted_cols(inc_q)[0] )
    tax = find_any(inc_a if mode!="TTM" else inc_q, TAX_KEYS, sorted_cols(inc_a)[0] if mode!="TTM" else sorted_cols(inc_q)[0] )
    eff_tax_rate = tax / pbt if (pbt and not pd.isna(pbt) and pbt!=0 and not pd.isna(tax)) else 0.25
    eff_tax_rate = float(np.clip(eff_tax_rate, 0.0, 0.6))
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan
    roic = safe_div(nopat, invested)

    # هوامش
    cogs = np.nan
    if mode == "TTM":
        cogs = compute_ttm(inc_q, cf_q)[0] - (find_any(inc_q, GP_KEYS, sorted_cols(inc_q)[0]) or 0) if not inc_q.empty else np.nan
    else:
        cogs = find_any(inc_a, COGS_KEYS, sorted_cols(inc_a)[0]) if not inc_a.empty else np.nan
    gp = (rev - cogs) if (not pd.isna(rev) and not pd.isna(cogs)) else np.nan
    gross_margin = safe_div(gp, rev)
    op_margin = safe_div(ebit, rev)
    net_margin = safe_div(ni, rev)

    # جودة الأرباح/النقد
    oe = owner_earnings(ocf, capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(oe, rev)

    # الدين والتغطية
    interest = abs(find_any(inc_a if mode!="TTM" else inc_q, INT_EXP_KEYS,
                            sorted_cols(inc_a)[0] if mode!="TTM" else sorted_cols(inc_q)[0]))
    interest_cov = safe_div(ebit, interest)

    # التحويل النقدي التشغيلي
    # CCC: نستخدم أحدث ميزانية وأرقام الدخل المستخدمة
    ar = find_any(bal, AR_KEYS, bal_curr)
    ap = find_any(bal, AP_KEYS, bal_curr)
    inv = find_any(bal, INV_KEYS, bal_curr)
    # متوسطات مبسطة (الحالي فقط إذا لم تتوفر السابقة)
    bal_prev = bal_cols[1] if len(bal_cols) > 1 else None
    ar_avg = np.nanmean([ar, find_any(bal, AR_KEYS, bal_prev)])
    ap_avg = np.nanmean([ap, find_any(bal, AP_KEYS, bal_prev)])
    inv_avg = np.nanmean([inv, find_any(bal, INV_KEYS, bal_prev)])

    rev_used = rev if not pd.isna(rev) else np.nan
    cogs_used = cogs if not pd.isna(cogs) else rev_used

    rec_turn = safe_div(rev_used, ar_avg)
    pay_turn = safe_div(cogs_used, ap_avg)
    inv_turn = safe_div(cogs_used, inv_avg)
    dso = safe_div(365, rec_turn)
    dpo = safe_div(365, pay_turn)
    dio = safe_div(365, inv_turn)
    ccc = dso + dio - dpo if not any(pd.isna(x) for x in [dso, dio, dpo]) else np.nan

    # أسهم قائمة (اتجاه إعادة الشراء)
    shares_now = data.get("shares", np.nan)
    shares_trend = np.nan
    try:
        if isinstance(data.get("shares_hist"), pd.Series) and data["shares_hist"].size >= 2:
            s_hist = data["shares_hist"].dropna()
            if s_hist.size >= 2:
                first = float(s_hist.iloc[0])
                last = float(s_hist.iloc[-1])
                years = max(1, (s_hist.index[-1].year - s_hist.index[0].year))
                shares_trend = (last - first) / first  # نسبة التغير التراكمية
    except Exception:
        pass

    # تقييم مبسّط
    market_cap = data.get("market_cap", np.nan)
    if (pd.isna(market_cap) or market_cap == 0) and (not pd.isna(data.get("price")) and not pd.isna(shares_now)):
        market_cap = data["price"] * shares_now
    owner_earnings_yield = safe_div(oe, market_cap)
    p_to_oe = safe_div(market_cap, oe)

    # حِزَم نتائج
    ratios = {
        "Revenue": rev, "EBIT": ebit, "NetIncome": ni,
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "NOPAT": nopat, "InvestedCapital": invested, "ROIC": roic,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": oe,
        "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "TotalDebt": total_debt, "Cash": cash, "InterestCoverage": interest_cov,
        "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        "SharesLatest": shares_now, "SharesTrend": shares_trend,
        "MarketCap": market_cap, "OwnerEarningsYield": owner_earnings_yield, "P/OwnerEarnings": p_to_oe
    }

    return ratios

# =============================
# ترقيم (Scoring) وفق قائمة تحقق بافيت
# =============================
def buffett_scorecard(r):
    # أوزان (مجموع 100)
    pts = 0
    details = {}

    def flag(name, ok, mid=None):
        nonlocal pts
        if ok is True:
            details[name] = "✅"
        elif mid is True:
            details[name] = "⚠️"
        else:
            details[name] = "❌"

    # 1) ROIC ≥ 15% وثبات معقول
    roic = r["ROIC"]
    ok_roic = (not pd.isna(roic) and roic >= 0.15)
    pts += 20 if ok_roic else 8 if (not pd.isna(roic) and roic >= 0.10) else 0
    flag("ROIC ≥15%", ok_roic, mid=(not pd.isna(roic) and roic >= 0.10))

    # 2) هامش إجمالي واستقراره (لا نحتسب الانحراف بدون سلسلة متعددة السنوات، نكتفي بالمستوى)
    gm = r["GrossMargin"]
    ok_gm = (not pd.isna(gm) and gm >= 0.25)
    pts += 10 if ok_gm else 5 if (not pd.isna(gm) and gm >= 0.18) else 0
    flag("هامش إجمالي قوي", ok_gm, mid=(not pd.isna(gm) and gm >= 0.18))

    # 3) جودة الأرباح: OCF/NI ≥ 1
    ocfni = r["OCF/NI"]
    ok_qual = (not pd.isna(ocfni) and ocfni >= 1.0)
    pts += 10 if ok_qual else 5 if (not pd.isna(ocfni) and ocfni >= 0.8) else 0
    flag("جودة الأرباح OCF/NI", ok_qual, mid=(not pd.isna(ocfni) and ocfni >= 0.8))

    # 4) هامش التدفق الحر (أرباح المالك/الإيراد) ≥ 8%
    fcfm = r["FCF_Margin"]
    ok_fcfm = (not pd.isna(fcfm) and fcfm >= 0.08)
    pts += 10 if ok_fcfm else 5 if (not pd.isna(fcfm) and fcfm >= 0.05) else 0
    flag("هامش التدفق الحر", ok_fcfm, mid=(not pd.isna(fcfm) and fcfm >= 0.05))

    # 5) صافي الدين متحفظ: NetDebt ≤ 0 أو Debt/OwnerEarnings ≤ 2
    total_debt = r["TotalDebt"]
    cash = r["Cash"]
    oe = r["OwnerEarnings"]
    net_debt = np.nan if pd.isna(total_debt) else total_debt - (0 if pd.isna(cash) else cash)
    crit = (not pd.isna(net_debt) and net_debt <= 0) or (not any(pd.isna(x) for x in [total_debt, oe]) and oe > 0 and total_debt / oe <= 2.0)
    mid_crit = (not any(pd.isna(x) for x in [total_debt, oe]) and oe > 0 and total_debt / oe <= 3.0)
    pts += 10 if crit else 5 if mid_crit else 0
    flag("هيكل دين متحفظ", crit, mid=mid_crit)

    # 6) تغطية الفوائد ≥ 10x
    ic = r["InterestCoverage"]
    ok_ic = (not pd.isna(ic) and ic >= 10.0)
    pts += 10 if ok_ic else 5 if (not pd.isna(ic) and ic >= 6.0) else 0
    flag("تغطية الفوائد", ok_ic, mid=(not pd.isna(ic) and ic >= 6.0))

    # 7) دورة التحويل النقدي ≤ 0 (أو منخفضة)
    ccc = r["CCC"]
    ok_ccc = (not pd.isna(ccc) and ccc <= 0)
    pts += 5 if ok_ccc else 2 if (not pd.isna(ccc) and ccc <= 30) else 0
    flag("دورة التحويل النقدي", ok_ccc, mid=(not pd.isna(ccc) and ccc <= 30))

    # 8) اتجاه عدد الأسهم: تناقص/ثبات
    sh_trend = r["SharesTrend"]  # نسبة تغير تراكمية (سالب أفضل)
    ok_sh = (not pd.isna(sh_trend) and sh_trend <= 0.0)
    pts += 5 if ok_sh else 2 if (not pd.isna(sh_trend) and sh_trend <= 0.05) else 0
    flag("انضباط إعادة الشراء/عدم التخفيف", ok_sh, mid=(not pd.isna(sh_trend) and sh_trend <= 0.05))

    # 9) هامش أمان في السعر: Owner Earnings Yield ≥ 6% أو P/OwnerEarnings ≤ 20
    oey = r["OwnerEarningsYield"]
    pto = r["P/OwnerEarnings"]
    crit_val = (not pd.isna(oey) and oey >= 0.06) or (not pd.isna(pto) and pto <= 20)
    mid_val = (not pd.isna(oey) and oey >= 0.04) or (not pd.isna(pto) and pto <= 25)
    pts += 10 if crit_val else 5 if mid_val else 0
    flag("تقييم معقول (OE Yield/ P-to-OE)", crit_val, mid=mid_val)

    score = float(pts)
    verdict = "✅ جذّابة مع هامش أمان" if score >= 75 else ("🟧 جيدة لكن انتظر سعرًا أفضل" if score >= 55 else "🕒 راقِب ولا تتعجل")
    return score, details, verdict, {"NetDebt": net_debt}

# =============================
# توليد التحليل النصي المستلهم من نهج بافيت
# =============================
def buffett_narrative(ticker, r, score, verdict):
    parts = []
    parts.append(f"**الرمز:** {ticker}")
    # جودة العمل
    gm = r["GrossMargin"]; roic = r["ROIC"]; fcfm = r["FCF_Margin"]
    qual = f"- **جودة العمل:** هامش إجمالي {to_percent(gm)}، وعائد على رأس المال المستثمر ROIC عند {to_percent(roic)}، وهامش تدفق حر {to_percent(fcfm)}."
    parts.append(qual)

    # القوة المالية
    net_debt = r.get("TotalDebt", np.nan) - (0 if pd.isna(r.get("Cash")) else r.get("Cash", np.nan)) if not pd.isna(r.get("TotalDebt")) else np.nan
    ic = r["InterestCoverage"]
    fin = f"- **القوة المالية:** صافي الدين {to_num(net_debt)}، وتغطية الفوائد {to_ratio(ic)}."
    parts.append(fin)

    # جودة الأرباح والتحويل النقدي
    ocfni = r["OCF/NI"]; ccc = r["CCC"]
    conv = f"- **جودة الأرباح:** OCF/NI عند {to_ratio(ocfni)}، و**دورة التحويل النقدي (CCC)** {to_days(ccc)}."
    parts.append(conv)

    # رأس المال والتقييم
    oey = r["OwnerEarningsYield"]; pto = r["P/OwnerEarnings"]
    capalloc = f"- **أرباح المالك والتقييم:** عائد أرباح المالك {to_percent(oey)}، ومضاعف السعر إلى أرباح المالك {to_ratio(pto)}."
    parts.append(capalloc)

    # الخلاصة
    parts.append(f"**النتيجة:** درجة {score:.0f}/100 — {verdict}.")
    return "\n".join(parts)

# =============================
# واجهة المستخدم
# =============================
st.title("📊 التحليل الأساسي بمبادئ بافيت (Buffett Principles)")
st.caption("تحليل مُستلهم من مبادئ وارن بافيت: قياس جودة الأعمال، قوة الميزانية، تحويل النقد، والتقييم — مع تحليلٍ نصي موجز.")

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1, help="TTM = مجموع 4 أرباع أخيرة؛ Annual = آخر سنة مالية منشورة.")
    show_table = st.checkbox("عرض جدول النِّسَب", value=True)
    show_text = st.checkbox("عرض التحليل النصي", value=True)
    st.markdown("---")
    st.markdown("#### 🧪 أمثلة")
    if st.button("USA: AAPL MSFT NVDA"):
        st.session_state.syms = "AAPL MSFT NVDA"
    if st.button("KSA: 1120 2380 1050"):
        st.session_state.syms = "1120 2380 1050"

symbols_input = st.text_area("أدخل الرموز (مسافة/سطر). سأضيف .SR تلقائيًا عند اختيار السوق السعودي.", 
                             st.session_state.get("syms", ""))

raw = [s.strip().upper() for s in symbols_input.replace("\n"," ").split() if s.strip()]
symbols = []
for s in raw:
    if suffix and not s.endswith(suffix) and s.isalnum():
        symbols.append(s + suffix)
    else:
        symbols.append(s)
symbols = sorted(set(symbols))

if st.button("🚀 تنفيذ التحليل"):
    if not symbols:
        st.warning("يرجى إدخال رمز واحد على الأقل.")
        st.stop()

    rows = []
    buffett_rows = []
    narratives = []
    errors = []
    prog = st.progress(0.0, text="بدء التحليل...")

    for i, sym in enumerate(symbols, start=1):
        try:
            data = load_company_data(sym)
            ratios = compute_buffett_ratios(data, mode=mode)
            score, flags, verdict, extras = buffett_scorecard(ratios)

            if show_table:
                row = {
                    "الرمز": sym,
                    "الهامش الإجمالي": to_percent(ratios["GrossMargin"]),
                    "هامش التشغيل": to_percent(ratios["OperatingMargin"]),
                    "هامش صافي الربح": to_percent(ratios["NetMargin"]),
                    "ROIC": to_percent(ratios["ROIC"]),
                    "OCF/NI": to_ratio(ratios["OCF/NI"]),
                    "هامش التدفق الحر": to_percent(ratios["FCF_Margin"]),
                    "تغطية الفوائد": to_ratio(ratios["InterestCoverage"]),
                    "CCC": to_days(ratios["CCC"]),
                    "صافي الدين": to_num(extras["NetDebt"]),
                    "عائد أرباح المالك": to_percent(ratios["OwnerEarningsYield"]),
                    "P/أرباح المالك": to_ratio(ratios["P/OwnerEarnings"]),
                    "النتيجة/100": f"{score:.0f}",
                    "التوصية": verdict
                }
                rows.append(row)

                bf_row = {"الرمز": sym, "الدرجة": f"{score:.0f}/100"}
                bf_row.update(flags)
                bf_row["التوصية"] = verdict
                buffett_rows.append(bf_row)

            if show_text:
                narratives.append(buffett_narrative(sym, ratios, score, verdict))

        except Exception as e:
            errors.append(f"{sym} → {e}")

        prog.progress(i/len(symbols), text=f"تم تحليل {i}/{len(symbols)}")

    if show_table and rows:
        st.subheader(f"📋 النِّسَب الأساسية ({mode}) — {len(rows)} شركة")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # تصدير
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 تنزيل CSV", csv_bytes, file_name=f"buffett_ratios_{mode}.csv", mime="text/csv")

        st.subheader("✅ قائمة تحقق بافيت (Scoring)")
        df_b = pd.DataFrame(buffett_rows)
        # جدول HTML بسيط مع تلوين الرموز
        def html_table(df):
            html = "<table class='buffett-table'><thead><tr>"
            for c in df.columns: html += f"<th>{escape(str(c))}</th>"
            html += "</tr></thead><tbody>"
            for _, row in df.iterrows():
                html += "<tr>"
                for c in df.columns:
                    v = str(row[c])
                    cls = ""
                    if v in ("✅","⚠️","❌"):
                        cls = "green" if v=="✅" else "yellow" if v=="⚠️" else "red"
                    html += f"<td class='{cls}'>{escape(v)}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            return html
        st.markdown(html_table(df_b), unsafe_allow_html=True)
        csv_b = df_b.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 تنزيل Buffett Score CSV", csv_b, file_name=f"buffett_score_{mode}.csv", mime="text/csv")

    if show_text and narratives:
        st.subheader("🧠 التحليل النصي المُستلهم من مبادئ بافيت")
        for block in narratives:
            st.markdown(block)
            st.markdown("---")

    if errors:
        st.info("⚠️ ملاحظات:")
        for e in errors:
            st.write("• ", e)

with st.expander("📌 منهجية مختصرة"):
    st.markdown("""
- **ROIC**: NOPAT/رأس المال المستثمر ≈ EBIT×(1–الضريبة) ÷ (الدين + حقوق المساهمين – النقد).
- **أرباح المالك**: تقريبًا OCF – Capex.
- **جودة الأرباح**: OCF/NI ≥ 1 مؤشر جيد.
- **التقييم**: عائد أرباح المالك (OE/MktCap) ومضاعف السعر لأرباح المالك.
- **TTM**: نجمع 4 أرباع أخيرة لبنود الدخل والتدفق التشغيلي/الإنفاق الرأسمالي.
- **القيود**: تعتمد الأسماء على Yahoo وقد لا تتوفر كل البنود لكل الشركات؛ نُظهر "—" عند النقص.
""")
