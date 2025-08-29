# -*- coding: utf-8 -*-
"""
📊 Financial Analysis Model (Buffett Principles) — Streamlit
ملف واحد — تحليل مالي شامل: ملخص تنفيذي، نظرة عامة، قوائم، نسب، اتجاهات، مخاطر، تقييم (DCF+مقارنات)، سيناريوهات، توصيات، وملحقات.
تشغيل: streamlit run app.py
اعتماديات: streamlit, yfinance, pandas, numpy
"""

import re
from html import escape
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="📊 نموذج التحليل المالي | Buffett Principles", layout="wide")
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
    return {normalize_idx(raw): raw for raw in df.index.astype(str)}

def find_any(df: pd.DataFrame, keys: list[str], col):
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

# مرادفات شائعة لأسماء البنود (Yahoo)
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
# تحميل البيانات (قابلة للتسلسل)
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

    # معلومات أساسية اختيارية (قطاع/صناعة/اسم)
    info = {}
    try:
        # بعض نسخ yfinance: t.get_info()؛ أخرى: t.info
        data_info = {}
        try:
            data_info = t.get_info()
        except Exception:
            data_info = getattr(t, "info", {}) or {}
        if isinstance(data_info, dict):
            fields = ["longName","industry","sector","country","city","fullTimeEmployees","website","longBusinessSummary"]
            for f in fields:
                val = data_info.get(f, None)
                if isinstance(val, (str, int, float)) or val is None:
                    info[f] = val
    except Exception:
        pass

    price = np.nan
    shares = np.nan
    market_cap = np.nan
    try:
        fi = t.fast_info  # dict
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
# TTM وتجميع سنوي/ربعي
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
# نسب وقيم أساسية (معايير بافيت + نسب تقليدية)
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

    # بنود الميزانية
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

    # رأس المال المستثمر
    total_debt = find_any(bal, TOT_DEBT_KEYS, cur)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, cur) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan
    invested = np.nan if (pd.isna(total_debt) or pd.isna(te)) else total_debt + te - (0 if pd.isna(cash) else cash)

    # ضرائب تقريبية
    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = tax / pbt if (not pd.isna(pbt) and pbt != 0 and not pd.isna(tax)) else 0.25
    eff_tax = float(np.clip(eff_tax, 0.0, 0.6))
    nopat = ebit * (1 - eff_tax) if not pd.isna(ebit) else np.nan
    roic = safe_div(nopat, invested)

    # أرباح المالك + جودة الأرباح
    owner_earnings = np.nan if (pd.isna(ocf) or pd.isna(capex)) else (ocf - capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(owner_earnings, rev)

    # نسب السيولة والملاءة
    current_ratio = safe_div(ca, cl)
    quick_ratio   = safe_div((ca - (inv if not pd.isna(inv) else 0)), cl)
    debt_to_equity = safe_div(total_debt, te)
    roa = safe_div(ni, ta)
    roe = safe_div(ni, te)

    # تغطية الفوائد
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
    rec_turn = safe_div(rev, ar_avg)
    pay_turn = safe_div(cogs if not pd.isna(cogs) else rev, ap_avg)
    inv_turn = safe_div(cogs if not pd.isna(cogs) else rev, inv_avg)
    dso = safe_div(365, rec_turn)
    dpo = safe_div(365, pay_turn)
    dio = safe_div(365, inv_turn)
    ccc = dso + dio - dpo if not any(pd.isna(x) for x in [dso, dio, dpo]) else np.nan

    # تقييمات سوقية
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

    return {
        # قائمة دخل/ميزانية/تدفقات (مختصر)
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "EBIT": ebit, "NetIncome": ni,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "Cash": cash, "STInvest": sti, "TotalDebt": total_debt,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": owner_earnings,
        # الهوامش والجودة والكفاءة
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "ROA": roa, "ROE": roe, "ROIC": roic, "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "CurrentRatio": current_ratio, "QuickRatio": quick_ratio, "InterestCoverage": interest_cov,
        "AssetTurnover": asset_turn, "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        # السوق/التقييم
        "Price": price, "Shares": shares, "MarketCap": market_cap,
        "PE": pe, "PB": pb, "PS": ps, "BVPS": bvps,
        "OwnerEarningsYield": oe_yield, "P/OwnerEarnings": p_to_oe
    }

# =============================
# قائمة تحقق بافيت + الأسباب
# =============================
def buffett_scorecard(r):
    score = 0
    flags = {}
    reasons = []

    def set_flag(name, ok, mid=False):
        sym = "✅" if ok else ("⚠️" if mid else "❌")
        flags[name] = sym
        return sym

    # 1) ROIC ≥ 15%
    roic = r["ROIC"]
    ok = (not pd.isna(roic) and roic >= 0.15)
    mid = (not pd.isna(roic) and 0.10 <= roic < 0.15)
    sym = set_flag("ROIC ≥15%", ok, mid=mid)
    score += 20 if ok else (8 if mid else 0)
    reasons.append({"البند":"ROIC ≥15%","الحالة":status_word(sym),
                    "السبب": "غير متوفر" if pd.isna(roic) else f"ROIC = {to_percent(roic)} (الحد: ≥15%/مقبول ≥10%)."})

    # 2) هامش إجمالي قوي ≥25%
    gm = r["GrossMargin"]; ok=(not pd.isna(gm) and gm>=0.25); mid=(not pd.isna(gm) and 0.18<=gm<0.25)
    sym=set_flag("هامش إجمالي قوي", ok, mid); score+=10 if ok else (5 if mid else 0)
    reasons.append({"البند":"هامش إجمالي قوي","الحالة":status_word(sym),
                    "السبب": "غير متوفر" if pd.isna(gm) else f"الهامش الإجمالي = {to_percent(gm)}."})

    # 3) جودة الأرباح OCF/NI ≥1
    q=r["OCF/NI"]; ok=(not pd.isna(q) and q>=1.0); mid=(not pd.isna(q) and 0.8<=q<1.0)
    sym=set_flag("جودة الأرباح OCF/NI", ok, mid); score+=10 if ok else (5 if mid else 0)
    reasons.append({"البند":"جودة الأرباح OCF/NI","الحالة":status_word(sym),
                    "السبب":"غير متوفر" if pd.isna(q) else f"OCF/NI = {to_ratio(q)} (الحد: ≥1.0x/مقبول ≥0.8x)."})

    # 4) هامش أرباح المالك ≥8%
    f=r["FCF_Margin"]; ok=(not pd.isna(f) and f>=0.08); mid=(not pd.isna(f) and 0.05<=f<0.08)
    sym=set_flag("هامش التدفق الحر", ok, mid); score+=10 if ok else (5 if mid else 0)
    reasons.append({"البند":"هامش التدفق الحر","الحالة":status_word(sym),
                    "السبب":"غير متوفر" if pd.isna(f) else f"هامش OE = {to_percent(f)} (الحد: ≥8%/مقبول ≥5%)."})

    # 5) هيكل دين متحفظ: صافي الدين ≤0 أو Debt/OE ≤2
    td, cash = r["TotalDebt"], r["Cash"]; oe = r["OwnerEarnings"]
    net_debt = np.nan if pd.isna(td) else td - (0 if pd.isna(cash) else cash)
    ratio_debt_oe = (td/oe) if (not any(pd.isna(x) for x in [td, oe]) and oe>0) else np.nan
    crit = (not pd.isna(net_debt) and net_debt<=0) or (not pd.isna(ratio_debt_oe) and ratio_debt_oe<=2.0)
    mid  = (not pd.isna(ratio_debt_oe) and ratio_debt_oe<=3.0)
    sym=set_flag("هيكل دين متحفظ", crit, mid); score+=10 if crit else (5 if mid else 0)
    reason = "بيانات غير متوفرة"
    if not (pd.isna(td) and pd.isna(cash)):
        parts=[f"صافي الدين: {to_num(net_debt)}"]
        if not pd.isna(ratio_debt_oe): parts.append(f"الدين/أرباح المالك: {to_ratio(ratio_debt_oe)}")
        reason="، ".join(parts)+"؛ الحد ≤0 أو ≤2.0x (مقبول ≤3.0x)."
    reasons.append({"البند":"هيكل دين متحفظ","الحالة":status_word(sym),"السبب":reason})

    # 6) تغطية الفوائد ≥10x
    ic=r["InterestCoverage"]; ok=(not pd.isna(ic) and ic>=10.0); mid=(not pd.isna(ic) and 6.0<=ic<10.0)
    sym=set_flag("تغطية الفوائد", ok, mid); score+=10 if ok else (5 if mid else 0)
    reasons.append({"البند":"تغطية الفوائد","الحالة":status_word(sym),
                    "السبب":"غير متوفر" if pd.isna(ic) else f"التغطية = {to_ratio(ic)} (الحد: ≥10x/مقبول ≥6x)."})

    # 7) CCC ≤ 0 يوم (أو ≤ 30 يوم مقبول)
    ccc=r["CCC"]; ok=(not pd.isna(ccc) and ccc<=0); mid=(not pd.isna(ccc) and ccc<=30)
    sym=set_flag("دورة التحويل النقدي", ok, mid); score+=5 if ok else (2 if mid else 0)
    reasons.append({"البند":"دورة التحويل النقدي","الحالة":status_word(sym),
                    "السبب":"غير متوفر" if pd.isna(ccc) else f"CCC = {to_days(ccc)} (الحد: ≤0/مقبول ≤30)."})

    # 8) تقييم معقول: OE Yield ≥6% أو P/OE ≤20
    oey=r["OwnerEarningsYield"]; pto=r["P/OwnerEarnings"]
    ok = (not pd.isna(oey) and oey>=0.06) or (not pd.isna(pto) and pto<=20)
    mid= (not pd.isna(oey) and oey>=0.04) or (not pd.isna(pto) and pto<=25)
    sym=set_flag("تقييم معقول (OE Yield / P-to-OE)", ok, mid); score+=10 if ok else (5 if mid else 0)
    cond=[]
    if not pd.isna(oey): cond.append(f"OE Yield = {to_percent(oey)}")
    if not pd.isna(pto): cond.append(f"P/OE = {to_ratio(pto)}")
    reasons.append({"البند":"تقييم معقول (OE Yield / P-to-OE)","الحالة":status_word(sym),
                    "السبب":("، ".join(cond) if cond else "بيانات غير متوفرة")+ "؛ الحد ≥6% أو ≤20x (مقبول ≥4% أو ≤25x)."})

    verdict = "✅ جذّابة مع هامش أمان" if score >= 75 else ("🟧 جيدة لكن انتظر سعرًا أفضل" if score >= 55 else "🕒 راقِب")
    return float(score), flags, verdict, net_debt, reasons

# =============================
# اتجاهات تاريخية (3–5 سنوات)
# =============================
def historical_trends(inc_a: pd.DataFrame, cf_a: pd.DataFrame, years: int = 5):
    def take_series(df, keys):
        if df is None or df.empty: return pd.Series(dtype=float)
        cols = sorted_cols(df)
        cols = cols[:years] if years>0 else cols
        data = {str(c): find_any(df, keys, c) for c in cols}
        return pd.Series(data)

    rev_s = take_series(inc_a, REV_KEYS)
    gp_s  = take_series(inc_a, GP_KEYS)
    ni_s  = take_series(inc_a, NI_KEYS)
    ocf_s = take_series(cf_a, OCF_KEYS)
    cap_s = take_series(cf_a, CAPEX_KEYS)

    df = pd.DataFrame({
        "Revenue": rev_s, "GrossProfit": gp_s, "NetIncome": ni_s,
        "OCF": ocf_s, "Capex": cap_s
    })
    # نمو سنوي مبسّط
    def yoy(s):
        vals = s.values
        out = [np.nan]
        for i in range(1, len(vals)):
            out.append(safe_div(vals[i]-vals[i-1], abs(vals[i-1]) if not pd.isna(vals[i-1]) and vals[i-1]!=0 else np.nan))
        return pd.Series(out, index=s.index)
    df_growth = df.apply(yoy)
    return df, df_growth

# =============================
# تقييم مبسّط: DCF على أرباح المالك
# =============================
def simple_dcf(oe_base, discount_rate=0.12, growth_rate=0.05, years=5, terminal_growth=0.02):
    if pd.isna(oe_base) or oe_base<=0 or discount_rate<=terminal_growth:
        return np.nan, pd.DataFrame()
    flows = []
    pv = 0.0
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
# نصوص: ملخص تنفيذي + نظرة عامة + توصيات
# =============================
def executive_summary(sym, info, r, score, verdict, dcf_value_ps, price):
    bullets = []
    sector = info.get("sector") or "—"
    industry = info.get("industry") or "—"
    bullets.append(f"**الشركة/الرمز:** {info.get('longName') or sym} ({sym}) — القطاع: {sector} | الصناعة: {industry}")
    bullets.append(f"- **أبرز النتائج:** هامش إجمالي {to_percent(r['GrossMargin'])}، ROIC {to_percent(r['ROIC'])}، جودة أرباح OCF/NI {to_ratio(r['OCF/NI'])}، CCC {to_days(r['CCC'])}.")
    bullets.append(f"- **السيولة والملاءة:** Current {to_ratio(r['CurrentRatio'])}، Quick {to_ratio(r['QuickRatio'])}، D/E {to_ratio(safe_div(r['TotalDebt'], r['TotalEquity'] if 'TotalEquity' in r else np.nan))}.")
    bullets.append(f"- **القوة المالية:** صافي الدين {to_num(r['TotalDebt'] - (0 if pd.isna(r['Cash']) else r['Cash']) if not pd.isna(r['TotalDebt']) else np.nan)}، تغطية الفوائد {to_ratio(r['InterestCoverage'])}.")
    if not pd.isna(dcf_value_ps) and not pd.isna(price):
        disc = ((dcf_value_ps/price)-1) if (price>0) else np.nan
        bullets.append(f"- **التقييم (DCF مبسّط):** القيمة الجوهرية/سهم ≈ {to_num(dcf_value_ps)} مقابل السعر {to_num(price)} → هامش أمان {to_percent(disc) if not pd.isna(disc) else '—'}.")
    bullets.append(f"**الخلاصة:** درجة بافيت {score:.0f}/100 — {verdict}.")
    return "\n".join(bullets)

def company_overview(info):
    lines = []
    nm = info.get("longName") or "—"
    lines.append(f"**الاسم:** {nm}")
    lines.append(f"**القطاع/الصناعة:** {info.get('sector') or '—'} / {info.get('industry') or '—'}")
    lines.append(f"**الموظفون:** {info.get('fullTimeEmployees') or '—'} | **الموقع:** {info.get('city') or '—'}, {info.get('country') or '—'}")
    if info.get("website"):
        lines.append(f"**الموقع:** {info.get('website')}")
    if info.get("longBusinessSummary"):
        lines.append(f"**وصف مختصر:** {info.get('longBusinessSummary')[:800]}{'…' if len(info.get('longBusinessSummary'))>800 else ''}")
    return "\n".join(lines)

def auto_swot(r, growth_df):
    strengths, weaknesses, opportunities, threats = [], [], [], []
    if not pd.isna(r["ROIC"]) and r["ROIC"]>=0.15: strengths.append("ROIC مرتفع يدل على كفاءة استخدام رأس المال.")
    if not pd.isna(r["GrossMargin"]) and r["GrossMargin"]>=0.25: strengths.append("هوامش إجمالية قوية وخندق تنافسي محتمل.")
    if not pd.isna(r["OCF/NI"]) and r["OCF/NI"]>=1.0: strengths.append("جودة أرباح جيدة (النقد يدعم الربح المحاسبي).")
    if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]>0: strengths.append("توليد أرباح مالك إيجابية.")
    if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0: weaknesses.append("سيولة جارية ضعيفة (<1.0).")
    if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0: weaknesses.append("تغطية الفوائد منخفضة قد تزيد المخاطر.")
    if not pd.isna(r["CCC"]) and r["CCC"]>30: weaknesses.append("دورة تحويل نقدي طويلة.")
    # فرص/تهديدات بالاعتماد على النمو التاريخي
    try:
        rev_growth = growth_df["Revenue"].dropna()
        if len(rev_growth)>=2 and np.nanmean(rev_growth.tail(3))>0:
            opportunities.append("اتجاه نمو إيرادات إيجابي يمكن البناء عليه.")
        elif len(rev_growth)>=2 and np.nanmean(rev_growth.tail(3))<0:
            threats.append("تباطؤ/انكماش في الإيرادات مؤخراً.")
    except Exception:
        pass
    return strengths, weaknesses, opportunities, threats

# =============================
# واجهة المستخدم
# =============================
st.title("📊 نموذج التحليل المالي (مستلهَم من مبادئ بافيت)")

with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1)
    st.markdown("---")
    st.markdown("#### إعدادات DCF (افتراضيّة)")
    disc_rate = st.number_input("معدل الخصم (r)", 0.05, 0.30, 0.12, 0.01)
    growth_rate = st.number_input("نمو السنوات (g)", 0.00, 0.30, 0.05, 0.01)
    years = st.number_input("عدد السنوات", 3, 10, 5, 1)
    term_growth = st.number_input("نمو نهائي (gₜ)", 0.00, 0.05, 0.02, 0.005)
    st.markdown("---")
    comps_input = st.text_input("مقارنات (رموز مفصولة بمسافة/سطر)", "")
    st.markdown("---")
    st.markdown("#### أمثلة سريعة")
    if st.button("USA: AAPL"):
        st.session_state.syms = "AAPL"
    if st.button("KSA: 1120"):
        st.session_state.syms = "1120"

symbols_input = st.text_area("أدخل رمزًا واحدًا للتحليل (سأحلّل أول رمز فقط إذا أدخلت أكثر).",
                             st.session_state.get("syms",""))

# تجهيز الرمز
raw = [s.strip().upper() for s in symbols_input.replace("\n"," ").split() if s.strip()]
symbols = []
for s in raw:
    if suffix and not s.endswith(suffix) and s.isalnum():
        symbols.append(s + suffix)
    else:
        symbols.append(s)
symbols = [s for i,s in enumerate(symbols) if i==0]  # تحليل أول رمز فقط

if st.button("🚀 تنفيذ التحليل"):
    if not symbols:
        st.warning("يرجى إدخال رمز واحد.")
        st.stop()

    sym = symbols[0]
    data = load_company_data(sym)
    r = compute_core_metrics(data, mode)
    score, flags, verdict, net_debt, reasons = buffett_scorecard(r)
    hist_df, growth_df = historical_trends(data["inc_a"], data["cf_a"], years=5)

    # DCF
    dcf_total, dcf_table = simple_dcf(r["OwnerEarnings"], disc_rate, growth_rate, int(years), term_growth)
    dcf_per_share = (dcf_total / r["Shares"]) if (not pd.isna(dcf_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan

    # ملخص تنفيذي + نظرة عامة + SWOT
    summary_text = executive_summary(sym, data.get("info", {}), r, score, verdict, dcf_per_share, r["Price"])
    overview_text = company_overview(data.get("info", {}))
    s, w, o, t = auto_swot(r, growth_df)

    # تبويب هيكل النموذج
    tabs = st.tabs([
        "1) ملخص تنفيذي", "2) نظرة عامة على الشركة", "3) تحليل البيانات المالية",
        "4) النسب المالية", "5) تحليل الاتجاهات", "6) تقييم المخاطر",
        "7) التقييم (Valuation)", "8) التوقعات والسيناريوهات", "9) التوصيات والخلاصة",
        "10) الملاحق والمصادر"
    ])

    # 1) ملخص تنفيذي
    with tabs[0]:
        st.markdown(summary_text)

    # 2) نظرة عامة
    with tabs[1]:
        st.markdown(overview_text)
        st.markdown("**تحليل SWOT (تلقائي مبسّط):**")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**نقاط القوة:**")
            st.write("- " + "\n- ".join(s) if s else "—")
            st.markdown("**الفرص:**")
            st.write("- " + "\n- ".join(o) if o else "—")
        with c2:
            st.markdown("**نقاط الضعف:**")
            st.write("- " + "\n- ".join(w) if w else "—")
            st.markdown("**التهديدات:**")
            st.write("- " + "\n- ".join(t) if t else "—")

    # 3) تحليل البيانات المالية (BS/IS/CF)
    with tabs[2]:
        st.markdown("### الميزانية العمومية (مختصر)")
        bs_rows = [{
            "الأصول المتداولة": to_num(r["CurrentAssets"]),
            "الأصول غير المتداولة ~": "—",
            "الخصوم المتداولة": to_num(r["CurrentLiabilities"]),
            "الخصوم طويلة الأجل ~": "—",
            "حقوق الملكية": to_num(r["TotalEquity"]),
            "النقد وما في حكمه": to_num(r["Cash"]),
            "الاستثمارات القصيرة": to_num(r["STInvest"]),
            "إجمالي الأصول": to_num(r["TotalAssets"]),
            "إجمالي الدين": to_num(r["TotalDebt"]),
        }]
        st.dataframe(pd.DataFrame(bs_rows))

        st.markdown("### قائمة الدخل (مختصر)")
        is_rows = [{
            "الإيرادات": to_num(r["Revenue"]),
            "الربح الإجمالي": to_num(r["GrossProfit"]),
            "EBIT": to_num(r["EBIT"]),
            "صافي الربح": to_num(r["NetIncome"]),
            "الهامش الإجمالي": to_percent(r["GrossMargin"]),
            "هامش التشغيل": to_percent(r["OperatingMargin"]),
            "هامش صافي الربح": to_percent(r["NetMargin"])
        }]
        st.dataframe(pd.DataFrame(is_rows))

        st.markdown("### قائمة التدفقات النقدية (مختصر)")
        cf_rows = [{
            "تشغيلي OCF": to_num(r["OCF"]),
            "استثماري (Capex)": to_num(r["Capex"]),
            "أرباح المالك (OE=OCF-Capex)": to_num(r["OwnerEarnings"])
        }]
        st.dataframe(pd.DataFrame(cf_rows))

        st.caption("(* ~ = تبسيطات عند نقص التفصيل في Yahoo Finance *)")

    # 4) النسب المالية (حسب طلبك)
    with tabs[3]:
        ratios_tbl = [{
            "الربحية: Gross": to_percent(r["GrossMargin"]),
            "الربحية: Net": to_percent(r["NetMargin"]),
            "ROA": to_percent(r["ROA"]),
            "ROE": to_percent(r["ROE"]),
            "السيولة: Current": to_ratio(r["CurrentRatio"]),
            "السيولة: Quick": to_ratio(r["QuickRatio"]),
            "المديونية: D/E": to_ratio(safe_div(r["TotalDebt"], r["TotalEquity"] if "TotalEquity" in r else np.nan)),
            "تغطية الفوائد": to_ratio(r["InterestCoverage"]),
            "الكفاءة: دوران الأصول": to_ratio(r["AssetTurnover"]),
            "الكفاءة: DSO": to_days(r["DSO"]),
            "الكفاءة: DIO": to_days(r["DIO"]),
            "الكفاءة: DPO": to_days(r["DPO"]),
            "السوق: P/E": "—" if pd.isna(r["PE"]) else f"{r['PE']:.2f}x",
            "السوق: P/B": "—" if pd.isna(r["PB"]) else f"{r['PB']:.2f}x",
            "BVPS": to_num(r["BVPS"]),
        }]
        st.dataframe(pd.DataFrame(ratios_tbl))

        st.subheader("✅ قائمة تحقق بافيت + الأسباب")
        df_flags = pd.DataFrame([{"البند":k, "التقييم":v} for k,v in flags.items()])
        st.dataframe(df_flags, use_container_width=True)
        st.markdown("**الأسباب التفصيلية لكل بند:**")
        st.dataframe(pd.DataFrame(reasons), use_container_width=True)

    # 5) تحليل الاتجاهات
    with tabs[4]:
        st.markdown("### السلاسل التاريخية (آخر 3–5 سنوات حسب التوفر)")
        st.dataframe(hist_df.T, use_container_width=True)
        st.markdown("### نمو سنوي (YoY) مبسّط")
        st.dataframe((growth_df*100).round(2).T, use_container_width=True)
        st.caption("القيم %: موجبة = نمو، سالبة = تراجع.")

    # 6) تقييم المخاطر
    with tabs[5]:
        risks = []
        if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0:
            risks.append("سيولة جارية دون 1.0 قد تضغط على السداد القصير.")
        if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0:
            risks.append("تغطية فوائد منخفضة تُعلي حساسية السعر لارتفاع الفائدة/انخفاض الأرباح.")
        if not pd.isna(r["CCC"]) and r["CCC"]>30:
            risks.append("سلسلة تحويل النقد بطيئة نسبيًا مقارنة بالأداء المرغوب.")
        if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]<=0:
            risks.append("تدفقات حرة ضعيفة/سلبية قد تحد من المرونة الاستثمارية.")
        st.markdown("**المخاطر المالية/التشغيلية:**")
        st.write("- " + "\n- ".join(risks) if risks else "—")
        st.caption("**مخاطر خارجية (عامّة):** دورات اقتصادية، تغيّر لوائح، ومنافسة سعرية/تقنية.")

    # 7) التقييم
    with tabs[6]:
        st.markdown("### تقييم DCF مبسّط (على أرباح المالك)")
        if not pd.isna(dcf_total):
            st.dataframe(dcf_table)
            st.write("**القيمة الحالية الإجمالية (للشركة):**", to_num(dcf_total))
            if not pd.isna(dcf_per_share):
                st.write("**القيمة الجوهرية لكل سهم:**", to_num(dcf_per_share))
        else:
            st.info("لا يمكن حساب DCF لغياب OE أو عدم اتساق الافتراضات (تحقق من r > gₜ و OE>0).")

        st.markdown("---")
        st.markdown("### مقارنة شركات (اختياري)")
        comps_raw = [c.strip().upper() for c in comps_input.replace("\n"," ").split() if c.strip()]
        if comps_raw:
            comp_rows=[]
            for c in comps_raw[:8]:
                try:
                    d = load_company_data(c if (suffix=="" or c.endswith(".SR")) else c+suffix)
                    rr = compute_core_metrics(d, mode)
                    comp_rows.append({
                        "الرمز": c if (suffix=="" or c.endswith(".SR")) else c+suffix,
                        "P/E": "—" if pd.isna(rr["PE"]) else f"{rr['PE']:.2f}",
                        "P/B": "—" if pd.isna(rr["PB"]) else f"{rr['PB']:.2f}",
                        "ROE": to_percent(rr["ROE"]),
                        "ROIC": to_percent(rr["ROIC"]),
                        "هامش صافي": to_percent(rr["NetMargin"])
                    })
                except Exception as e:
                    comp_rows.append({"الرمز": c, "P/E":"—","P/B":"—","ROE":"—","ROIC":"—","هامش صافي":f"خطأ: {e}"})
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
        else:
            st.caption("أدخل رموزًا في الشريط الجانبي لعرض مقارنة مبسطة.")

    # 8) التوقعات والسيناريوهات
    with tabs[7]:
        st.markdown("**سيناريو قاعدة:** يستخدم مدخلات الشريط الجانبي كما هي.")
        st.write(f"نمو = {growth_rate*100:.1f}٪ | خصم = {disc_rate*100:.1f}٪ | نمو نهائي = {term_growth*100:.1f}٪ | سنوات = {int(years)}")
        st.write("القيمة/سهم (قاعدة):", to_num(dcf_per_share))
        # سيناريو أفضل/أسوأ (±2% نمو و ±2% خصم)
        best_total,_ = simple_dcf(r["OwnerEarnings"], disc_rate-0.02, growth_rate+0.02, int(years), term_growth)
        worst_total,_= simple_dcf(r["OwnerEarnings"], disc_rate+0.02, growth_rate-0.02, int(years), term_growth)
        best_ps = (best_total/r["Shares"]) if (not pd.isna(best_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan
        worst_ps= (worst_total/r["Shares"]) if (not pd.isna(worst_total) and not pd.isna(r["Shares"]) and r["Shares"]>0) else np.nan
        st.write("أفضل حالة (Higher g / Lower r):", to_num(best_ps))
        st.write("أسوأ حالة (Lower g / Higher r):", to_num(worst_ps))

    # 9) التوصيات والخلاصة
    with tabs[8]:
        recs=[]
        if not pd.isna(r["OwnerEarnings"]) and r["OwnerEarnings"]>0 and (score>=75):
            recs.append("السهم جذّاب وفق مبادئ القيمة مع هامش أمان معقول.")
        if not pd.isna(r["CurrentRatio"]) and r["CurrentRatio"]<1.0:
            recs.append("تعزيز السيولة (زيادة النقد/خفض الالتزامات الجارية).")
        if not pd.isna(r["InterestCoverage"]) and r["InterestCoverage"]<6.0:
            recs.append("تقليل الدين أو تحسين الربحية لرفع تغطية الفوائد.")
        if not pd.isna(r["CCC"]) and r["CCC"]>30:
            recs.append("تحسين إدارة رأس المال العامل (تحصيل أسرع، مخزون أخف، تفاوض أجل الدفع).")
        if not recs:
            recs.append("لا توجد توصيات تشغيلية مُلحة استنادًا للبيانات المتاحة.")
        st.markdown("**توصيات عملية للشركة/الإدارة:**")
        st.write("- " + "\n- ".join(recs))
        st.markdown("**توصية للمستثمر:** " + verdict)

    # 10) الملاحق والمصادر
    with tabs[9]:
        st.markdown("**البيانات المالية الخام (مقتطفات من Yahoo Finance):**")
        st.markdown("**Income (Annual):**")
        st.dataframe(data["inc_a"], use_container_width=True)
        st.markdown("**Balance Sheet (Annual):**")
        st.dataframe(data["bal_a"], use_container_width=True)
        st.markdown("**Cash Flow (Annual):**")
        st.dataframe(data["cf_a"], use_container_width=True)
        st.caption("المصدر: Yahoo Finance عبر yfinance | الافتراضات موضّحة في إعدادات DCF.")

