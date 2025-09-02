مع اكسل بس اكسل يبغاله تعديل 
# -*- coding: utf-8 -*-
"""
📊 Financial Analysis Model (Buffett Principles) — v3 (Detailed Report + Excel VBA 2x2)
ملف واحد — تحليل مالي شامل + تقرير Markdown مفصل + مُصدِّر كود VBA بسيط (جدول 2×2).
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
# تهيئة الصفحة + RTL + ثيم بصري
# =============================
st.set_page_config(page_title="📊 نموذج التحليل المالي | Buffett Principles", layout="wide")
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
    rec_turn = safe_div(rev, ar_avg)
    pay_turn = safe_div(cogs if not pd.isna(cogs) else rev, ap_avg)
    inv_turn = safe_div(cogs if not pd.isna(cogs) else rev, inv_avg)
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
        "income_period": (str(sorted_cols(inc_q)[0]) if (mode=="TTM" and not inc_q.empty) else (str(sorted_cols(inc_a)[0]) if not inc_a.empty else "—")),
        "balance_period": (str(sorted_cols(bal_q)[0]) if (mode=="TTM" and not bal_q.empty) else (str(sorted_cols(bal_a)[0]) if not bal_a.empty else "—")),
        "cashflow_period": (str(sorted_cols(cf_q)[0]) if (mode=="TTM" and not cf_q.empty) else (str(sorted_cols(cf_a)[0]) if not cf_a.empty else "—")),
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
    components = []  # تفصيل النقاط

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
                    dcf_table, base_ps, best_ps, worst_ps, comps_rows, data):
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

    # 11) ملاحق
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
"""

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
        "",
        "_المصدر: Yahoo Finance عبر yfinance. هذا التقرير لأغراض تعليمية/بحثية وليس توصية استثمارية._"
    ]
    return "\n".join(sections)

# =============================
# مُصدِّر Excel VBA Module — جدول 2×2 بسيط (نسخة إنجليزية مصحّحة)
# =============================
def build_company_summary_vba_module() -> str:
    # VBA module: creates "Company Summary" sheet and a 2x2 formatted table (English-only labels)
    return """Option Explicit

'=========================
'  Minimal Company Sheet: 2x2 Table Only (English-only labels)
'=========================
Sub RunCompanySummary()
    On Error GoTo EH
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    Dim ws As Worksheet
    Set ws = CreateOrClearSheet("Company Summary")

    With ws
        .Cells.Clear

        ' 2x2 header + sample row (English only)
        .Range("A1").Value = "Field"
        .Range("B1").Value = "Value"
        .Range("A2").Value = "Example"
        .Range("B2").Value = "123"

        ' Remove existing table (if any)
        On Error Resume Next
        .ListObjects("SummaryTable").Unlist
        If Err.Number <> 0 Then
            Err.Clear
        End If
        On Error GoTo 0

        ' Create a new ListObject table
        Dim lo As ListObject
        Set lo = .ListObjects.Add(SourceType:=xlSrcRange, _
                                  Source:=.Range("A1:B2"), _
                                  XlListObjectHasHeaders:=xlYes)
        lo.Name = "SummaryTable"

        ' Try to apply an English table style; fallback if not available
        On Error Resume Next
        lo.TableStyle = "TableStyleMedium2"
        If Err.Number <> 0 Then
            Err.Clear
            lo.TableStyle = "TableStyleMedium1"
        End If
        On Error GoTo 0

        .Columns("A:B").AutoFit
    End With

    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    MsgBox "A 2x2 table was created on 'Company Summary'.", vbInformation, "Done"
    Exit Sub

EH:
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    MsgBox "Error: " & Err.Number & " - " & Err.Description, vbCritical, "RunCompanySummary"
End Sub

'=========================
' Helper
'=========================
Private Function CreateOrClearSheet(ByVal sheetName As String) As Worksheet
    On Error Resume Next
    Set CreateOrClearSheet = ThisWorkbook.Worksheets(sheetName)
    If CreateOrClearSheet Is Nothing Then
        Set CreateOrClearSheet = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        CreateOrClearSheet.Name = sheetName
    Else
        CreateOrClearSheet.Cells.Clear
    End If
    On Error GoTo 0
End Function
"""

# =============================
# واجهة المستخدم
# =============================
st.markdown("<div class='hero'><h1>📊 نموذج التحليل المالي (مستلهَم من مبادئ بافيت)</h1><div class='muted'>واجهة محسّنة + تقرير Markdown مفصل + مُصدِّر كود Excel VBA (جدول 2×2 بسيط)</div></div>", unsafe_allow_html=True)

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

    st.write("**درجة بافيت:** ", f"{score:.0f}/100 — {verdict}")
    st.progress(min(max(int(score), 0), 100)/100)

    # ======== تبويبات ========
    tabs = st.tabs([
        "1) ملخص تنفيذي", "2) نظرة عامة", "3) القوائم (BS/IS/CF)",
        "4) النسب", "5) الاتجاهات", "6) المخاطر", "7) التقييم", "8) مقارنات", "9) الأسباب/التحقق", "10) تقرير للتنزيل", "11) تصدير Excel VBA"
    ])

    with tabs[0]:
        st.markdown(executive_summary(sym, data.get("info", {}), r, score, verdict, dcf_per_share, r["Price"]))
        if simple_mode: st.caption("تم تبسيط العرض. عطّل 'وضع مبسّط' لعرض مزيد من التفاصيل.")

    with tabs[1]:
        st.markdown(company_overview(data.get("info", {})))

    with tabs[2]:
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
        ratios_tbl_df = [{
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
        st.dataframe(pd.DataFrame(ratios_tbl_df), use_container_width=True)

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
        comps_raw = [c.strip().upper() for c in comps_input.replace("\n"," ").split() if c.strip()]
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
        report_md = build_report_md(
            sym, data.get("info", {}), r, score, verdict, dcf_per_share, r["Price"], reasons, components,
            mode, trend_df, dcf_table, dcf_per_share, best_ps, worst_ps, comps_rows, data
        )
        st.download_button("📥 تنزيل التقرير المفصل (Markdown)", report_md.encode("utf-8"),
                           file_name=f"Detailed_Financial_Report_{sym}.md", mime="text/markdown")
        st.caption("يشمل: ملخص تنفيذي، نظرة عامة، تحليل القوائم، نسب مشروحة، بافيت (نقاط/مبررات)، اتجاهات، تقييم DCF، حساسية، مخاطر، توصيات، وملاحق.")

    with tabs[10]:
        st.markdown("### ⬇️ تصدير كود Excel VBA — **RunCompanySummary** (ينشئ جدول 2×2 بسيط)")
        vba_code = build_company_summary_vba_module()
        st.text_area("📄 معاينة الكود", vba_code, height=280)
        st.download_button(
            "⬇️ تنزيل ملف VBA (.bas)",
            data=vba_code.encode("utf-8"),
            file_name=f"CompanySummary_RunCompanySummary_{sym if sym else 'TICKER'}.bas",
            mime="text/plain"
        )
        st.markdown("""
**طريقة الاستخدام في Excel (النسخة البسيطة):**
1) افتح Excel → **Developer** → **Visual Basic**.  
2) من **File** داخل محرر VBA اختر **Import File…** واستورد الملف `.bas`.  
3) شغِّل الإجراء **RunCompanySummary** وسيُنشئ ورقة **Company Summary** بجدول 2×2 فقط.
""")

# دليل مبسّط
with st.expander("ℹ️ ماذا تعني المؤشرات؟"):
    st.markdown("""
- **ROIC**: كفاءة تحويل رأس المال إلى أرباح تشغيلية بعد الضرائب.
- **OCF/NI**: جودة الأرباح؛ ≥1.0 يعني أن النقد يدعم الربح المحاسبي.
- **CCC**: زمن دورة النقد؛ أقل أفضل.
- **OE Yield**: أرباح المالك/القيمة السوقية؛ مقياس لعائد ضمني.
- **DCF مبسّط**: تقدير أولي للقيمة الجوهرية اعتمادًا على OE وافتراضات محافظة.
""")

