# -*- coding: utf-8 -*-
"""
📊 Buffett Principles — Streamlit (ملف واحد بسيط)
- نسب أساسية مستوحاة من مبادئ وارن بافيت
- تحليل نصي + أسباب قائمة التحقق لكل بند
تشغيل: streamlit run app.py
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
        "price": price, "shares": shares, "market_cap": market_cap
    }

# =============================
# حساب TTM بسيط
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
# نسب بافيت
# =============================
def compute_buffett_ratios(data: dict, mode: str):
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

    # رأس المال المستثمر
    bal_cols = sorted_cols(bal)
    cur = bal_cols[0] if bal_cols else None
    total_debt = find_any(bal, TOT_DEBT_KEYS, cur)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, cur) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan
    te   = find_any(bal, TE_KEYS, cur)
    cash = find_any(bal, CASH_KEYS, cur)
    invested = np.nan if (pd.isna(total_debt) or pd.isna(te)) else total_debt + te - (0 if pd.isna(cash) else cash)

    # NOPAT والضريبة الفعّالة
    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = tax / pbt if (not pd.isna(pbt) and pbt != 0 and not pd.isna(tax)) else 0.25
    eff_tax = float(np.clip(eff_tax, 0.0, 0.6))
    nopat = ebit * (1 - eff_tax) if not pd.isna(ebit) else np.nan
    roic = safe_div(nopat, invested)

    # جودة الأرباح والتدفق
    owner_earnings = np.nan if (pd.isna(ocf) or pd.isna(capex)) else (ocf - capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(owner_earnings, rev)

    # تغطية الفوائد
    int_exp = find_any(inc_used, INT_EXP_KEYS, col_income)
    if not pd.isna(int_exp): int_exp = abs(int_exp)
    interest_cov = safe_div(ebit, int_exp)

    # CCC
    prev = bal_cols[1] if len(bal_cols) > 1 else None
    ar = find_any(bal, AR_KEYS, cur);  ar_prev = find_any(bal, AR_KEYS, prev)
    ap = find_any(bal, AP_KEYS, cur);  ap_prev = find_any(bal, AP_KEYS, prev)
    inv= find_any(bal, INV_KEYS, cur); inv_prev= find_any(bal, INV_KEYS, prev)
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

    # تقييم مبسّط
    market_cap = data.get("market_cap", np.nan)
    if (pd.isna(market_cap) or market_cap == 0) and (not pd.isna(data.get("price")) and not pd.isna(data.get("shares"))):
        market_cap = data["price"] * data["shares"]
    oe_yield = safe_div(owner_earnings, market_cap)
    p_to_oe  = safe_div(market_cap, owner_earnings)

    return {
        "Revenue": rev, "COGS": cogs, "EBIT": ebit, "NetIncome": ni,
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "NOPAT": nopat, "InvestedCapital": invested, "ROIC": roic,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": owner_earnings,
        "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "TotalDebt": total_debt, "Cash": cash, "InterestCoverage": interest_cov,
        "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        "MarketCap": market_cap, "OwnerEarningsYield": oe_yield, "P/OwnerEarnings": p_to_oe
    }

# =============================
# Score + أسباب مفصلة (مجموع 100)
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
    if pd.isna(roic):
        reason = "ROIC غير متوفر لقصور البيانات."
    else:
        thr = "≥ 15%" if ok else ("بين 10% و15%" if mid else "< 10%")
        reason = f"ROIC = {to_percent(roic)}، والحد {thr}."
    reasons.append({"البند": "ROIC ≥15%", "الحالة": status_word(sym), "السبب": reason})

    # 2) هامش إجمالي قوي (≥ 25%)
    gm = r["GrossMargin"]
    ok = (not pd.isna(gm) and gm >= 0.25)
    mid = (not pd.isna(gm) and 0.18 <= gm < 0.25)
    sym = set_flag("هامش إجمالي قوي", ok, mid=mid)
    score += 10 if ok else (5 if mid else 0)
    if pd.isna(gm):
        reason = "الهامش الإجمالي غير متوفر."
    else:
        thr = "≥ 25%" if ok else ("بين 18% و25%" if mid else "< 18%")
        reason = f"الهامش الإجمالي = {to_percent(gm)}، والحد {thr}."
    reasons.append({"البند": "هامش إجمالي قوي", "الحالة": status_word(sym), "السبب": reason})

    # 3) جودة الأرباح OCF/NI ≥ 1
    q = r["OCF/NI"]
    ok = (not pd.isna(q) and q >= 1.0)
    mid = (not pd.isna(q) and 0.8 <= q < 1.0)
    sym = set_flag("جودة الأرباح OCF/NI", ok, mid=mid)
    score += 10 if ok else (5 if mid else 0)
    if pd.isna(q):
        reason = "نسبة OCF/NI غير متوفرة."
    else:
        thr = "≥ 1.0x" if ok else ("بين 0.8x و1.0x" if mid else "< 0.8x")
        reason = f"OCF/NI = {to_ratio(q)}، والحد {thr}."
    reasons.append({"البند": "جودة الأرباح OCF/NI", "الحالة": status_word(sym), "السبب": reason})

    # 4) هامش التدفق الحر (أرباح المالك/الإيراد) ≥ 8%
    f = r["FCF_Margin"]
    ok = (not pd.isna(f) and f >= 0.08)
    mid = (not pd.isna(f) and 0.05 <= f < 0.08)
    sym = set_flag("هامش التدفق الحر", ok, mid=mid)
    score += 10 if ok else (5 if mid else 0)
    if pd.isna(f):
        reason = "هامش أرباح المالك غير متوفر."
    else:
        thr = "≥ 8%" if ok else ("بين 5% و8%" if mid else "< 5%")
        reason = f"هامش أرباح المالك = {to_percent(f)}، والحد {thr}."
    reasons.append({"البند": "هامش التدفق الحر", "الحالة": status_word(sym), "السبب": reason})

    # 5) هيكل دين متحفظ: صافي الدين ≤ 0 أو Debt/OE ≤ 2
    td, cash = r["TotalDebt"], r["Cash"]
    oe = r["OwnerEarnings"]
    net_debt = np.nan if pd.isna(td) else td - (0 if pd.isna(cash) else cash)
    ratio_debt_oe = (td / oe) if (not any(pd.isna(x) for x in [td, oe]) and oe > 0) else np.nan
    crit = (not pd.isna(net_debt) and net_debt <= 0) or (not pd.isna(ratio_debt_oe) and ratio_debt_oe <= 2.0)
    mid  = (not pd.isna(ratio_debt_oe) and ratio_debt_oe <= 3.0)
    sym = set_flag("هيكل دين متحفظ", crit, mid=mid)
    score += 10 if crit else (5 if mid else 0)
    if pd.isna(td) and pd.isna(cash):
        reason = "بيانات الدين/النقد غير متوفرة."
    else:
        nd_txt = f"صافي الدين = {to_num(net_debt)}"
        if not pd.isna(ratio_debt_oe):
            nd_txt += f"، ونسبة الدين/أرباح المالك = {to_ratio(ratio_debt_oe)}"
        thr = "≤ 0 أو ≤ 2.0x" if crit else ("≤ 3.0x (مقبول)" if mid else "> 3.0x أو بيانات ناقصة")
        reason = f"{nd_txt}. الحد {thr}."
    reasons.append({"البند": "هيكل دين متحفظ", "الحالة": status_word(sym), "السبب": reason})

    # 6) تغطية الفوائد ≥ 10x
    ic = r["InterestCoverage"]
    ok = (not pd.isna(ic) and ic >= 10.0)
    mid = (not pd.isna(ic) and 6.0 <= ic < 10.0)
    sym = set_flag("تغطية الفوائد", ok, mid=mid)
    score += 10 if ok else (5 if mid else 0)
    if pd.isna(ic):
        reason = "تغطية الفوائد غير متوفرة."
    else:
        thr = "≥ 10x" if ok else ("بين 6x و10x" if mid else "< 6x")
        reason = f"تغطية الفوائد = {to_ratio(ic)}، والحد {thr}."
    reasons.append({"البند": "تغطية الفوائد", "الحالة": status_word(sym), "السبب": reason})

    # 7) دورة التحويل النقدي ≤ 0 يوم (أو ≤ 30 يوم مقبول)
    ccc = r["CCC"]
    ok = (not pd.isna(ccc) and ccc <= 0)
    mid = (not pd.isna(ccc) and ccc <= 30)
    sym = set_flag("دورة التحويل النقدي", ok, mid=mid)
    score += 5 if ok else (2 if mid else 0)
    if pd.isna(ccc):
        reason = "CCC غير متوفرة."
    else:
        thr = "≤ 0 يوم" if ok else ("≤ 30 يوم (مقبول)" if mid else "> 30 يوم")
        reason = f"CCC = {to_days(ccc)}، والحد {thr}."
    reasons.append({"البند": "دورة التحويل النقدي", "الحالة": status_word(sym), "السبب": reason})

    # 8) تقييم معقول: OE Yield ≥ 6% أو P/OE ≤ 20
    oey, pto = r["OwnerEarningsYield"], r["P/OwnerEarnings"]
    ok = (not pd.isna(oey) and oey >= 0.06) or (not pd.isna(pto) and pto <= 20)
    mid = (not pd.isna(oey) and oey >= 0.04) or (not pd.isna(pto) and pto <= 25)
    sym = set_flag("تقييم معقول (OE Yield / P-to-OE)", ok, mid=mid)
    score += 10 if ok else (5 if mid else 0)
    if pd.isna(oey) and pd.isna(pto):
        reason = "بيانات التقييم (أرباح المالك/القيمة السوقية) غير متوفرة."
    else:
        cond = []
        if not pd.isna(oey): cond.append(f"OE Yield = {to_percent(oey)}")
        if not pd.isna(pto): cond.append(f"P/OE = {to_ratio(pto)}")
        thr = "≥ 6% أو ≤ 20x" if ok else ("≥ 4% أو ≤ 25x (مقبول)" if mid else "< 4% و > 25x")
        reason = f"{'، '.join(cond)}. الحد {thr}."
    reasons.append({"البند": "تقييم معقول (OE Yield / P-to-OE)", "الحالة": status_word(sym), "السبب": reason})

    verdict = "✅ جذّابة مع هامش أمان" if score >= 75 else ("🟧 جيدة لكن انتظر سعرًا أفضل" if score >= 55 else "🕒 راقِب")
    return float(score), flags, verdict, net_debt, reasons

# =============================
# نص تحليلي موجز
# =============================
def narrative(symbol, r, score, verdict):
    lines = []
    lines.append(f"**الرمز:** {symbol}")
    lines.append(f"- **جودة العمل:** هامش إجمالي {to_percent(r['GrossMargin'])}، ROIC {to_percent(r['ROIC'])}، وهامش تدفق حر {to_percent(r['FCF_Margin'])}.")
    lines.append(f"- **القوة المالية:** صافي الدين {to_num((r['TotalDebt'] - (0 if pd.isna(r['Cash']) else r['Cash'])) if not pd.isna(r['TotalDebt']) else np.nan)}، وتغطية الفوائد {to_ratio(r['InterestCoverage'])}.")
    lines.append(f"- **التحويل النقدي:** OCF/NI {to_ratio(r['OCF/NI'])}، وCCC {to_days(r['CCC'])}.")
    lines.append(f"- **التقييم (أرباح المالك):** العائد {to_percent(r['OwnerEarningsYield'])}، ومضاعف السعر/أرباح المالك {to_ratio(r['P/OwnerEarnings'])}.")
    lines.append(f"**الخلاصة:** درجة {score:.0f}/100 — {verdict}.")
    return "\n".join(lines)

# =============================
# واجهة المستخدم
# =============================
st.title("📊 التحليل الأساسي بمبادئ بافيت")
st.caption("حساب نسب أساسية مستلهمة من نهج بافيت + قائمة تحقق مع أسباب تفصيلية لكل بند.")

with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1)
    st.markdown("---")
    st.markdown("#### أمثلة سريعة")
    if st.button("USA: AAPL MSFT NVDA"):
        st.session_state.syms = "AAPL MSFT NVDA"
    if st.button("KSA: 1120 2380 1050"):
        st.session_state.syms = "1120 2380 1050"

symbols_input = st.text_area("أدخل الرموز (مسافة/سطر). عند اختيار السوق السعودي سأضيف .SR تلقائياً.", 
                             st.session_state.get("syms",""))

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

    rows, score_rows, texts, notes = [], [], [], []
    reasons_map = {}  # لكل رمز: قائمة أسباب مفصلة
    prog = st.progress(0.0, text="بدء التحليل...")

    for i, sym in enumerate(symbols, start=1):
        try:
            data = load_company_data(sym)
            ratios = compute_buffett_ratios(data, mode)
            score, flags, verdict, net_debt, reasons = buffett_scorecard(ratios)

            # جدول النِّسَب
            rows.append({
                "الرمز": sym,
                "الهامش الإجمالي": to_percent(ratios["GrossMargin"]),
                "هامش التشغيل": to_percent(ratios["OperatingMargin"]),
                "هامش صافي الربح": to_percent(ratios["NetMargin"]),
                "ROIC": to_percent(ratios["ROIC"]),
                "OCF/NI": to_ratio(ratios["OCF/NI"]),
                "هامش التدفق الحر": to_percent(ratios["FCF_Margin"]),
                "تغطية الفوائد": to_ratio(ratios["InterestCoverage"]),
                "CCC": to_days(ratios["CCC"]),
                "صافي الدين": to_num(net_debt),
                "عائد أرباح المالك": to_percent(ratios["OwnerEarningsYield"]),
                "P/أرباح المالك": to_ratio(ratios["P/OwnerEarnings"]),
                "النتيجة/100": f"{score:.0f}",
                "التوصية": verdict
            })

            # قائمة التحقق
            sr = {"الرمز": sym, "الدرجة": f"{score:.0f}/100", **flags, "التوصية": verdict}
            score_rows.append(sr)

            # أسباب مفصلة
            reasons_map[sym] = pd.DataFrame(reasons)

            # نص تحليلي
            texts.append(narrative(sym, ratios, score, verdict))

        except Exception as e:
            notes.append(f"{sym} → {e}")

        prog.progress(i/len(symbols), text=f"تم تحليل {i}/{len(symbols)}")

    if rows:
        st.subheader(f"📋 النِّسَب الأساسية ({mode}) — {len(rows)} شركة")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 تنزيل CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"buffett_ratios_{mode}.csv", mime="text/csv")

    if score_rows:
        st.subheader("✅ قائمة تحقق بافيت (Scoring)")
        dfb = pd.DataFrame(score_rows)
        def html_table(df):
            html = "<table class='buffett-table'><thead><tr>"
            for c in df.columns: html += f"<th>{escape(str(c))}</th>"
            html += "</tr></thead><tbody>"
            for _, row in df.iterrows():
                html += "<tr>"
                for c in df.columns:
                    v = str(row[c]); cls = "green" if v=="✅" else ("yellow" if v=="⚠️" else ("red" if v=="❌" else ""))
                    html += f"<td class='{cls}'>{escape(v)}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            return html
        st.markdown(html_table(dfb), unsafe_allow_html=True)
        st.download_button("📥 تنزيل Buffett Score CSV", dfb.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"buffett_score_{mode}.csv", mime="text/csv")

    if reasons_map:
        st.subheader("📝 أسباب قائمة تحقق بافيت")
        for sym, df_r in reasons_map.items():
            with st.expander(f"أسباب التقييم — {sym}"):
                st.dataframe(df_r, use_container_width=True)
                st.download_button(f"تنزيل الأسباب ({sym}) CSV", df_r.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"buffett_reasons_{sym}_{mode}.csv", mime="text/csv")

    if texts:
        st.subheader("🧠 التحليل النصي (مستلهَم من مبادئ بافيت)")
        for t in texts:
            st.markdown(t)
            st.markdown("---")

    if notes:
        st.info("⚠️ ملاحظات:")
        for n in notes: st.write("•", n)

with st.expander("📌 منهجية مختصرة"):
    st.markdown("""
- **ROIC ≈** NOPAT / رأس المال المستثمر = EBIT×(1–الضريبة) ÷ (الدين + حقوق المساهمين – النقد).
- **أرباح المالك ≈** OCF – Capex؛ **الهامش** = أرباح المالك / الإيراد.
- **جودة الأرباح**: OCF/NI ≥ 1 جيد.
- **التقييم**: OE Yield = OE / القيمة السوقية، و P/OwnerEarnings.
- **TTM**: جمع آخر 4 أرباع لبنود الدخل والتدفقات.
- تُعرض «—» عند نقص البيانات في Yahoo.
""")
