# -*- coding: utf-8 -*-
"""
📊 Financial Analysis — Matrix UI (Zero-Assumptions)
تشغيل: streamlit run app.py
اعتماديات: streamlit, yfinance, pandas, numpy
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from html import escape

# =============================
# إعداد الصفحة + RTL + ستايل
# =============================
st.set_page_config(page_title="📊 Matrix UI — Zero Assumptions", layout="wide")
THEME_CSS = """
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; font-family: -apple-system, Segoe UI, Tahoma, Arial, sans-serif; }
  input, textarea, select { direction: rtl; text-align: right; }
  .hero { background:#f8fafc; border:1px solid #e2e8f0; padding:14px 18px; border-radius:14px; margin-bottom:12px; }
  .hero h1 { margin:0; font-size:22px; }
  .muted { color:#475569; font-size:13px; }

  .matrix-table { width:100%; border-collapse:collapse; table-layout: fixed; }
  .matrix-table th, .matrix-table td { border:1px solid #e5e7eb; padding:8px 10px; font-size:13px; vertical-align:middle; }
  .matrix-table th { background:#0ea5e9; color:#fff; font-weight:700; }
  .matrix-table tr:nth-child(even){ background:#f9fafb; }
  .matrix-table .k { text-align:right; width:24%; }
  .matrix-table .d { text-align:right; width:40%; color:#334155;}
  .matrix-table .v { text-align:center; width:16%; font-weight:700;}
  .matrix-table .p { text-align:center; width:10%; font-weight:600; color:#475569;}
  .matrix-table .chg { text-align:center; width:10%; font-weight:700;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><h1>📊 مصفوفة المؤشرات — صفر افتراضات</h1>"
    "<div class='muted'>كل القيم مباشرة من القوائم والسوق فقط. لا حدود، لا أحكام، لا DCF.</div></div>",
    unsafe_allow_html=True
)

# =============================
# Utilities (فورمات فقط—بدون افتراضات)
# =============================
def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def build_index_map(df: pd.DataFrame):
    return {normalize_idx(raw): raw for raw in df.index.astype(str)}

def find_any(df: pd.DataFrame, keys, col):
    if df is None or df.empty or col is None:
        return np.nan
    idx = build_index_map(df)
    for k in keys:
        kk = normalize_idx(k)
        if kk in idx:
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
    return "غير متاح" if x is None or pd.isna(x) else f"{x*100:.{digits}f}%"

def to_ratio(x, digits=2):
    return "غير متاح" if x is None or pd.isna(x) else f"{x:.{digits}f}x"

def to_days(x, digits=1):
    return "غير متاح" if x is None or pd.isna(x) else f"{x:.{digits}f} يوم"

def to_num(x, digits=2):
    if x is None or pd.isna(x): return "غير متاح"
    ax = abs(float(x))
    if ax >= 1_000_000_000_000: return f"{x/1_000_000_000_000:.{digits}f}T"
    if ax >= 1_000_000_000:     return f"{x/1_000_000_000:.{digits}f}B"
    if ax >= 1_000_000:         return f"{x/1_000_000:.{digits}f}M"
    if ax >= 1_000:             return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

def arrow(v):
    if v is None or pd.isna(v) or v == 0: return "—"
    return "▲" if v > 0 else "▼"

def pct_change(cur, prev):
    # تغيّر نسبي %
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev) or prev == 0: return None
    return (cur - prev) / abs(prev) * 100.0

def pp_change(cur, prev):
    # فرق نقاط مئوية (للنسيات)
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev): return None
    return (cur - prev) * 100.0

# =============================
# مفاتيح Yahoo
# =============================
REV_KEYS = ["Total Revenue","Revenue","TotalRevenue","Sales"]
COGS_KEYS = ["Cost Of Revenue","Cost of Revenue","CostOfRevenue","COGS"]
EBIT_KEYS = ["EBIT","Operating Income","OperatingIncome"]
OPINC_KEYS= ["Operating Income","OperatingIncome"]
NI_KEYS   = ["Net Income","NetIncome","Net Income Common Stockholders","Net Income Applicable To Common Shares"]
TA_KEYS   = ["Total Assets","TotalAssets"]
TE_KEYS   = ["Total Stockholder Equity","Total Shareholder Equity","Total Stockholders Equity","Total Equity Gross Minority Interest"]
CA_KEYS   = ["Total Current Assets","Current Assets","TotalCurrentAssets"]
CL_KEYS   = ["Total Current Liabilities","Current Liabilities","TotalCurrentLiabilities"]
INV_KEYS  = ["Inventory","Inventory Net"]
AR_KEYS   = ["Net Receivables","Accounts Receivable","Receivables"]
AP_KEYS   = ["Accounts Payable","Payables"]
CASH_KEYS = ["Cash And Cash Equivalents","Cash And Cash Equivalents, And Short Term Investments","Cash"]
STI_KEYS  = ["Short Term Investments"]
TOT_DEBT_KEYS = ["Total Debt"]
LTD_KEYS  = ["Long Term Debt"]
SLTD_KEYS = ["Short Long Term Debt"]
CUR_DEBT_KEYS = ["Current Debt"]
INT_EXP_KEYS = ["Interest Expense"]
OCF_KEYS  = ["Operating Cash Flow","Total Cash From Operating Activities"]
CAPEX_KEYS = ["Capital Expenditure","Capital Expenditures"]
PBT_KEYS  = ["Income Before Tax","Pretax Income","Earnings Before Tax"]
TAX_KEYS  = ["Income Tax Expense","Tax Provision","Provision For Income Taxes"]

# =============================
# تحميل البيانات (بدون افتراضات)
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

    # لا نحسب أي شيء غير متاح.
    return {
        "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q,
        "cf_a": cf_a, "cf_q": cf_q, "price": price, "shares": shares,
        "market_cap": market_cap, "info": info
    }

# =============================
# TTM (آخر 4 أرباع) بلا افتراضات
# =============================
def sum_last4_offset(df: pd.DataFrame, keys, offset=0):
    if df is None or df.empty: return np.nan
    cols_all = sorted_cols(df)
    cols = cols_all[offset:offset+4]
    if len(cols) < 4: return np.nan
    vals = [find_any(df, keys, c) for c in cols]
    vals = [v for v in vals if not pd.isna(v)]
    return sum(vals) if vals else np.nan

# =============================
# حساب المؤشرات للفترة (أحدث/سابقة)
# — لا افتراضات: أي عنصر ناقص => الناتج غير متاح
# =============================
def compute_metrics_for_period(data: dict, mode: str, offset: int = 0):
    inc_a, inc_q = data["inc_a"], data["inc_q"]
    bal_a, bal_q = data["bal_a"], data["bal_q"]
    cf_a,  cf_q  = data["cf_a"],  data["cf_q"]

    if mode == "TTM" and not inc_q.empty:
        rev  = sum_last4_offset(inc_q, REV_KEYS, offset)
        ebit = sum_last4_offset(inc_q, EBIT_KEYS, offset)
        if pd.isna(ebit): ebit = sum_last4_offset(inc_q, OPINC_KEYS, offset)
        ni   = sum_last4_offset(inc_q, NI_KEYS, offset)
        ocf  = sum_last4_offset(cf_q, OCF_KEYS, offset)
        capex= sum_last4_offset(cf_q, CAPEX_KEYS, offset)
        cogs = sum_last4_offset(inc_q, COGS_KEYS, offset)
        bal  = bal_q if not bal_q.empty else bal_a
        bal_cols = sorted_cols(bal)
        cur_col = bal_cols[0] if bal_cols else None
        prev_col= bal_cols[1] if len(bal_cols)>1 else None
        income_period = "TTM"
    else:
        col_i = sorted_cols(inc_a)[offset] if not inc_a.empty and len(inc_a.columns)>offset else None
        col_c = sorted_cols(cf_a)[offset]  if not cf_a.empty  and len(cf_a.columns)>offset  else None
        rev  = find_any(inc_a, REV_KEYS, col_i)
        ebit = find_any(inc_a, EBIT_KEYS, col_i)
        if pd.isna(ebit): ebit = find_any(inc_a, OPINC_KEYS, col_i)
        ni   = find_any(inc_a, NI_KEYS, col_i)
        ocf  = find_any(cf_a, OCF_KEYS, col_c)
        capex= find_any(cf_a, CAPEX_KEYS, col_c)
        cogs = find_any(inc_a, COGS_KEYS, col_i)
        bal  = bal_a
        bal_cols = sorted_cols(bal)
        cur_col = bal_cols[offset] if len(bal_cols)>offset else None
        prev_col= bal_cols[offset+1] if len(bal_cols)>(offset+1) else None
        income_period = str(col_i) if col_i is not None else "—"

    # ميزانية
    ta = find_any(bal, TA_KEYS, cur_col)
    te = find_any(bal, TE_KEYS, cur_col)
    ca = find_any(bal, CA_KEYS, cur_col)
    cl = find_any(bal, CL_KEYS, cur_col)
    inv = find_any(bal, INV_KEYS, cur_col)
    cash = find_any(bal, CASH_KEYS, cur_col)
    sti  = find_any(bal, STI_KEYS, cur_col)

    # الدين الإجمالي (من بند Total Debt، وإن غاب نحاول تجميع مكونات الدين—هذا ليس افتراضاً بل تجميع مباشر)
    total_debt = find_any(bal, TOT_DEBT_KEYS, cur_col)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, cur_col) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    # هوامش
    gp = np.nan if (pd.isna(rev) or pd.isna(cogs)) else (rev - cogs)
    gross_margin = safe_div(gp, rev)
    op_margin    = safe_div(ebit, rev)
    net_margin   = safe_div(ni, rev)

    # ROA / ROE (مباشر)
    roa = safe_div(ni, ta)
    roe = safe_div(ni, te)

    # ROIC (بدون افتراض ضريبة): إن توفر كل من PBT والضرائب نحسب معدل الضريبة الفعّال، غير ذلك => غير متاح
    inc_used = inc_q if (mode=="TTM" and not inc_q.empty) else inc_a
    col_income = sorted_cols(inc_used)[0] if not inc_used.empty else None
    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax_exp = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = np.nan
    if not pd.isna(pbt) and pbt != 0 and not pd.isna(tax_exp):
        eff_tax = tax_exp / pbt
    nopat = np.nan if (pd.isna(ebit) or pd.isna(eff_tax)) else (ebit * (1 - eff_tax))
    invested = np.nan
    # تعريف محدد (بدون افتراض): رأس المال المستثمر = TotalDebt + TotalEquity - Cash - STInvest
    if not any(pd.isna(x) for x in [total_debt, te, cash]):
        invested = total_debt + te - (cash if pd.isna(sti) else (cash + sti))
    roic = safe_div(nopat, invested)

    # جودة الأرباح/التدفق الحر (تعريف مباشر)
    owner_earnings = np.nan if (pd.isna(ocf) or pd.isna(capex)) else (ocf - capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(owner_earnings, rev)

    # تغطية الفوائد (مباشر)
    int_exp = find_any(inc_used, INT_EXP_KEYS, col_income)
    if not pd.isna(int_exp): int_exp = abs(int_exp)
    interest_cov = safe_div(ebit, int_exp)

    # كفاءة رأس المال العامل + CCC
    ar = find_any(bal, AR_KEYS, cur_col);  ar_prev  = find_any(bal, AR_KEYS, prev_col)
    ap = find_any(bal, AP_KEYS, cur_col);  ap_prev  = find_any(bal, AP_KEYS, prev_col)
    inv_prev = find_any(bal, INV_KEYS, prev_col)
    ta_prev  = find_any(bal, TA_KEYS, prev_col)

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

    # السوق/التقييم (مباشر)
    price = data.get("price", np.nan)
    shares = data.get("shares", np.nan)
    market_cap = data.get("market_cap", np.nan)
    eps = safe_div(ni, shares)
    pe = safe_div(price, eps)
    bvps = safe_div(te, shares)
    pb = safe_div(price, bvps)
    sales_ps = safe_div(rev, shares)
    ps = safe_div(price, sales_ps)

    if (pd.isna(market_cap) or market_cap == 0) and (not pd.isna(price) and not pd.isna(shares)):
        market_cap = price * shares
    oe_yield = safe_div(owner_earnings, market_cap)

    meta = {
        "income_period": income_period,
        "balance_period": str(cur_col) if cur_col is not None else "—",
        "cashflow_period": "TTM" if (mode=="TTM") else (str(sorted_cols(data["cf_a"])[offset]) if (not data["cf_a"].empty and len(data["cf_a"].columns)>offset) else "—"),
    }

    return {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "EBIT": ebit, "NetIncome": ni,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "Cash": cash, "STInvest": sti, "TotalDebt": total_debt,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": owner_earnings,
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "ROA": roa, "ROE": roe, "ROIC": roic, "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "CurrentRatio": safe_div(ca, cl),
        "QuickRatio": safe_div((ca - (inv if not pd.isna(inv) else 0)), cl),
        "InterestCoverage": interest_cov,
        "AssetTurnover": asset_turn, "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        "Price": price, "Shares": shares, "MarketCap": market_cap,
        "PE": pe, "PB": pb, "PS": ps, "BVPS": bvps, "OE_Yield": oe_yield,
        "_meta": meta
    }

# =============================
# بناء صفوف المصفوفة (بدون تقييمات)
# =============================
def build_matrix_rows(rcur: dict, rprev: dict):
    rows = []

    def add_row(title, desc, cur_val, prev_val, fmt_type):
        # fmt_type: 'num' | 'ratio' | 'percent' | 'days' | 'plain'
        if fmt_type == 'percent':
            cur_s = to_percent(cur_val); prev_s = to_percent(prev_val)
            diff = pp_change(cur_val, prev_val)
            chg = f"{arrow(diff)} {abs(diff):.2f} نقطة" if diff is not None else "—"
        elif fmt_type == 'ratio':
            cur_s = to_ratio(cur_val); prev_s = to_ratio(prev_val)
            diff = pct_change(cur_val, prev_val)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        elif fmt_type == 'days':
            cur_s = to_days(cur_val); prev_s = to_days(prev_val)
            diff = pct_change(cur_val, prev_val)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        elif fmt_type == 'num':
            cur_s = to_num(cur_val); prev_s = to_num(prev_val)
            diff = pct_change(cur_val, prev_val)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        else:
            cur_s = "غير متاح" if (cur_val is None or pd.isna(cur_val)) else str(cur_val)
            prev_s = "غير متاح" if (prev_val is None or pd.isna(prev_val)) else str(prev_val)
            chg = "—"
        rows.append({"k": title, "d": desc, "v": cur_s, "p": prev_s, "chg": chg})

    # الملاءة والسيولة
    add_row("هيكل المديونية (D/E)", "إجمالي الدين ÷ حقوق الملكية.", 
            safe_div(rcur["TotalDebt"], rcur["TotalEquity"]), 
            safe_div(rprev.get("TotalDebt"), rprev.get("TotalEquity")), 'ratio')
    add_row("السيولة الجارية (Current)", "الأصول المتداولة ÷ الخصوم المتداولة.",
            safe_div(rcur["CurrentAssets"], rcur["CurrentLiabilities"]),
            safe_div(rprev.get("CurrentAssets"), rprev.get("CurrentLiabilities")), 'ratio')
    add_row("السيولة السريعة (Quick)", "استبعاد المخزون من الأصول المتداولة.",
            rcur["QuickRatio"], rprev.get("QuickRatio"), 'ratio')

    # ربحية وهوامش
    add_row("الهامش الإجمالي", "Gross Profit ÷ Revenue.",
            rcur["GrossMargin"], rprev.get("GrossMargin"), 'percent')
    add_row("هامش التشغيل", "EBIT ÷ Revenue.",
            rcur["OperatingMargin"], rprev.get("OperatingMargin"), 'percent')
    add_row("هامش صافي", "Net Income ÷ Revenue.",
            rcur["NetMargin"], rprev.get("NetMargin"), 'percent')

    # عوائد
    add_row("ROA", "صافي الربح ÷ إجمالي الأصول.",
            rcur["ROA"], rprev.get("ROA"), 'percent')
    add_row("ROE", "صافي الربح ÷ حقوق الملكية.",
            rcur["ROE"], rprev.get("ROE"), 'percent')
    add_row("ROIC", "NOPAT ÷ (الدين + حقوق – النقد – الاستثمارات القصيرة). (محسوب فقط عند توفر الضريبة الفعّالة)",
            rcur["ROIC"], rprev.get("ROIC"), 'percent')

    # جودة أرباح/تدفق حر
    add_row("OCF/NI", "النقد التشغيلي ÷ صافي الربح.",
            rcur["OCF/NI"], rprev.get("OCF/NI"), 'ratio')
    add_row("هامش أرباح المالك", "(OCF - Capex) ÷ Revenue.",
            rcur["FCF_Margin"], rprev.get("FCF_Margin"), 'percent')

    # الفوائد والـ CCC
    add_row("تغطية الفوائد", "EBIT ÷ مصروف الفائدة.",
        rcur["InterestCoverage"], rprev.get("InterestCoverage"), 'ratio')
    add_row("CCC", "DSO + DIO - DPO.",
        rcur["CCC"], rprev.get("CCC"), 'days')

    # التقييمات السوقية
    add_row("P/E", "السعر ÷ ربح السهم.",
        rcur["PE"], rprev.get("PE"), 'ratio')
    add_row("P/B", "السعر ÷ القيمة الدفترية للسهم.",
        rcur["PB"], rprev.get("PB"), 'ratio')
    add_row("P/S", "السعر ÷ المبيعات للسهم.",
        rcur["PS"], rprev.get("PS"), 'ratio')
    add_row("OE Yield", "أرباح المالك ÷ القيمة السوقية.",
        rcur["OE_Yield"], rprev.get("OE_Yield"), 'percent')

    return rows

def render_matrix_table(rows):
    html = ["<table class='matrix-table'>"]
    html.append("<tr><th class='k'>البند</th><th class='d'>شرح النسبة/التعريف</th><th class='v'>أحدث قيمة</th><th class='p'>سابقة</th><th class='chg'>التغيّر</th></tr>")
    for r in rows:
        html.append(
            f"<tr>"
            f"<td class='k'>{escape(str(r['k']))}</td>"
            f"<td class='d'>{escape(str(r['d']))}</td>"
            f"<td class='v'>{escape(str(r['v']))}</td>"
            f"<td class='p'>{escape(str(r['p']))}</td>"
            f"<td class='chg'>{escape(str(r['chg']))}</td>"
            f"</tr>"
        )
    html.append("</table>")
    return "\n".join(html)

# =============================
# الشريط الجانبي
# =============================
with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1)
    st.markdown("---")
    if st.button("USA: AAPL"): st.session_state.syms = "AAPL"
    if st.button("KSA: 1120"): st.session_state.syms = "1120"

symbols_input = st.text_input("أدخل رمزًا واحدًا:", st.session_state.get("syms","")).strip()
sym = symbols_input.upper() if symbols_input else ""
if suffix and sym and sym.isalnum() and not sym.endswith(".SR"):
    sym = sym + suffix

# =============================
# التنفيذ
# =============================
if st.button("🚀 تحليل الشركة"):
    if not sym:
        st.warning("يرجى إدخال رمز واحد.")
        st.stop()

    with st.spinner("جاري التحميل والتحليل بدون افتراضات..."):
        data = load_company_data(sym)
        r_cur  = compute_metrics_for_period(data, mode, offset=0)
        r_prev = compute_metrics_for_period(data, mode, offset=1)

        rows = build_matrix_rows(r_cur, r_prev)
        table_html = render_matrix_table(rows)
        info = data.get("info", {})

    st.markdown(
        f"**الشركة/الرمز:** {(info.get('longName') or sym)} — "
        f"**وضع الفترة:** {r_cur['_meta'].get('income_period','—')} | "
        f"**ميزانية:** {r_cur['_meta'].get('balance_period','—')} | "
        f"**تدفقات:** {r_cur['_meta'].get('cashflow_period','—')}"
    )

    st.markdown(table_html, unsafe_allow_html=True)

    with st.expander("📌 نظرة عامة (مباشرة من المصدر)"):
        sector = info.get("sector") or "—"
        industry = info.get("industry") or "—"
        st.write(f"- القطاع/الصناعة: **{sector} / {industry}**")
        st.write(f"- السعر الحالي: **{to_num(r_cur['Price'])}** | القيمة السوقية: **{to_num(r_cur['MarketCap'])}**")
        st.caption("جميع القيم محوسبة من القوائم/السوق فقط. عند غياب بند، تظهر 'غير متاح' بلا أي تعويض.")

    with st.expander("📈 اتجاهات بسيطة (سنوي عند توفر البيانات)"):
        try:
            rev_s = pd.Series({str(c): find_any(data["inc_a"], REV_KEYS, c) for c in sorted_cols(data["inc_a"])[:6]})
            ni_s  = pd.Series({str(c): find_any(data["inc_a"], NI_KEYS,  c) for c in sorted_cols(data["inc_a"])[:6]})
            ocf_s = pd.Series({str(c): find_any(data["cf_a"], OCF_KEYS, c) for c in sorted_cols(data["cf_a"])[:6]})
            cap_s = pd.Series({str(c): find_any(data["cf_a"], CAPEX_KEYS, c) for c in sorted_cols(data["cf_a"])[:6]})
            oe_s  = ocf_s - cap_s
            chart_df = pd.DataFrame({"Revenue":rev_s, "NetIncome":ni_s, "OwnerEarnings":oe_s}).dropna(how="all")
            st.line_chart(chart_df)
        except Exception:
            st.info("لا تتوفر بيانات كافية للرسم.")
