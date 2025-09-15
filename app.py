# -*- coding: utf-8 -*-
"""
📊 Financial Analysis — Matrix UI (Zero-Assumptions + Click-to-Explore)
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
  .matrix-table .d { text-align:right; width:38%; color:#334155;}
  .matrix-table .v { text-align:center; width:14%; font-weight:700;}
  .matrix-table .p { text-align:center; width:12%; font-weight:600; color:#475569;}
  .matrix-table .chg { text-align:center; width:12%; font-weight:700;}

  .chipwrap { display:flex; flex-wrap:wrap; gap:8px; }
  .chipbtn { display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid #e5e7eb;
             background:#fff; font-size:12px; cursor:pointer; }
  .chipbtn.active { border-color:#0ea5e9; color:#0ea5e9; font-weight:700; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><h1>📊 مصفوفة المؤشرات — صفر افتراضات + استكشاف بالنقر</h1>"
    "<div class='muted'>كل القيم مباشرة من القوائم والسوق فقط. لا حدود، لا أحكام، لا DCF.</div></div>",
    unsafe_allow_html=True
)

# =============================
# Utilities (فورمات — بدون افتراضات)
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
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev) or prev == 0: return None
    return (cur - prev) / abs(prev) * 100.0

def pp_change(cur, prev):
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
# تحميل البيانات
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
        "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q,
        "cf_a": cf_a, "cf_q": cf_q, "price": price, "shares": shares,
        "market_cap": market_cap, "info": info
    }

# =============================
# أدوات مساعدة للسلاسل الزمنية
# =============================
def sum_last4_at(df: pd.DataFrame, keys, start_idx=0):
    # يجمع 4 أرباع بدءاً من start_idx (الأعمدة مرتبة تنازلياً = أحدث أولاً)
    if df is None or df.empty: return np.nan
    cols = sorted_cols(df)
    if start_idx + 4 > len(cols): return np.nan
    cols = cols[start_idx:start_idx+4]
    vals = [find_any(df, keys, c) for c in cols]
    vals = [v for v in vals if not pd.isna(v)]
    return sum(vals) if vals else np.nan

def match_balance_col(bal_df: pd.DataFrame, target_col):
    # حاول مطابقته بنفس التاريخ، وإلا اختر العمود الأقرب زمنياً
    if bal_df is None or bal_df.empty or target_col is None:
        return None
    cols = sorted_cols(bal_df)
    if target_col in cols:
        return target_col
    try:
        t = pd.to_datetime(str(target_col))
        diffs = [(abs(pd.to_datetime(str(c)) - t), c) for c in cols]
        diffs.sort(key=lambda x: x[0])
        return diffs[0][1] if diffs else None
    except Exception:
        return cols[0] if cols else None

# =============================
# لقطة فترة واحدة (Series-aware)
# =============================
def snapshot_for_series(data: dict, mode: str, idx: int):
    inc_a, inc_q = data["inc_a"], data["inc_q"]
    bal_a, bal_q = data["bal_a"], data["bal_q"]
    cf_a,  cf_q  = data["cf_a"],  data["cf_q"]

    if mode == "TTM" and not inc_q.empty:
        inc_cols = sorted_cols(inc_q)
        if idx + 4 > len(inc_cols):  # غير كافٍ لحساب TTM
            return None, None, {}
        rev  = sum_last4_at(inc_q, REV_KEYS, idx)
        ebit = sum_last4_at(inc_q, EBIT_KEYS, idx)
        if pd.isna(ebit): ebit = sum_last4_at(inc_q, OPINC_KEYS, idx)
        ni   = sum_last4_at(inc_q, NI_KEYS, idx)
        ocf  = sum_last4_at(cf_q,  OCF_KEYS, idx)
        capex= sum_last4_at(cf_q,  CAPEX_KEYS, idx)
        cogs = sum_last4_at(inc_q, COGS_KEYS, idx)

        cur_income_col = inc_cols[idx]  # نهاية نافذة TTM
        bal_col = match_balance_col(bal_q if not bal_q.empty else bal_a, cur_income_col)
        bal = bal_q if not bal_q.empty else bal_a

        meta = {
            "income_period": f"TTM حتى {str(cur_income_col)}",
            "balance_period": str(bal_col) if bal_col is not None else "—",
            "cashflow_period": f"TTM حتى {str(cur_income_col)}"
        }
    else:
        inc_cols = sorted_cols(inc_a)
        if idx >= len(inc_cols): return None, None, {}
        col_i = inc_cols[idx]
        cf_cols = sorted_cols(cf_a)
        col_c = cf_cols[idx] if idx < len(cf_cols) else None

        rev  = find_any(inc_a, REV_KEYS, col_i)
        ebit = find_any(inc_a, EBIT_KEYS, col_i)
        if pd.isna(ebit): ebit = find_any(inc_a, OPINC_KEYS, col_i)
        ni   = find_any(inc_a, NI_KEYS, col_i)
        ocf  = find_any(cf_a,  OCF_KEYS, col_c)
        capex= find_any(cf_a,  CAPEX_KEYS, col_c)
        cogs = find_any(inc_a, COGS_KEYS, col_i)

        bal = bal_a
        bal_cols = sorted_cols(bal)
        bal_col = bal_cols[idx] if idx < len(bal_cols) else None

        meta = {
            "income_period": str(col_i),
            "balance_period": str(bal_col) if bal_col is not None else "—",
            "cashflow_period": str(col_c) if col_c is not None else "—"
        }

    # ميزانية
    ta = find_any(bal, TA_KEYS, bal_col)
    te = find_any(bal, TE_KEYS, bal_col)
    ca = find_any(bal, CA_KEYS, bal_col)
    cl = find_any(bal, CL_KEYS, bal_col)
    inv = find_any(bal, INV_KEYS, bal_col)
    cash = find_any(bal, CASH_KEYS, bal_col)
    sti  = find_any(bal, STI_KEYS, bal_col)

    # دين
    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_col)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, bal_col) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    # هوامش
    gp = np.nan if (pd.isna(rev) or pd.isna(cogs)) else (rev - cogs)
    gross_margin = safe_div(gp, rev)
    op_margin    = safe_div(ebit, rev)
    net_margin   = safe_div(ni, rev)

    # ROA/ROE
    roa = safe_div(ni, ta)
    roe = safe_div(ni, te)

    # ROIC (بدون افتراضات)
    inc_used = inc_q if (mode=="TTM" and not inc_q.empty) else inc_a
    col_income = sorted_cols(inc_used)[idx] if idx < len(sorted_cols(inc_used)) else None
    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax_exp = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = np.nan
    if not pd.isna(pbt) and pbt != 0 and not pd.isna(tax_exp):
        eff_tax = tax_exp / pbt
    nopat = np.nan if (pd.isna(ebit) or pd.isna(eff_tax)) else (ebit * (1 - eff_tax))
    invested = np.nan
    if not any(pd.isna(x) for x in [total_debt, te, cash]):
        invested = total_debt + te - (cash if pd.isna(sti) else (cash + sti))
    roic = safe_div(nopat, invested)

    # جودة الأرباح/التدفق الحر
    owner_earnings = np.nan if (pd.isna(ocf) or pd.isna(capex)) else (ocf - capex)
    ocf_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(owner_earnings, rev)

    # الفوائد و CCC
    int_exp = find_any(inc_used, INT_EXP_KEYS, col_income)
    if not pd.isna(int_exp): int_exp = abs(int_exp)
    interest_cov = safe_div(ebit, int_exp)

    # CCC (سلسلة: تحتاج نقطة سابقة لمتوسطات)
    # سنحسبها خارجياً عبر لقطتين متتاليتين عند بناء السلسلة.
    r = {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "EBIT": ebit, "NetIncome": ni,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "Cash": cash, "STInvest": sti, "TotalDebt": total_debt,
        "OCF": ocf, "Capex": capex, "OwnerEarnings": owner_earnings,
        "GrossMargin": gross_margin, "OperatingMargin": op_margin, "NetMargin": net_margin,
        "ROA": roa, "ROE": roe, "ROIC": roic, "OCF/NI": ocf_ni, "FCF_Margin": fcf_margin,
        "CurrentRatio": safe_div(ca, cl),
        "QuickRatio": safe_div((ca - (inv if not pd.isna(inv) else 0)), cl),
        "InterestCoverage": interest_cov,
        "_meta": meta
    }
    label = meta["income_period"]
    return r, label, meta

# =============================
# تعريف البنود + دوال استخراج القيمة
# =============================
METRICS = [
    {"key":"DE",      "title":"هيكل المديونية (D/E)", "fmt":"ratio",   "value":lambda r: safe_div(r["TotalDebt"], r["TotalEquity"])},
    {"key":"Current", "title":"السيولة الجارية (Current)", "fmt":"ratio", "value":lambda r: r["CurrentRatio"]},
    {"key":"Quick",   "title":"السيولة السريعة (Quick)", "fmt":"ratio",  "value":lambda r: r["QuickRatio"]},
    {"key":"GrossM",  "title":"الهامش الإجمالي",        "fmt":"percent","value":lambda r: r["GrossMargin"]},
    {"key":"OpM",     "title":"هامش التشغيل",           "fmt":"percent","value":lambda r: r["OperatingMargin"]},
    {"key":"NetM",    "title":"هامش صافي",              "fmt":"percent","value":lambda r: r["NetMargin"]},
    {"key":"ROA",     "title":"ROA",                    "fmt":"percent","value":lambda r: r["ROA"]},
    {"key":"ROE",     "title":"ROE",                    "fmt":"percent","value":lambda r: r["ROE"]},
    {"key":"ROIC",    "title":"ROIC",                   "fmt":"percent","value":lambda r: r["ROIC"]},
    {"key":"OCFNI",   "title":"OCF/NI",                 "fmt":"ratio",  "value":lambda r: r["OCF/NI"]},
    {"key":"FCFMar",  "title":"هامش أرباح المالك",      "fmt":"percent","value":lambda r: r["FCF_Margin"]},
    {"key":"IntCov",  "title":"تغطية الفوائد",           "fmt":"ratio",  "value":lambda r: r["InterestCoverage"]},
    {"key":"CCC",     "title":"CCC (سلسلة عند توفر البيانات)", "fmt":"days", "value":None},  # يحسب بالسلسلة أدناه
    {"key":"Revenue", "title":"الإيرادات",               "fmt":"num",    "value":lambda r: r["Revenue"]},
    {"key":"EBIT",    "title":"EBIT",                    "fmt":"num",    "value":lambda r: r["EBIT"]},
    {"key":"NI",      "title":"صافي الربح",              "fmt":"num",    "value":lambda r: r["NetIncome"]},
    {"key":"OCF",     "title":"التدفق التشغيلي (OCF)",   "fmt":"num",    "value":lambda r: r["OCF"]},
    {"key":"Capex",   "title":"Capex",                   "fmt":"num",    "value":lambda r: r["Capex"]},
    {"key":"OE",      "title":"أرباح المالك",            "fmt":"num",    "value":lambda r: r["OwnerEarnings"]},
]

# =============================
# جدول المصفوفة (أحدث + سابقة)
# =============================
def render_matrix_table_rows(rcur: dict, rprev: dict):
    def fmt(fmt_type, cur, prev):
        if fmt_type == 'percent':
            cur_s, prev_s = to_percent(cur), to_percent(prev)
            diff = pp_change(cur, prev)
            chg = f"{arrow(diff)} {abs(diff):.2f} نقطة" if diff is not None else "—"
        elif fmt_type == 'ratio':
            cur_s, prev_s = to_ratio(cur), to_ratio(prev)
            diff = pct_change(cur, prev)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        elif fmt_type == 'days':
            cur_s, prev_s = to_days(cur), to_days(prev)
            diff = pct_change(cur, prev)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        elif fmt_type == 'num':
            cur_s, prev_s = to_num(cur), to_num(prev)
            diff = pct_change(cur, prev)
            chg = f"{arrow(diff)} {abs(diff):.2f}%" if diff is not None else "—"
        else:
            cur_s = "غير متاح" if (cur is None or pd.isna(cur)) else str(cur)
            prev_s = "غير متاح" if (prev is None or pd.isna(prev)) else str(prev)
            chg = "—"
        return cur_s, prev_s, chg

    html = ["<table class='matrix-table'>"]
    html.append("<tr><th class='k'>البند</th><th class='d'>شرح النسبة/التعريف</th><th class='v'>أحدث قيمة</th><th class='p'>سابقة</th><th class='chg'>التغيّر</th></tr>")

    explanations = {
        "DE":"إجمالي الدين ÷ حقوق الملكية.",
        "Current":"الأصول المتداولة ÷ الخصوم المتداولة.",
        "Quick":"استبعاد المخزون من الأصول المتداولة.",
        "GrossM":"Gross Profit ÷ Revenue.",
        "OpM":"EBIT ÷ Revenue.",
        "NetM":"Net Income ÷ Revenue.",
        "ROA":"صافي الربح ÷ إجمالي الأصول.",
        "ROE":"صافي الربح ÷ حقوق الملكية.",
        "ROIC":"NOPAT ÷ (الدين + حقوق – النقد – الاستثمارات القصيرة) — يحسب فقط عند توفر الضريبة الفعّالة.",
        "OCFNI":"النقد التشغيلي ÷ صافي الربح.",
        "FCFMar":"(OCF - Capex) ÷ Revenue.",
        "IntCov":"EBIT ÷ مصروف الفائدة.",
        "CCC":"DSO + DIO - DPO (سلسلة عند توفر بيانات متتالية).",
        "Revenue":"مبيعات/دخل تشغيلي.",
        "EBIT":"ربح قبل الفوائد والضرائب.",
        "NI":"الربح بعد الضرائب.",
        "OCF":"النقد التشغيلي.",
        "Capex":"الإنفاق الرأسمالي.",
        "OE":"أرباح المالك = OCF - Capex."
    }

    for m in METRICS:
        key = m["key"]; title = m["title"]; fmt_type = m["fmt"]; f = m["value"]
        cur_val = None if f is None else f(rcur)
        prev_val = None if f is None else f(rprev)
        if key == "CCC":
            # CCC للصف الحالي: استخدم قيمة اللقطة الحالية فقط (السلسلة تُحسب في لوحة الاستكشاف)
            cur_val = np.nan
            prev_val = np.nan
        cur_s, prev_s, chg_s = fmt(fmt_type, cur_val, prev_val)
        desc = explanations.get(key, "")
        html.append(
            "<tr>"
            + f"<td class='k'>{escape(title)}</td>"
            + f"<td class='d'>{escape(desc)}</td>"
            + f"<td class='v'>{escape(cur_s)}</td>"
            + f"<td class='p'>{escape(prev_s)}</td>"
            + f"<td class='chg'>{escape(chg_s)}</td>"
            + "</tr>"
        )
    html.append("</table>")
    return "\n".join(html)

# =============================
# بناء السلسلة الزمنية لأي بند
# =============================
def build_timeseries(data: dict, mode: str, metric_key: str):
    # نجمع لقطات متتالية من الأحدث للأقدم، ثم نعكس للعرض زمنياً
    snapshots = []
    # احسب أقصى عدد لقطات ممكنة
    if mode == "TTM" and not data["inc_q"].empty:
        max_n = max(0, len(sorted_cols(data["inc_q"])) - 3)  # كل نافذة 4 أرباع
    else:
        max_n = len(sorted_cols(data["inc_a"])) if not data["inc_a"].empty else 0

    for i in range(0, max_n):
        snap, label, meta = snapshot_for_series(data, mode, i)
        if snap is None:
            continue
        snapshots.append((label, snap))

    # حضّر السلسلة
    labels = []
    values = []

    # دالة القيمة حسب المفتاح
    getter = None
    fmt_type = "num"
    for m in METRICS:
        if m["key"] == metric_key:
            getter = m["value"]
            fmt_type = m["fmt"]
            break

    if metric_key == "CCC":
        # نحتاج متوسطات لاحتساب DSO/DIO/DPO عبر نقاط متعاقبة
        # نحسب فقط حيث تتوفر نقطتان متتاليتان
        for j in range(len(snapshots)-1, 0, -1):  # من الأقدم للأحدث
            label = snapshots[j][0]
            r_cur  = snapshots[j][1]
            r_prev = snapshots[j-1][1]

            # حسابات سنوية/TTM متوافقة مع كل لقطة
            rev = r_cur["Revenue"]; cogs = r_cur["COGS"]
            # متوسطات من الميزانية
            ar_avg  = np.nanmean([r_cur.get("AccountsReceivable", np.nan), r_prev.get("AccountsReceivable", np.nan)])
            # لو ما خزنا AR/AP/Inv بأسماء، نستخدم حقول الميزانية مباشرة:
            ar_avg = np.nanmean([find_any(data["bal_q"] if mode=="TTM" else data["bal_a"], AR_KEYS, None), ar_avg]) if pd.isna(ar_avg) else ar_avg
            ap_avg  = np.nanmean([r_cur.get("AccountsPayable", np.nan), r_prev.get("AccountsPayable", np.nan)])
            ap_avg = np.nanmean([find_any(data["bal_q"] if mode=="TTM" else data["bal_a"], AP_KEYS, None), ap_avg]) if pd.isna(ap_avg) else ap_avg
            inv_avg = np.nanmean([r_cur.get("Inventory", np.nan), r_prev.get("Inventory", np.nan)])

            rec_turn = safe_div(rev, ar_avg)
            pay_turn = safe_div(cogs if not pd.isna(cogs) else rev, ap_avg)
            inv_turn = safe_div(cogs if not pd.isna(cogs) else rev, inv_avg)

            dso = safe_div(365, rec_turn)
            dpo = safe_div(365, pay_turn)
            dio = safe_div(365, inv_turn)
            ccc = dso + dio - dpo if not any(pd.isna(x) for x in [dso, dio, dpo]) else np.nan

            labels.append(label)
            values.append(ccc)
        return fmt_type, labels, values

    if getter is None:
        return fmt_type, [], []

    for j in range(len(snapshots)-1, -1, -1):  # من الأقدم للأحدث
        label = snapshots[j][0]
        r = snapshots[j][1]
        v = getter(r)
        labels.append(label)
        values.append(v)
    return fmt_type, labels, values

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

    with st.spinner("جاري التحميل والتحليل..."):
        data = load_company_data(sym)
        # أحدث/سابقة للمصفوفة التنفيذية
        r_cur, _, _  = snapshot_for_series(data, mode, 0)   # أحدث
        r_prev, _, _ = snapshot_for_series(data, mode, 1)   # سابقة
        info = data.get("info", {})

    st.markdown(
        f"**الشركة/الرمز:** {(info.get('longName') or sym)} — "
        f"**وضع الفترة:** {r_cur['_meta'].get('income_period','—')} | "
        f"**ميزانية:** {r_cur['_meta'].get('balance_period','—')} | "
        f"**تدفقات:** {r_cur['_meta'].get('cashflow_period','—')}"
    )

    # ====== مصفوفة المؤشرات ======
    st.markdown(render_matrix_table_rows(r_cur, r_prev), unsafe_allow_html=True)

    # ====== منطقة النقر/الاستكشاف ======
    st.markdown("### 🖱️ استكشف أي بند عبر الزمن")
    if "sel_key" not in st.session_state:
        st.session_state.sel_key = METRICS[0]["key"]

    # شبكة أزرار (Chips)
    cols_per_row = 4
    for i in range(0, len(METRICS), cols_per_row):
        row = METRICS[i:i+cols_per_row]
        cs = st.columns(len(row))
        for j, m in enumerate(row):
            key = m["key"]; ttl = m["title"]
            active = (st.session_state.sel_key == key)
            btn = cs[j].button(("✅ " if active else "") + ttl, key=f"chip_{key}")
            if btn:
                st.session_state.sel_key = key

    # بناء السلسلة المختارة
    fmt_type, labels, values = build_timeseries(data, mode, st.session_state.sel_key)

    # جدول السلسلة + التغيرات
    st.markdown(f"#### السلسلة الزمنية: {next((m['title'] for m in METRICS if m['key']==st.session_state.sel_key), st.session_state.sel_key)}")
    if labels:
        df = pd.DataFrame({"الفترة": labels, "القيمة": values})
        # فروق
        diffs = [None]
        for k in range(1, len(values)):
            if fmt_type == "percent":
                d = pp_change(values[k], values[k-1])
                diffs.append(None if d is None else round(d, 2))  # نقاط مئوية
            else:
                d = pct_change(values[k], values[k-1])
                diffs.append(None if d is None else round(d, 2))  # %
        df["التغير"] = diffs
        st.dataframe(df, use_container_width=True)

        # عرض كرسم
        show_chart = st.checkbox("عرض كرسم بياني")
        if show_chart:
            try:
                # نرسم القيم كما هي (Streamlit line_chart)
                chart_df = pd.DataFrame({"value": values}, index=pd.Index(labels, name="period"))
                st.line_chart(chart_df)
            except Exception:
                st.info("تعذر الرسم لهذه السلسلة.")
    else:
        st.info("لا تتوفر سلسلة زمنية كافية لهذا البند من المصدر.")

    with st.expander("ℹ️ ملاحظات منهجية"):
        st.markdown("""
- **Annual**: كل نقطة تمثّل عمودًا سنويًا كما هو في Yahoo Finance.
- **TTM**: كل نقطة = مجموع آخر 4 أرباع من نفس تاريخ النهاية، والميزانية تُطابِق تاريخ الربع الأخير قدر الإمكان.
- **ROIC** لا يُعرض إلا إذا توفّر **Tax Expense** و **PBT** لنفس الفترة (لا افتراض لضريبة).
- **CCC** يُحتسب فقط عندما تتوفر نقاط متتالية كافية لمتوسطات AR/AP/Inventory ومبيعات/تكلفة بضائع مطابقة زمنيًا.
- لا حسابات تاريخية لمضاعفات تعتمد على **سعر/عدد أسهم تاريخي** (بياناتها ليست ضمن نفس الواجهة).
""")
