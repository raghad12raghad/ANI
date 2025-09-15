# -*- coding: utf-8 -*-
"""
📊 Financial Analysis Model — Matrix UI (Arabic RTL)
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
st.set_page_config(page_title="📊 نموذج التحليل المالي | Matrix UI", layout="wide")
THEME_CSS = """
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; font-family: -apple-system, Segoe UI, Tahoma, Arial, sans-serif; }
  input, textarea, select { direction: rtl; text-align: right; }
  .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
  .stNumberInput input, .stDateInput input, .stMultiSelect [data-baseweb],
  label, .stButton button { text-align: right; }

  .hero { background:#f8fafc; border:1px solid #e2e8f0; padding:14px 18px; border-radius:14px; margin-bottom:12px; }
  .hero h1 { margin:0; font-size:22px; }
  .muted { color:#475569; font-size:13px; }

  /* جدول المصفوفة (مشابه للصورة) */
  .matrix-table { width:100%; border-collapse:collapse; table-layout: fixed; }
  .matrix-table th, .matrix-table td { border:1px solid #e5e7eb; padding:8px 10px; font-size:13px; vertical-align:middle; }
  .matrix-table th { background:#0ea5e9; color:#fff; font-weight:700; }
  .matrix-table tr:nth-child(even){ background:#f9fafb; }
  .matrix-table .k { text-align:right; width:22%; } /* البند */
  .matrix-table .d { text-align:right; width:38%; color:#334155;} /* الشرح */
  .matrix-table .v { text-align:center; width:14%; font-weight:700;}
  .matrix-table .p { text-align:center; width:14%; font-weight:600; color:#475569;}
  .matrix-table .s { text-align:center; width:12%; font-weight:700;}

  .s.ok   { color:#059669; }   /* أخضر */
  .s.mid  { color:#d97706; }   /* أصفر */
  .s.bad  { color:#dc2626; }   /* أحمر */

  .chip { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #e5e7eb; background:#fff; }
  .chip.ok{ color:#059669; border-color:#bbf7d0;}
  .chip.mid{ color:#d97706; border-color:#fde68a;}
  .chip.bad{ color:#dc2626; border-color:#fecaca;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='hero'><h1>📊 مصفوفة المؤشرات المالية — واجهة شبيهة بالجدول</h1>"
    "<div class='muted'>قيمة تنفيذية أنيقة، مع تفاصيل كاملة تحت مُوسّعات</div></div>",
    unsafe_allow_html=True
)

# =============================
# أدوات مساعدة (تنسيقات/تصنيف)
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

# =============================
# مفاتيح Yahoo
# =============================
REV_KEYS = ["Total Revenue","Revenue","TotalRevenue","Sales"]
COGS_KEYS = ["Cost Of Revenue","Cost of Revenue","CostOfRevenue","COGS"]
GP_KEYS   = ["Gross Profit","GrossProfit"]
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
LTD_KEYS  = ["Long Term Debt"]
SLTD_KEYS = ["Short Long Term Debt"]
CUR_DEBT_KEYS = ["Current Debt"]
TOT_DEBT_KEYS = ["Total Debt"]
INT_EXP_KEYS = ["Interest Expense"]
OCF_KEYS  = ["Operating Cash Flow","Total Cash From Operating Activities"]
CAPEX_KEYS = ["Capital Expenditure","Capital Expenditures"]
PBT_KEYS  = ["Income Before Tax","Pretax Income","Earnings Before Tax"]
TAX_KEYS  = ["Income Tax Expense","Tax Provision","Provision For Income Taxes"]

# =============================
# التحميل — كاش قابل للتسلسل
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

    inc_a = get_df(lambda: t.financials)                # سنوي
    inc_q = get_df(lambda: t.quarterly_financials)      # ربعي
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
# TTM مع إزاحة (للعمود السابق)
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
# حساب المؤشرات لفترة محددة (أحدث/سابقة)
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
        cur_col = bal_cols[0] if bal_cols else None   # أحدث ميزانية (لا معنى لإزاحة ميزانية مع TTM غالبًا)
        prev_col= bal_cols[1] if len(bal_cols)>1 else None
        income_period = "TTM (آخر 4 أرباع" + (f" بإزاحة {offset}" if offset else "") + ")"
    else:
        # سنوي: استخدم عموداً بإزاحة offset
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

    # الميزانية
    ta = find_any(bal, TA_KEYS, cur_col)
    te = find_any(bal, TE_KEYS, cur_col)
    ca = find_any(bal, CA_KEYS, cur_col)
    cl = find_any(bal, CL_KEYS, cur_col)
    inv = find_any(bal, INV_KEYS, cur_col)
    cash = find_any(bal, CASH_KEYS, cur_col)
    sti  = find_any(bal, STI_KEYS, cur_col)

    # مكرر الدين
    total_debt = find_any(bal, TOT_DEBT_KEYS, cur_col)
    if pd.isna(total_debt):
        parts = [find_any(bal, ks, cur_col) for ks in (LTD_KEYS, SLTD_KEYS, CUR_DEBT_KEYS)]
        parts = [x for x in parts if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    # هوامش
    gp = (rev - cogs) if (not pd.isna(rev) and not pd.isna(cogs)) else np.nan
    gross_margin = safe_div(gp, rev)
    op_margin    = safe_div(ebit, rev)
    net_margin   = safe_div(ni, rev)

    # ROIC (تقريبي)
    pbt = np.nan  # غير مستخدم هنا مباشرة
    tax = np.nan
    # حاول استخراج ضريبة فعالة من آخر عمود دخل متاح
    inc_used = inc_q if (mode=="TTM" and not inc_q.empty) else inc_a
    col_income = sorted_cols(inc_used)[0] if not inc_used.empty else None
    pbt = find_any(inc_used, PBT_KEYS, col_income)
    tax = find_any(inc_used, TAX_KEYS, col_income)
    eff_tax = tax / pbt if (not pd.isna(pbt) and pbt != 0 and not pd.isna(tax)) else 0.25
    eff_tax = float(np.clip(eff_tax, 0.0, 0.6))
    nopat = ebit * (1 - eff_tax) if not pd.isna(ebit) else np.nan
    invested = np.nan if (pd.isna(total_debt) or pd.isna(te)) else total_debt + te - (0 if pd.isna(cash) else cash)
    roic = safe_div(nopat, invested)

    # جودة الأرباح/التدفق الحر
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
    ar = find_any(bal, AR_KEYS, cur_col);  ap = find_any(bal, AP_KEYS, cur_col)
    inv_cur = find_any(bal, INV_KEYS, cur_col)

    ar_prev  = find_any(bal, AR_KEYS, prev_col)
    ap_prev  = find_any(bal, AP_KEYS, prev_col)
    inv_prev = find_any(bal, INV_KEYS, prev_col)
    ta_prev  = find_any(bal, TA_KEYS, prev_col)

    avg_assets = np.nanmean([ta, ta_prev])
    asset_turn = safe_div(rev, avg_assets)
    ar_avg  = np.nanmean([ar, ar_prev])
    ap_avg  = np.nanmean([ap, ap_prev])
    inv_avg = np.nanmean([inv_cur, inv_prev])

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

    meta = {
        "income_period": income_period,
        "balance_period": str(cur_col) if cur_col is not None else "—",
        "cashflow_period": "TTM" if (mode=="TTM") else str(sorted_cols(data["cf_a"])[offset]) if (not data["cf_a"].empty and len(data["cf_a"].columns)>offset) else "—",
    }

    return {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "EBIT": ebit, "NetIncome": ni,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv_cur, "Cash": cash, "STInvest": sti, "TotalDebt": total_debt,
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
# بناء مصفوفة الجدول (شبيهة بالصورة)
# =============================
def build_matrix_rows(rcur: dict, rprev: dict):
    rows = []

    def add_row(key, desc, cur_val, prev_val, status):
        rows.append({
            "k": key, "d": desc,
            "v": cur_val, "p": prev_val,
            "s": status
        })

    # 1) حقوق الملكية/الملاءة
    de = safe_div(rcur["TotalDebt"], rcur["TotalEquity"])
    de_prev = safe_div(rprev.get("TotalDebt"), rprev.get("TotalEquity"))
    add_row("هيكل المديونية (D/E)",
            "إجمالي الدين ÷ حقوق الملكية — كلّما أقلّ كان التحمل المالي أمتن.",
            to_ratio(de), to_ratio(de_prev),
            classify(de, ok=0.5, mid=1.0, reverse=True))

    cur = rcur["CurrentRatio"]; cur_prev = rprev.get("CurrentRatio")
    add_row("السيولة الجارية (Current)",
            "الأصول المتداولة ÷ الخصوم المتداولة — قدرة السداد القصير.",
            to_ratio(cur), to_ratio(cur_prev),
            classify(cur, ok=1.5, mid=1.0))

    qk = rcur["QuickRatio"]; qk_prev = rprev.get("QuickRatio")
    add_row("السيولة السريعة (Quick)",
            "استبعاد المخزون لقياس سيولة أشد تحفظًا.",
            to_ratio(qk), to_ratio(qk_prev),
            classify(qk, ok=1.0, mid=0.8))

    # 2) الربحية/العوائد
    add_row("الهامش الإجمالي",
            "Gross Profit ÷ Revenue — قوة التسعير وكفاءة الكلفة.",
            to_percent(rcur["GrossMargin"]), to_percent(rprev.get("GrossMargin")),
            classify(rcur["GrossMargin"], ok=0.25, mid=0.18))

    add_row("هامش التشغيل",
            "EBIT ÷ Revenue — كفاءة العمليات التشغيلية.",
            to_percent(rcur["OperatingMargin"]), to_percent(rprev.get("OperatingMargin")),
            classify(rcur["OperatingMargin"], ok=0.15, mid=0.10))

    add_row("هامش صافي",
            "Net Income ÷ Revenue — ربحية شاملة بعد كل البنود.",
            to_percent(rcur["NetMargin"]), to_percent(rprev.get("NetMargin")),
            classify(rcur["NetMargin"], ok=0.12, mid=0.07))

    add_row("ROE",
            "صافي الربح ÷ حقوق الملكية — عائد المالكين.",
            to_percent(rcur["ROE"]), to_percent(rprev.get("ROE")),
            classify(rcur["ROE"], ok=0.15, mid=0.10))

    add_row("ROIC",
            "NOPAT ÷ (الدين + حقوق – النقد) — العائد على رأس المال المستثمر.",
            to_percent(rcur["ROIC"]), to_percent(rprev.get("ROIC")),
            classify(rcur["ROIC"], ok=0.15, mid=0.10))

    # 3) جودة الأرباح/التدفق الحر
    add_row("جودة الأرباح OCF/NI",
            "≥1.0 يعني النقد يدعم الربح المحاسبي.",
            to_ratio(rcur["OCF/NI"]), to_ratio(rprev.get("OCF/NI")),
            classify(rcur["OCF/NI"], ok=1.0, mid=0.8))

    add_row("هامش أرباح المالك",
            "(OCF - Capex) ÷ Revenue — تقدير للتدفق الحر.",
            to_percent(rcur["FCF_Margin"]), to_percent(rprev.get("FCF_Margin")),
            classify(rcur["FCF_Margin"], ok=0.08, mid=0.05))

    # 4) تغطية الفوائد والملاءة
    add_row("تغطية الفوائد",
            "EBIT ÷ مصروف الفائدة — ≥10x مريح.",
            to_ratio(rcur["InterestCoverage"]), to_ratio(rprev.get("InterestCoverage")),
            classify(rcur["InterestCoverage"], ok=10.0, mid=6.0))

    # 5) كفاءة رأس المال العامل
    add_row("CCC (دورة التحويل النقدي)",
            "DSO + DIO - DPO — أقل أفضل (≤0 مثالي).",
            to_days(rcur["CCC"]), to_days(rprev.get("CCC")),
            classify(rcur["CCC"], ok=0, mid=30, reverse=True))

    # 6) تقييمات السوق
    pe_cur = rcur["PE"]; pe_prev = rprev.get("PE")
    add_row("P/E",
            "السعر ÷ ربح السهم — مرجع تقييم نسبي.",
            ("—" if pd.isna(pe_cur) else f"{pe_cur:.2f}x"),
            ("—" if pd.isna(pe_prev) else f"{pe_prev:.2f}x"),
            classify(1.0/(pe_cur if not pd.isna(pe_cur) and pe_cur>0 else np.nan), ok=0.06, mid=0.04))  # تقريب لعكس P/E

    pb_cur = rcur["PB"]; pb_prev = rprev.get("PB")
    add_row("P/B",
            "السعر ÷ الدفترية — مرجع للقيمة الدفترية.",
            ("—" if pd.isna(pb_cur) else f"{pb_cur:.2f}x"),
            ("—" if pd.isna(pb_prev) else f"{pb_prev:.2f}x"),
            classify(1.0/(pb_cur if not pd.isna(pb_cur) and pb_cur>0 else np.nan), ok=0.20, mid=0.10))

    oey = rcur["OwnerEarningsYield"]; oey_prev = rprev.get("OwnerEarningsYield")
    add_row("OE Yield",
            "أرباح المالك ÷ القيمة السوقية — ≥6% معقول.",
            to_percent(oey), to_percent(oey_prev),
            classify(oey, ok=0.06, mid=0.04))

    return rows

def render_matrix_table(rows):
    # يبني HTML مشابه للصورة، مع تلوين حالة التقييم
    html = ["<table class='matrix-table'>"]
    html.append("<tr><th class='k'>البند</th><th class='d'>شرح النسبة/التعريف</th><th class='v'>أحدث قيمة</th><th class='p'>قيمة سابقة</th><th class='s'>التقييم</th></tr>")
    for r in rows:
        cls = r["s"]
        status_text = {"ok":"جيد","mid":"مقبول","bad":"ضعيف"}.get(cls, "—")
        html.append(
            f"<tr>"
            f"<td class='k'>{escape(str(r['k']))}</td>"
            f"<td class='d'>{escape(str(r['d']))}</td>"
            f"<td class='v'>{escape(str(r['v']))}</td>"
            f"<td class='p'>{escape(str(r['p']))}</td>"
            f"<td class='s {cls}'>{escape(status_text)}</td>"
            f"</tr>"
        )
    html.append("</table>")
    return "\n".join(html)

# =============================
# الشريط الجانبي (مدخلات)
# =============================
with st.sidebar:
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي (.SR)"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("وضع الفترة", ["Annual", "TTM"], index=1)
    st.markdown("---")
    st.markdown("#### إعدادات DCF (على أرباح المالك)")
    disc_rate = st.number_input("معدل الخصم (r)", 0.05, 0.30, 0.12, 0.01)
    growth_rate = st.number_input("نمو السنوات (g)", 0.00, 0.30, 0.05, 0.01)
    years = st.number_input("عدد السنوات", 3, 10, 5, 1)
    term_growth = st.number_input("نمو نهائي (gₜ)", 0.00, 0.05, 0.02, 0.005)
    st.caption("تذكير: r > gₜ وإلا يفشل التقييم.")
    st.markdown("---")
    st.markdown("#### أمثلة")
    if st.button("USA: AAPL"): st.session_state.syms = "AAPL"
    if st.button("KSA: 1120"): st.session_state.syms = "1120"

symbols_input = st.text_input("أدخل رمزًا واحدًا:", st.session_state.get("syms","")).strip()
if symbols_input:
    sym = symbols_input.upper()
    if suffix and sym.isalnum() and not sym.endswith(".SR"): sym = sym + suffix
else:
    sym = ""

# =============================
# التنفيذ
# =============================
if st.button("🚀 تحليل الشركة"):
    if not sym:
        st.warning("يرجى إدخال رمز واحد.")
        st.stop()

    with st.spinner("جاري التحميل والتحليل..."):
        data = load_company_data(sym)
        r_cur  = compute_metrics_for_period(data, mode, offset=0)   # أحدث
        r_prev = compute_metrics_for_period(data, mode, offset=1)   # سابقة

        rows = build_matrix_rows(r_cur, r_prev)
        table_html = render_matrix_table(rows)

        info = data.get("info", {})
        price = r_cur["Price"]
        oe = r_cur["OwnerEarnings"]
        dcf_total, dcf_table = simple_dcf(oe, disc_rate, growth_rate, int(years), term_growth)
        dcf_per_share = (dcf_total / r_cur["Shares"]) if (not pd.isna(dcf_total) and not pd.isna(r_cur["Shares"]) and r_cur["Shares"]>0) else np.nan

    # ===== واجهة الجدول الشبيه بالصورة =====
    st.markdown(f"**الشركة/الرمز:** {(info.get('longName') or sym)} — **الفترة:** {r_cur['_meta'].get('income_period','—')} | **سابق:** {r_prev['_meta'].get('income_period','—')}")
    st.markdown(table_html, unsafe_allow_html=True)

    # ===== شرائح تفصيلية (نفس التفاصيل القديمة ولكن تحت مُوسّعات) =====
    with st.expander("📌 نظرة عامة سريعة"):
        sector = info.get("sector") or "—"
        industry = info.get("industry") or "—"
        st.write(f"- القطاع/الصناعة: **{sector} / {industry}**")
        st.write(f"- السعر الحالي: **{to_num(price)}** | القيمة السوقية: **{to_num(r_cur['MarketCap'])}**")
        st.write(f"- ROIC: **{to_percent(r_cur['ROIC'])}** | Gross: **{to_percent(r_cur['GrossMargin'])}** | OCF/NI: **{to_ratio(r_cur['OCF/NI'])}** | CCC: **{to_days(r_cur['CCC'])}**")

    with st.expander("📈 اتجاهات بسيطة (سنوي فقط عند توفر البيانات)"):
        try:
            # اتجاه مبسّط من القوائم السنوية
            rev_s = pd.Series({str(c): find_any(data["inc_a"], REV_KEYS, c) for c in sorted_cols(data["inc_a"])[:5]})
            ni_s  = pd.Series({str(c): find_any(data["inc_a"], NI_KEYS,  c) for c in sorted_cols(data["inc_a"])[:5]})
            oe_s  = pd.Series({str(c): (find_any(data["cf_a"], OCF_KEYS, c)-find_any(data["cf_a"], CAPEX_KEYS, c)) for c in sorted_cols(data["cf_a"])[:5]})
            chart_df = pd.DataFrame({"Revenue":rev_s, "NetIncome":ni_s, "OwnerEarnings":oe_s}).dropna(how="all")
            st.line_chart(chart_df)
        except Exception:
            st.info("لا تتوفر بيانات تاريخية كافية للرسم.")

    with st.expander("💵 تقييم DCF مبسّط"):
        if not pd.isna(dcf_total):
            st.write("**القيمة الحالية الإجمالية (للشركة):**", to_num(dcf_total))
            if not pd.isna(dcf_per_share):
                st.write("**القيمة الجوهرية/سهم:**", to_num(dcf_per_share))
            st.dataframe(dcf_table, use_container_width=True)
        else:
            st.info("لا يمكن حساب DCF — تحقّق من (OE>0 و r>gₜ).")

    with st.expander("🧾 تفصيل القوائم (مختصر)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**قائمة الدخل (أحدث)**")
            st.dataframe(pd.DataFrame([{
                "الإيرادات": to_num(r_cur["Revenue"]),
                "الربح الإجمالي": to_num(r_cur["GrossProfit"]),
                "EBIT": to_num(r_cur["EBIT"]),
                "صافي الربح": to_num(r_cur["NetIncome"]),
            }]), use_container_width=True)
        with c2:
            st.markdown("**الميزانية (أحدث)**")
            st.dataframe(pd.DataFrame([{
                "الأصول المتداولة": to_num(r_cur["CurrentAssets"]),
                "الخصوم المتداولة": to_num(r_cur["CurrentLiabilities"]),
                "حقوق الملكية": to_num(r_cur["TotalEquity"]),
                "إجمالي الأصول": to_num(r_cur["TotalAssets"]),
                "إجمالي الدين": to_num(r_cur["TotalDebt"]),
                "النقد": to_num(r_cur["Cash"]),
            }]), use_container_width=True)
        with c3:
            st.markdown("**التدفقات النقدية (أحدث)**")
            st.dataframe(pd.DataFrame([{
                "تشغيلي OCF": to_num(r_cur["OCF"]),
                "Capex": to_num(r_cur["Capex"]),
                "أرباح المالك": to_num(r_cur["OwnerEarnings"])
            }]), use_container_width=True)

    with st.expander("🧠 مسرد سريع / لماذا هذه المؤشرات؟"):
        st.markdown("""
- **ROIC**: يعبر عن قدرة الشركة على توليد عائد فوق تكلفة رأس المال — محرّك القيمة الحقيقي.
- **OCF/NI**: جودة الأرباح؛ كل ما كان ≥1.0 كان الربح “نقدي” أكثر.
- **CCC**: سرعة تحويل المبيعات إلى نقد؛ القيم المنخفضة/السالبة تعني دورة تشغيل رشاقة.
- **D/E & الفوائد**: حمولة الديون ومدى أمان تغطية الفوائد.
- **OE Yield**: عائد ضمني للمستثمر بالنسبة للقيمة السوقية (بديل سريع عن DCF).
""")

# تلميح: واجهة “مصفوفة المؤشرات” فوق تماثل النمط المرئي للصورة (جدول واحد غني)،
# بينما أبقينا بقية التفاصيل تحت مُوسّعات للحفاظ على العمق بدون تشتيت الواجهة الأساسية.
