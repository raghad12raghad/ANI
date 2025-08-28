# === TriplePower Fundamentals — تحليل أساسي مُحافظ مع نسب وتقييم نوعي ===
# الكاتب: Saeed + GPT-5 Thinking (منقّح)
# المتطلبات: streamlit, yfinance, pandas, numpy, python 3.10+
# التشغيل: streamlit run app.py

import os, re, math, warnings
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from html import escape

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="📊 التحليل الأساسي | Buffett-Style", layout="wide")

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
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# =============================
# أدوات مساعدة مشتركة
# =============================

def safe_div(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan

def to_percent(x, digits=2):
    if x is None or pd.isna(x): return "—"
    return f"{x*100:.{digits}f}%"

def to_num(x, digits=2):
    if x is None or pd.isna(x): return "—"
    absx = abs(x)
    if absx >= 1_000_000_000: return f"{x/1_000_000_000:.{digits}f}B"
    if absx >= 1_000_000:      return f"{x/1_000_000:.{digits}f}M"
    if absx >= 1_000:          return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def build_index_map(df: pd.DataFrame):
    idx = {}
    for raw in df.index.astype(str):
        idx[normalize_idx(raw)] = raw
    return idx

def find_any(df: pd.DataFrame, keys: list[str], col):
    if df is None or df.empty: return np.nan
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

def sorted_cols(df: pd.DataFrame):
    try:
        cols = sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=True)
        return cols
    except Exception:
        return list(df.columns)

def capex_outflow(value):
    """Yahoo يسجل CapEx عادةً بسالب؛ نحتاج قيمة موجبة كإنفاق."""
    if value is None or pd.isna(value): return np.nan
    return abs(float(value))

def html_table(df: pd.DataFrame) -> str:
    html = """
    <style>
    table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    th {background-color: #0ea5e9; color: white;}
    tr:nth-child(even){background-color: #f8fafc;}
    tr:hover {background-color: #eef2ff;}
    </style>
    <table><thead><tr>"""
    for col in df.columns:
        html += f"<th>{escape(str(col))}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            html += f"<td>{escape(str(row[col]))}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# مرادفات البنود (Yahoo Finance)
REV_KEYS = ["Total Revenue", "Revenue", "TotalRevenue", "Sales"]
COGS_KEYS = ["Cost Of Revenue", "Cost of Revenue", "CostOfRevenue", "Cost Of Goods Sold", "COGS"]
GP_KEYS   = ["Gross Profit", "GrossProfit"]
OPINC_KEYS= ["Operating Income", "OperatingIncome", "EBIT"]  # EBIT كمساعد
EBIT_KEYS = ["EBIT", "Operating Income", "OperatingIncome"]
NI_KEYS   = ["Net Income", "NetIncome", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
PBT_KEYS  = ["Income Before Tax", "Pretax Income", "Earnings Before Tax"]
TAX_KEYS  = ["Income Tax Expense", "Tax Provision", "Provision For Income Taxes"]

TA_KEYS   = ["Total Assets", "TotalAssets"]
TE_KEYS   = ["Total Stockholder Equity", "Total Shareholder Equity", "Total Equity Gross Minority Interest", "Total Stockholders Equity"]
CA_KEYS   = ["Total Current Assets", "Current Assets", "TotalCurrentAssets"]
CL_KEYS   = ["Total Current Liabilities", "Current Liabilities", "TotalCurrentLiabilities"]
INV_KEYS  = ["Inventory", "Inventory Net"]
AR_KEYS   = ["Net Receivables", "Accounts Receivable", "Receivables"]
AP_KEYS   = ["Accounts Payable", "Payables"]
CASH_KEYS = ["Cash And Cash Equivalents", "Cash And Cash Equivalents, And Short Term Investments", "Cash"]
STI_KEYS  = ["Short Term Investments"]
LTD_KEYS  = ["Long Term Debt"]
SLTD_KEYS = ["Short Long Term Debt"]
CUR_DEBT_KEYS = ["Current Debt"]
TOT_DEBT_KEYS = ["Total Debt"]

INT_EXP_KEYS = ["Interest Expense"]
OCF_KEYS  = ["Operating Cash Flow", "Total Cash From Operating Activities"]
CAPEX_KEYS = ["Capital Expenditure", "Capital Expenditures"]
DA_KEYS = ["Depreciation", "Depreciation & Amortization", "Depreciation Amortization Depletion"]

# =============================
# تحميل بيانات شركة من Yahoo
# =============================

@st.cache_data(ttl=3600)
def load_company_data(ticker: str):
    """يرجع dict يحوي القوائم المالية السنوية/الربع سنوية + السعر/الأسهم/القيمة السوقية + تاريخ الأسهم إن توفر."""
    t = yf.Ticker(ticker)
    def _df(getter, fallback=None):
        try:
            val = getter()
            return val if isinstance(val, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame() if fallback is None else fallback

    inc_a = _df(lambda: t.financials)
    inc_q = _df(lambda: t.quarterly_financials)
    bal_a = _df(lambda: t.balance_sheet)
    bal_q = _df(lambda: t.quarterly_balance_sheet)
    cf_a  = _df(lambda: t.cashflow)
    cf_q  = _df(lambda: t.quarterly_cashflow)

    price = np.nan; shares = np.nan; mcap = np.nan
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        shares = float(fi.get("shares", np.nan))
        mcap = float(fi.get("market_cap", np.nan))
    except Exception:
        pass
    # بدائل
    if (pd.isna(price) or price == 0):
        try:
            hist = t.history(period="1d")
            if not hist.empty: price = float(hist["Close"].iloc[-1])
        except Exception: pass
    if (pd.isna(shares) or shares == 0):
        try:
            info = t.get_info()
            shares = float(info.get("sharesOutstanding", np.nan))
            if pd.isna(mcap): mcap = float(info.get("marketCap", np.nan))
        except Exception: pass
    if pd.isna(mcap) and not pd.isna(price) and not pd.isna(shares) and shares>0:
        mcap = price * shares

    # تاريخ الأسهم (اختياري)
    shares_hist = pd.Series(dtype=float)
    try:
        s = t.get_shares_full()
        if isinstance(s, pd.Series) and not s.empty:
            shares_hist = s.dropna()
    except Exception:
        pass

    return {
        "inc_a": inc_a, "inc_q": inc_q,
        "bal_a": bal_a, "bal_q": bal_q,
        "cf_a":  cf_a,  "cf_q":  cf_q,
        "price": price, "shares": shares, "mcap": mcap,
        "shares_hist": shares_hist
    }

# =============================
# حساب النِّسَب — سنوي أو TTM + اتجاهات
# =============================

def compute_cagr_5y(inc_a: pd.DataFrame):
    rev_cagr = np.nan; ni_cagr = np.nan
    if inc_a is None or inc_a.empty: return rev_cagr, ni_cagr
    cols = sorted_cols(inc_a)
    if len(cols) < 2: return rev_cagr, ni_cagr
    # نحاول 5 فترات؛ إن لم تتوفر نستخدم أول/آخر
    use = cols[:min(5, len(cols))]
    first, last = use[-1], use[0]  # الأقدم، الأحدث
    rev_first = find_any(inc_a, REV_KEYS, first)
    rev_last  = find_any(inc_a, REV_KEYS, last)
    ni_first  = find_any(inc_a, NI_KEYS, first)
    ni_last   = find_any(inc_a, NI_KEYS, last)
    years = max(1, len(use)-1)
    try:
        if rev_first and rev_first>0 and rev_last and rev_last>0:
            rev_cagr = (rev_last/rev_first)**(1/years)-1
    except Exception: pass
    try:
        if ni_first and abs(ni_first)>0 and ni_last and abs(ni_last)>0 and ni_first>0 and ni_last>0:
            ni_cagr = (ni_last/ni_first)**(1/years)-1
    except Exception:
        # صافي الدخل قد يتقلب حول الصفر؛ نتجاهل CAGR حينها
        pass
    return rev_cagr, ni_cagr

def margin_stability_trend(inc_a: pd.DataFrame):
    """انحراف معياري لهامش إجمالي الربح واتجاه بسيط بين أول سنتين وآخر سنتين."""
    if inc_a is None or inc_a.empty: return np.nan, np.nan
    cols = sorted_cols(inc_a)
    take = cols[:min(6, len(cols))]
    margins = []
    for c in take:
        rev = find_any(inc_a, REV_KEYS, c)
        gp  = find_any(inc_a, GP_KEYS,  c)
        if not pd.isna(rev) and rev!=0 and not pd.isna(gp):
            margins.append(gp/rev)
    if len(margins) < 3: return np.nan, np.nan
    margins = list(reversed(margins))  # زمنياً من الأقدم إلى الأحدث
    std = float(np.nanstd(margins))
    first_avg = np.nanmean(margins[:2]) if len(margins)>=2 else np.nan
    last_avg  = np.nanmean(margins[-2:]) if len(margins)>=2 else np.nan
    trend = np.nan if (pd.isna(first_avg) or pd.isna(last_avg)) else (last_avg - first_avg)
    return std, trend

def compute_ratios(data: dict, mode: str = "Annual", maint_capex_ratio: float = 0.7):
    """
    mode: "Annual" أو "TTM"
    يعيد: core (عرض)، raw (بنود)، trends، checklist_inputs (للتقييم)
    """
    inc = data["inc_a"]; bal = data["bal_a"]; cf = data["cf_a"]
    quarterly = False
    if mode == "TTM" and not data["inc_q"].empty:
        inc = data["inc_q"].copy()
        bal = data["bal_q"] if not data["bal_q"].empty else data["bal_a"]
        cf  = data["cf_q"]  if not data["cf_q"].empty  else data["cf_a"]
        quarterly = True

    if inc is None or inc.empty or bal is None or bal.empty:
        return None, None, None, None

    inc_cols = sorted_cols(inc)
    bal_cols = sorted_cols(bal)
    cf_cols  = sorted_cols(cf) if cf is not None and not cf.empty else []

    use_inc_cols = inc_cols[:4] if quarterly else inc_cols[:1]
    use_cf_cols  = cf_cols[:4]  if quarterly else (cf_cols[:1] if cf_cols else [])

    # دخل
    rev  = sum([find_any(inc, REV_KEYS, c) for c in use_inc_cols])
    cogs = sum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols])
    gp   = sum([find_any(inc, GP_KEYS,   c) for c in use_inc_cols])
    opi  = sum([find_any(inc, OPINC_KEYS, c) for c in use_inc_cols])
    ni   = sum([find_any(inc, NI_KEYS,    c) for c in use_inc_cols])
    pbt  = sum([find_any(inc, PBT_KEYS,   c) for c in use_inc_cols])
    tax  = sum([find_any(inc, TAX_KEYS,   c) for c in use_inc_cols])
    tax_rate = safe_div(tax, pbt)
    ebit = sum([find_any(inc, EBIT_KEYS,  c) for c in use_inc_cols])
    if pd.isna(ebit) or ebit == 0: ebit = opi

    # ميزانية (أحدث + سابقة للمتوسطات)
    bal_curr = bal_cols[0] if bal_cols else None
    bal_prev = bal_cols[1] if len(bal_cols) > 1 else None
    ta   = find_any(bal, TA_KEYS, bal_curr)
    te   = find_any(bal, TE_KEYS, bal_curr)
    ca   = find_any(bal, CA_KEYS, bal_curr)
    cl   = find_any(bal, CL_KEYS, bal_curr)
    inv  = find_any(bal, INV_KEYS, bal_curr)
    ar   = find_any(bal, AR_KEYS,  bal_curr)
    ap   = find_any(bal, AP_KEYS,  bal_curr)
    cash = find_any(bal, CASH_KEYS, bal_curr)
    sti  = find_any(bal, STI_KEYS,  bal_curr)
    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_curr)
    if pd.isna(total_debt) or total_debt == 0:
        ltd  = find_any(bal, LTD_KEYS, bal_curr)
        sltd = find_any(bal, SLTD_KEYS, bal_curr)
        cdebt= find_any(bal, CUR_DEBT_KEYS, bal_curr)
        parts = [x for x in [ltd, sltd, cdebt] if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    ta_prev = find_any(bal, TA_KEYS, bal_prev) if bal_prev else np.nan
    te_prev = find_any(bal, TE_KEYS, bal_prev) if bal_prev else np.nan
    avg_assets = np.nanmean([ta, ta_prev]) if not pd.isna(ta) else np.nan
    avg_equity = np.nanmean([te, te_prev]) if not pd.isna(te) else np.nan

    # تدفقات نقدية
    if cf is not None and not cf.empty and use_cf_cols:
        ocf = sum([find_any(cf, OCF_KEYS, c) for c in use_cf_cols])
        capex_vals = [find_any(cf, CAPEX_KEYS, c) for c in use_cf_cols]
        capex = sum([x for x in capex_vals if not pd.isna(x)])
        capex_out = capex_outflow(capex)  # تصحيح الإشارة
        # D&A (اختياري للـ Owner Earnings)
        da_vals = [find_any(cf, DA_KEYS, c) for c in use_cf_cols]
        da = sum([x for x in da_vals if not pd.isna(x)])
    else:
        ocf = np.nan; capex_out = np.nan; da = np.nan

    int_exp = sum([find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols])
    int_exp_abs = abs(int_exp) if not pd.isna(int_exp) else np.nan

    # هوامش وربحية
    gross_margin     = safe_div(gp,  rev)
    operating_margin = safe_div(opi, rev)
    net_margin       = safe_div(ni,  rev)
    roe              = safe_div(ni,  avg_equity)
    roa              = safe_div(ni,  avg_assets)

    # ROIC محافظ
    eff_tax_rate = tax_rate if (not pd.isna(tax_rate) and 0 <= tax_rate <= 0.6) else 0.25
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan
    invested_capital = np.nan
    if not pd.isna(total_debt) and not pd.isna(te):
        invested_capital = total_debt + te - (cash if not pd.isna(cash) else 0)
    roic = safe_div(nopat, invested_capital)

    # سيولة ومديونية
    current_ratio = safe_div(ca, cl)
    quick_ratio   = safe_div((ca - (inv if not pd.isna(inv) else 0)), cl)
    cash_ratio    = safe_div((cash if not pd.isna(cash) else 0) + (sti if not pd.isna(sti) else 0), cl)
    debt_to_equity = safe_div(total_debt, te)
    debt_to_assets = safe_div(total_debt, ta)
    interest_coverage = safe_div(ebit, int_exp_abs)

    # كفاءة
    asset_turnover = safe_div(rev, avg_assets)

    # FCF و Owner Earnings
    fcf = np.nan if (pd.isna(ocf) or pd.isna(capex_out)) else (ocf - capex_out)
    owner_earnings = np.nan
    if not pd.isna(ocf) and not pd.isna(capex_out):
        maint_capex = maint_capex_ratio * capex_out  # تقريب محافظ
        owner_earnings = ocf - maint_capex
    fcf_margin = safe_div(fcf, rev)
    ocf_to_ni  = safe_div(ocf, ni)

    # تقييمات سوقية
    price, shares, mcap = data.get("price", np.nan), data.get("shares", np.nan), data.get("mcap", np.nan)
    eps = safe_div(ni, shares)
    pe  = safe_div(price, eps)
    bvps = safe_div(te, shares)
    pb  = safe_div(price, bvps)
    sales_ps = safe_div(rev, shares)
    ps  = safe_div(price, sales_ps)
    fcf_yield = safe_div(fcf, mcap)
    earn_yield = safe_div(ni, mcap)
    ev = np.nan
    if not pd.isna(mcap):
        ev = mcap + (total_debt if not pd.isna(total_debt) else 0) - (cash if not pd.isna(cash) else 0)
    ev_ebit = safe_div(ev, ebit)

    # اتجاهات سنوية
    rev_cagr_5y, ni_cagr_5y = compute_cagr_5y(data["inc_a"])
    margin_std_5y, margin_trend_5y = margin_stability_trend(data["inc_a"])

    core = {
        # ربحية وجودة
        "الهامش الإجمالي": gross_margin,
        "هامش التشغيل": operating_margin,
        "هامش صافي الربح": net_margin,
        "ROA": roa,
        "ROE": roe,
        "ROIC~": roic,

        # سيولة/مديونية
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Cash Ratio": cash_ratio,
        "D/E": debt_to_equity,
        "D/A": debt_to_assets,
        "تغطية الفوائد": interest_coverage,

        # كفاءة مختصرة
        "دوران الأصول": asset_turnover,

        # تدفقات ونِسَب
        "هامش FCF": fcf_margin,
        "OCF/NI": ocf_to_ni,
        "FCF Yield": fcf_yield,
        "Earnings Yield": earn_yield,

        # تقييمات
        "P/E": pe,
        "P/B": pb,
        "P/S": ps,
        "EV/EBIT": ev_ebit
    }

    raw = {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "OperatingIncome": opi, "NetIncome": ni,
        "EBIT": ebit, "Tax": tax, "TaxRate": tax_rate,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "AR": ar, "AP": ap, "Cash": cash, "STInvest": sti,
        "TotalDebt": total_debt, "AvgAssets": avg_assets, "AvgEquity": avg_equity,
        "OCF": ocf, "CapexOut": capex_out, "FCF": fcf, "OwnerEarnings": owner_earnings,
        "Price": price, "Shares": shares, "MarketCap": mcap, "EV": ev
    }

    trends = {
        "Rev CAGR 5y": rev_cagr_5y,
        "NI CAGR 5y": ni_cagr_5y,
        "Gross Margin σ(5y)": margin_std_5y,
        "Gross Margin Trend(5y)": margin_trend_5y
    }

    checklist_inputs = {
        "shares_hist": data.get("shares_hist", pd.Series(dtype=float))
    }

    return core, raw, trends, checklist_inputs

# =============================
# تقييم محافظ (Buffett Score) + Checklist
# =============================

def buffett_checklist_and_score(core: dict, raw: dict, trends: dict,
                                moat_score: float, mgmt_score: float):
    """يرجع checklist dict و score (0-10)."""
    # قواعد محافظة
    moat_proxy = ( (core.get("ROIC~", np.nan) is not np.nan and core.get("ROIC~") >= 0.12) or
                   (core.get("الهامش الإجمالي", np.nan) is not np.nan and core.get("الهامش الإجمالي") >= 0.40) )
    prudent_leverage = (
        (core.get("D/E", np.nan) is not np.nan and core.get("D/E") <= 0.5) or
        (core.get("تغطية الفوائد", np.nan) is not np.nan and core.get("تغطية الفوائد") >= 8)
    )
    consistent_profitability = (
        (core.get("ROE", np.nan) is not np.nan and core.get("ROE") >= 0.15) and
        (core.get("هامش صافي الربح", np.nan) is not np.nan and core.get("هامش صافي الربح") > 0)
    )
    fcf_positive = (
        (core.get("هامش FCF", np.nan) is not np.nan and core.get("هامش FCF") > 0) and
        (core.get("OCF/NI", np.nan) is not np.nan and core.get("OCF/NI") >= 1)
    )
    buyback_trend = None
    sh = raw.get("Shares", np.nan)
    # إذا توفر تاريخ الأسهم: قارن آخر قيمة بأقدم قيمة خلال ~5 سنوات
    if "shares_hist" in trends or True:
        pass
    # نحاول داخل raw: لا يكفي. نستخدم trends-placeholder
    # سنعيد None إذا لا يوجد تاريخ
    # (ملاحظة: يُملأ عبر checklist_inputs خارج هذا الإجراء)

    score = 0
    score += 2 if (core.get("ROIC~", np.nan) is not np.nan and core.get("ROIC~") >= 0.12) else 0
    score += 2 if (core.get("ROE", np.nan) is not np.nan and core.get("ROE") >= 0.15) else 0
    score += 2 if prudent_leverage else 0
    score += 1 if (core.get("هامش FCF", np.nan) is not np.nan and core.get("هامش FCF") >= 0.05 and
                   core.get("FCF Yield", np.nan) is not np.nan and core.get("FCF Yield") >= 0.04) else 0
    score += 1 if (core.get("OCF/NI", np.nan) is not np.nan and core.get("OCF/NI") >= 1) else 0
    score += 1 if (trends.get("Rev CAGR 5y", np.nan) is not np.nan and trends["Rev CAGR 5y"] >= 0.05) else 0
    score += 1 if (trends.get("Gross Margin Trend(5y)", np.nan) is not np.nan and trends["Gross Margin Trend(5y)"] >= 0) else 0

    # تعديلات نوعية (±1 كحد أقصى لكل بُعد)
    if moat_score > 0.5: score += 1
    if moat_score < -0.5: score -= 1
    if mgmt_score > 0.5: score += 1
    if mgmt_score < -0.5: score -= 1

    score = max(0, min(10, score))
    checklist = {
        "خندق تنافسي (Proxy)": "✅" if moat_proxy else "⚠️",
        "رافعة متحفظة/تغطية فوائد": "✅" if prudent_leverage else "⚠️",
        "ربحية مستدامة": "✅" if consistent_profitability else "⚠️",
        "FCF إيجابي وجودته جيدة": "✅" if fcf_positive else "⚠️",
        "اتجاه هوامش موجب": "✅" if (trends.get("Gross Margin Trend(5y)", np.nan) is not np.nan and trends["Gross Margin Trend(5y)"] >= 0) else "⚠️",
        "نمو الإيرادات 5y ≥5%": "✅" if (trends.get("Rev CAGR 5y", np.nan) is not np.nan and trends["Rev CAGR 5y"] >= 0.05) else "⚠️",
        "مؤشر إعادة شراء أسهم": "—"  # يُحدّث لاحقًا إن توفر تاريخ
    }
    return checklist, score

def format_core_row(core: dict):
    view = {}
    as_pct = {"الهامش الإجمالي","هامش التشغيل","هامش صافي الربح","ROA","ROE","ROIC~","هامش FCF"}
    as_pct |= {"FCF Yield","Earnings Yield"}
    as_days = set()  # لا نستخدم أيام هنا
    as_mult = {"P/E","P/B","P/S","EV/EBIT"}
    ratios = {"Current Ratio","Quick Ratio","Cash Ratio","D/E","D/A","تغطية الفوائد","دوران الأصول","OCF/NI"}

    for k,v in core.items():
        if k in as_pct:
            view[k] = to_percent(v)
        elif k in as_mult or k in ratios:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}x"
        else:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}"
    return view

# =============================
# واجهة المستخدم
# =============================

st.title("📊 التحليل الأساسي | Buffett-Style Fundamentals")
st.caption("تركيز على القيمة الجوهرية: ربحية مستدامة، رافعة متحفظة، تدفقات نقدية حقيقية – بدون ضوضاء مضاربية.")

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1, help="TTM = مجموع 4 أرباع؛ Annual = آخر سنة مالية.")
    top_only = st.checkbox("عرض النِّسَب الأساسية فقط", value=True)
    show_raw = st.checkbox("إظهار البنود الخام", value=False)
    st.markdown("---")
    st.markdown("### 🧭 افتراضات نوعية (اختياري)")
    moat_score = st.slider("خندق تنافسي (–1 إلى +1)", -1.0, 1.0, 0.0, 0.1)
    mgmt_score = st.slider("جودة الإدارة (–1 إلى +1)", -1.0, 1.0, 0.0, 0.1)
    maint_capex_ratio = st.slider("٪ كابكس صيانة من CapEx", 0.4, 1.0, 0.7, 0.05, help="يؤثر على Owner Earnings (افتراضي محافظ 70%).")
    st.markdown("---")
    st.markdown("#### 🧪 رموز تجريبية")
    if st.button("USA: AAPL MSFT NVDA"): st.session_state.syms = "AAPL MSFT NVDA"
    if st.button("KSA: 1120 2380 1050"): st.session_state.syms = "1120 2380 1050"

symbols_input = st.text_area(
    "أدخل الرموز (مسافة/سطر). للسوق السعودي اختر السوق وسأضيف اللاحقة تلقائيًا.",
    st.session_state.get("syms","")
)

# تنظيف الرموز + منطق .SR المُحسَّن
raw_syms = [s.strip().upper() for s in symbols_input.replace("\n"," ").split() if s.strip()]
symbols = []
for s in raw_syms:
    if suffix == ".SR":
        if not s.endswith(".SR"): symbols.append(f"{s}.SR" if ".SR" not in s else s)
    else:
        symbols.append(s)
symbols = sorted(set(symbols))

if st.button("🚀 احسب"):
    if not symbols:
        st.warning("أدخل رمزًا واحدًا على الأقل.")
        st.stop()

    rows, raw_rows, score_rows, errors = [], [], [], []
    progress = st.progress(0, text=f"بدء الحساب... (0/{len(symbols)})")

    for i, code in enumerate(symbols, start=1):
        try:
            data = load_company_data(code)
            core, raw, trends, checklist_inputs = compute_ratios(data, mode=mode, maint_capex_ratio=maint_capex_ratio)
            if core is None:
                errors.append(code); continue

            # Checklist + Score
            checklist, score = buffett_checklist_and_score(core, raw, trends, moat_score, mgmt_score)

            # تحديث مؤشر إعادة شراء الأسهم إن توفر تاريخ
            sh_hist = checklist_inputs.get("shares_hist", pd.Series(dtype=float))
            if isinstance(sh_hist, pd.Series) and not sh_hist.empty:
                sh_hist = sh_hist.sort_index()
                first = float(sh_hist.iloc[0]); last = float(sh_hist.iloc[-1])
                buyback = (last - first)/first if first>0 else np.nan
                checklist["مؤشر إعادة شراء أسهم"] = "✅" if (not pd.isna(buyback) and buyback <= -0.01) else ("⚠️" if not pd.isna(buyback) else "—")

            view = format_core_row(core)
            row = {"الرمز": code}; row.update(view)
            rows.append(row)

            score_rows.append({
                "الرمز": code,
                "Buffett Score (0–10)": f"{score:.1f}",
                "Rev CAGR 5y": to_percent(trends.get("Rev CAGR 5y")),
                "NI CAGR 5y": to_percent(trends.get("NI CAGR 5y")),
                "ثبات الهامش σ(5y)": "—" if pd.isna(trends.get("Gross Margin σ(5y"))) else f"{trends.get('Gross Margin σ(5y'):.3f}",
                "اتجاه الهامش 5y": to_percent(trends.get("Gross Margin Trend(5y")))
            })

            if show_raw:
                rv = {"الرمز": code}
                for k,v in raw.items():
                    if k in ["Price","Shares","MarketCap","EV"]:
                        rv[k] = to_num(v, 2)
                    else:
                        rv[k] = to_num(v, 2)
                # نقاط نوعية و Owner Earnings/Yield
                oe = raw.get("OwnerEarnings", np.nan)
                mcap = raw.get("MarketCap", np.nan)
                rv["OwnerEarnings"] = to_num(oe)
                rv["Owner Earnings Yield"] = to_percent(safe_div(oe, mcap))
                # Checklist للشفافية
                rv.update({f"CHK:{k}": v for k,v in checklist.items()})
                raw_rows.append(rv)

        except Exception as e:
            errors.append(f"{code} → {e}")
        finally:
            progress.progress(i/len(symbols), text=f"تم {i}/{len(symbols)}")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader(f"نتائج النِّسَب ({mode}) — {len(df)} شركة")
        st.dataframe(df, use_container_width=True)

        # جدول النقاط والاتجاهات
        df_score = pd.DataFrame(score_rows)
        st.markdown("#### 🧮 نقاط الجودة والاتجاهات")
        st.dataframe(df_score, use_container_width=True)

        # تنزيلات
        html_out = html_table(df)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📥 تنزيل CSV", csv_bytes, file_name=f"fundamentals_{mode}.csv", mime="text/csv")
        with c2:
            st.download_button("📥 تنزيل HTML", html_out.encode("utf-8"), file_name=f"fundamentals_{mode}.html", mime="text/html")

    if show_raw and raw_rows:
        st.markdown("---")
        st.subheader("القيم الخام + شفافيات التقييم")
        df_raw = pd.DataFrame(raw_rows)
        st.dataframe(df_raw, use_container_width=True)

    if errors:
        st.info("⚠️ ملاحظات الرموز:")
        for e in errors: st.write("• ", e)

with st.expander("📌 منهجية وفرضيات"):
    st.markdown("""
- **CapEx** يُعامل كتدفق خارج (موجب) لضمان حساب **FCF=OCF−CapEx** بصورة صحيحة.  
- **ROIC~**: NOPAT≈EBIT×(1–الضريبة الفعّالة) على (الدين + حقوق المساهمين – النقد).  
- **Owner Earnings**: تقريبًا OCF − Maintenance CapEx (افتراضي 70% من CapEx؛ قابل للتعديل).  
- **CAGR 5y**: يُحسب من القوائم السنوية المتاحة (إذا صافي الدخل ≤0 في أي طرف، يُهمل CAGR للدخل).  
- **Buffett Score**: مزيج قواعد محافظة + مدخلات نوعية اختيارية؛ لا يُعد توصية استثمارية.  
- **السوق السعودي**: تُضاف اللاحقة `.SR` تلقائيًا عند اختيار السوق.
""")
