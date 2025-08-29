# === 📊 التحليل الأساسي | Buffett Score (نسخة مصححة) ===
import os
import re
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from html import escape
from zoneinfo import ZoneInfo

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="📊 التحليل الأساسي | Buffett Score", layout="wide")

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
  .buffett-table td.green { color: #16a34a; font-weight: bold; }
  .buffett-table td.yellow { color: #d97706; font-weight: bold; }
  .buffett-table td.red { color: #dc2626; font-weight: bold; }
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
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"

def to_num(x, digits=2):
    if x is None or pd.isna(x):
        return "—"
    absx = abs(x)
    if absx >= 1_000_000_000: return f"{x/1_000_000_000:.{digits}f}B"
    if absx >= 1_000_000:     return f"{x/1_000_000:.{digits}f}M"
    if absx >= 1_000:         return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

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

def sorted_cols(df: pd.DataFrame):
    try:
        return sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=True)
    except Exception:
        return list(df.columns)

def nansum(values):
    vals = [v for v in values if not pd.isna(v)]
    return np.nan if not vals else float(sum(vals))

# مرادفات البنود
REV_KEYS = ["Total Revenue", "Revenue", "TotalRevenue", "Sales"]
COGS_KEYS = ["Cost Of Revenue", "Cost of Revenue", "CostOfRevenue", "Cost Of Goods Sold", "COGS"]
GP_KEYS   = ["Gross Profit", "GrossProfit"]
OPINC_KEYS= ["Operating Income", "OperatingIncome", "EBIT"]
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

# =============================
# تحميل بيانات شركة من Yahoo
# =============================
@st.cache_data(ttl=3600)
def load_company_data(ticker: str):
    t = yf.Ticker(ticker)
    try:    inc_a = t.financials
    except: inc_a = pd.DataFrame()
    try:    inc_q = t.quarterly_financials
    except: inc_q = pd.DataFrame()
    try:    bal_a = t.balance_sheet
    except: bal_a = pd.DataFrame()
    try:    bal_q = t.quarterly_balance_sheet
    except: bal_q = pd.DataFrame()
    try:    cf_a  = t.cashflow
    except: cf_a  = pd.DataFrame()
    try:    cf_q  = t.quarterly_cashflow
    except: cf_q  = pd.DataFrame()

    price = np.nan; shares = np.nan
    try:
        fi = t.fast_info
        price  = float(fi.get("last_price", np.nan))
        shares = float(fi.get("shares", np.nan))
    except Exception:
        try:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            pass
        # أسهم بديلة من info
        try:
            info = t.info or {}
            if pd.isna(shares):
                shares = float(info.get("sharesOutstanding", np.nan))
        except Exception:
            pass

    # سلاسل 5 سنوات لـ EPS (تقريبًا عبر NI/shares الحالية) و FCF
    eps_5y, fcf_5y = [], []
    try:
        inc_cols = sorted_cols(inc_a)
        cf_cols  = sorted_cols(cf_a)
        for k in range(min(5, len(inc_cols))):
            col = inc_cols[k]
            ni = find_any(inc_a, NI_KEYS, col)
            eps = safe_div(ni, shares) if not pd.isna(shares) else np.nan
            if not pd.isna(eps): eps_5y.append(eps)
            if col in cf_cols:
                ocf   = find_any(cf_a, OCF_KEYS, col)
                capex = find_any(cf_a, CAPEX_KEYS, col)
                fcf   = ocf - capex if not (pd.isna(ocf) or pd.isna(capex)) else np.nan
                if not pd.isna(fcf): fcf_5y.append(fcf)
    except Exception:
        pass

    return {
        "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q, "cf_a": cf_a, "cf_q": cf_q,
        "price": price, "shares": shares, "info_ok": True, "eps_5y": eps_5y, "fcf_5y": fcf_5y
    }

# =============================
# Buffett Score
# =============================
def buffett_score(data: dict, mode: str = "TTM"):
    inc = data["inc_q"] if mode == "TTM" else data["inc_a"]
    bal = data["bal_q"] if mode == "TTM" else data["bal_a"]
    cf  = data["cf_q"]  if mode == "TTM" else data["cf_a"]
    quarterly = (mode == "TTM")

    if inc.empty or bal.empty:
        return None

    inc_cols = sorted_cols(inc)
    bal_cols = sorted_cols(bal)
    cf_cols  = sorted_cols(cf) if not cf.empty else []

    use_inc_cols = inc_cols[:4] if quarterly else inc_cols[:1]
    use_cf_cols  = cf_cols[:4] if (quarterly and cf_cols) else (cf_cols[:1] if cf_cols else [])

    bal_curr = bal_cols[0] if bal_cols else None
    bal_prev = bal_cols[1] if len(bal_cols) > 1 else None

    # تجميع قيَم أساسية مرة واحدة
    rev  = nansum([find_any(inc, REV_KEYS,  c) for c in use_inc_cols])
    opi  = nansum([find_any(inc, OPINC_KEYS, c) for c in use_inc_cols])
    ebit = nansum([find_any(inc, EBIT_KEYS, c) for c in use_inc_cols])
    if pd.isna(ebit) or ebit == 0: ebit = opi
    pbt  = nansum([find_any(inc, PBT_KEYS,  c) for c in use_inc_cols])
    tax  = nansum([find_any(inc, TAX_KEYS,  c) for c in use_inc_cols])
    ni   = nansum([find_any(inc, NI_KEYS,   c) for c in use_inc_cols])

    cogs = nansum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols])
    cogs_eff = cogs if not pd.isna(cogs) else rev  # fallback منطقي

    ocf   = nansum([find_any(cf, OCF_KEYS,   c) for c in use_cf_cols]) if use_cf_cols else np.nan
    capex = nansum([find_any(cf, CAPEX_KEYS, c) for c in use_cf_cols]) if use_cf_cols else np.nan
    fcf   = ocf - capex if not (pd.isna(ocf) or pd.isna(capex)) else np.nan

    tax_rate = safe_div(tax, pbt)
    eff_tax_rate = tax_rate if (not pd.isna(tax_rate) and 0 <= tax_rate <= 0.6) else 0.25
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan

    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_curr)
    te   = find_any(bal, TE_KEYS,  bal_curr)
    cash = find_any(bal, CASH_KEYS, bal_curr)
    sti  = find_any(bal, STI_KEYS,  bal_curr)
    if pd.isna(total_debt):
        ltd  = find_any(bal, LTD_KEYS,  bal_curr)
        sltd = find_any(bal, SLTD_KEYS, bal_curr)
        cdebt= find_any(bal, CUR_DEBT_KEYS, bal_curr)
        parts = [x for x in [ltd, sltd, cdebt] if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    invested_capital = (total_debt + te - (0 if pd.isna(cash) else cash)) if not (pd.isna(total_debt) or pd.isna(te)) else np.nan
    roic = safe_div(nopat, invested_capital)

    # CCC (اعتماد متوسط الرصيد الحالي والسابق)
    ar_cur = find_any(bal, AR_KEYS,  bal_curr); ar_prev = find_any(bal, AR_KEYS,  bal_prev)
    inv_cur= find_any(bal, INV_KEYS, bal_curr); inv_prev= find_any(bal, INV_KEYS, bal_prev)
    ap_cur = find_any(bal, AP_KEYS,  bal_curr); ap_prev = find_any(bal, AP_KEYS,  bal_prev)
    avg_ar  = np.nanmean([ar_cur,  ar_prev])  if not pd.isna(ar_cur)  else np.nan
    avg_inv = np.nanmean([inv_cur, inv_prev]) if not pd.isna(inv_cur) else np.nan
    avg_ap  = np.nanmean([ap_cur,  ap_prev])  if not pd.isna(ap_cur)  else np.nan

    receivables_turnover = safe_div(rev, avg_ar)
    inventory_turnover   = safe_div(cogs_eff, avg_inv)
    payables_turnover    = safe_div(cogs_eff, avg_ap)
    dso = safe_div(365, receivables_turnover)
    dio = safe_div(365, inventory_turnover)
    dpo = safe_div(365, payables_turnover)
    ccc = dso + dio - dpo if not (pd.isna(dso) or pd.isna(dio) or pd.isna(dpo)) else np.nan

    # صافي الدين و تغطية الفوائد
    net_debt = np.nan
    if not pd.isna(total_debt):
        cash_like = 0.0
        if not pd.isna(cash): cash_like += cash
        if not pd.isna(sti):  cash_like += sti
        net_debt = total_debt - cash_like

    int_exp_vals = [find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols]
    int_exp = nansum(int_exp_vals)
    if not pd.isna(int_exp): int_exp = abs(float(int_exp))
    interest_coverage = safe_div(ebit, int_exp)

    # مؤشرات القرار
    op_margin = safe_div(opi, rev)
    fcf_margin = safe_div(fcf, rev)
    ocf_ni = safe_div(ocf, ni)

    score = 0
    results = {}

    # 1) الهوامش
    margin_stability = True  # تبسيط: يمكن توسيعها لاحقاً عبر 5Y
    if not pd.isna(op_margin) and op_margin > 0.20 and margin_stability:
        results['الهوامش'] = '✅'; score += 2
    else:
        results['الهوامش'] = '⚠️'; score += 1

    # 2) ROIC
    if not pd.isna(roic) and roic > 0.20:
        results['ROIC'] = '✅'; score += 2
    else:
        results['ROIC'] = '⚠️'; score += 1

    # 3) تحويل النقد OCF/NI
    if not pd.isna(ocf_ni) and ocf_ni > 1.0:
        results['تحويل النقد OCF/NI'] = '✅'; score += 2
    else:
        results['تحويل النقد OCF/NI'] = '⚠️'; score += 1

    # 4) هامش التدفق الحر
    if not pd.isna(fcf_margin) and fcf_margin > 0.10:
        results['هامش التدفق الحر'] = '✅'; score += 2
    else:
        results['هامش التدفق الحر'] = '⚠️'; score += 1

    # 5) CCC
    if not pd.isna(ccc) and ccc <= 0:
        results['دورة التحويل النقدي CCC'] = '✅'; score += 2
    else:
        results['دورة التحويل النقدي CCC'] = '⚠️'; score += 1

    # 6) صافي الدين
    if not pd.isna(net_debt) and net_debt <= 0:
        results['صافي الدين'] = '✅'; score += 2
    else:
        results['صافي الدين'] = '⚠️'; score += 1

    # 7) تغطية الفوائد
    if not pd.isna(interest_coverage) and interest_coverage > 10:
        results['تغطية الفوائد'] = '✅'; score += 2
    else:
        results['تغطية الفوائد'] = '⚠️'; score += 1

    # 8) نمو EPS/FCF (CAGR صحيح: last/first)
    def cagr(series):
        if not series or len(series) < 2: return np.nan
        first, last = series[-1], series[0]
        if first is None or last is None or pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
            return np.nan
        n = len(series) - 1
        return (last / first) ** (1/n) - 1

    eps_growth = cagr(data.get('eps_5y', []))
    fcf_growth = cagr(data.get('fcf_5y', []))
    if not pd.isna(eps_growth) and not pd.isna(fcf_growth) and eps_growth > 0 and fcf_growth > 0:
        results['نمو EPS/FCF'] = '✅'; score += 2
    else:
        results['نمو EPS/FCF'] = '⚠️'; score += 1

    # 9) عائد المالك (Owner Earnings Yield)
    market_cap = data['price'] * data['shares'] if not (pd.isna(data['price']) or pd.isna(data['shares'])) else np.nan
    oey = safe_div(fcf, market_cap)
    if not pd.isna(oey) and oey >= 0.08:
        results['عائد المالك'] = '✅'; score += 2
    else:
        results['عائد المالك'] = '⚠️'; score += 1

    final_score = (score / 18) * 100
    recommendation = 'Buy' if final_score >= 80 and results['عائد المالك'] == '✅' else ('Hold' if final_score >= 60 else 'Wait')

    return {
        'score': final_score,
        'results': results,
        'recommendation': recommendation,
        'metrics': {
            'roic': roic, 'ocf_ni': ocf_ni, 'fcf_margin': fcf_margin, 'ccc': ccc,
            'net_debt': net_debt, 'interest_coverage': interest_coverage, 'oey': oey
        }
    }

# =============================
# حساب النسب — سنوي أو TTM (كما هي مع إصلاحات طفيفة)
# =============================
def compute_ratios(data: dict, mode: str = "Annual"):
    inc = data["inc_a"]; bal = data["bal_a"]; cf = data["cf_a"]; quarterly = False
    if mode == "TTM" and not data["inc_q"].empty:
        inc = data["inc_q"].copy()
        bal = data["bal_q"] if not data["bal_q"].empty else data["bal_a"]
        cf  = data["cf_q"]  if not data["cf_q"].empty  else data["cf_a"]
        quarterly = True
    if inc is None or inc.empty or bal is None or bal.empty:
        return None, None

    inc_cols = sorted_cols(inc); bal_cols = sorted_cols(bal)
    cf_cols  = sorted_cols(cf) if cf is not None and not cf.empty else []
    use_inc_cols = inc_cols[:4] if quarterly else inc_cols[:1]
    use_cf_cols  = cf_cols[:4] if (quarterly and cf_cols) else (cf_cols[:1] if cf_cols else [])

    rev  = nansum([find_any(inc, REV_KEYS,  c) for c in use_inc_cols])
    cogs = nansum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols])
    gp   = nansum([find_any(inc, GP_KEYS,   c) for c in use_inc_cols])
    opi  = nansum([find_any(inc, OPINC_KEYS,c) for c in use_inc_cols])
    ni   = nansum([find_any(inc, NI_KEYS,   c) for c in use_inc_cols])
    pbt  = nansum([find_any(inc, PBT_KEYS,  c) for c in use_inc_cols])
    tax  = nansum([find_any(inc, TAX_KEYS,  c) for c in use_inc_cols])
    tax_rate = safe_div(tax, pbt)
    ebit = nansum([find_any(inc, EBIT_KEYS, c) for c in use_inc_cols])
    if pd.isna(ebit) or ebit == 0: ebit = opi

    bal_curr = bal_cols[0] if bal_cols else None
    bal_prev = bal_cols[1] if len(bal_cols) > 1 else None

    ta   = find_any(bal, TA_KEYS,  bal_curr)
    te   = find_any(bal, TE_KEYS,  bal_curr)
    ca   = find_any(bal, CA_KEYS,  bal_curr)
    cl   = find_any(bal, CL_KEYS,  bal_curr)
    inv  = find_any(bal, INV_KEYS, bal_curr)
    ar   = find_any(bal, AR_KEYS,  bal_curr)
    ap   = find_any(bal, AP_KEYS,  bal_curr)
    cash = find_any(bal, CASH_KEYS, bal_curr)
    sti  = find_any(bal, STI_KEYS,  bal_curr)

    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_curr)
    if pd.isna(total_debt) or total_debt == 0:
        ltd  = find_any(bal, LTD_KEYS,  bal_curr)
        sltd = find_any(bal, SLTD_KEYS, bal_curr)
        cdebt= find_any(bal, CUR_DEBT_KEYS, bal_curr)
        parts = [x for x in [ltd, sltd, cdebt] if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan

    ta_prev = find_any(bal, TA_KEYS,  bal_prev) if bal_prev else np.nan
    te_prev = find_any(bal, TE_KEYS,  bal_prev) if bal_prev else np.nan
    inv_prev= find_any(bal, INV_KEYS, bal_prev) if bal_prev else np.nan
    ar_prev = find_any(bal, AR_KEYS,  bal_prev) if bal_prev else np.nan
    ap_prev = find_any(bal, AP_KEYS,  bal_prev) if bal_prev else np.nan

    avg_assets = np.nanmean([ta, ta_prev]) if not pd.isna(ta) else np.nan
    avg_equity = np.nanmean([te, te_prev]) if not pd.isna(te) else np.nan
    avg_inv    = np.nanmean([inv, inv_prev]) if not pd.isna(inv) else np.nan
    avg_ar     = np.nanmean([ar,  ar_prev])  if not pd.isna(ar)  else np.nan
    avg_ap     = np.nanmean([ap,  ap_prev])  if not pd.isna(ap)  else np.nan

    if cf is not None and not cf.empty and use_cf_cols:
        ocf   = nansum([find_any(cf, OCF_KEYS,   c) for c in use_cf_cols])
        capex = nansum([find_any(cf, CAPEX_KEYS, c) for c in use_cf_cols])
    else:
        ocf = capex = np.nan

    int_exp_vals = [find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols]
    int_exp = nansum(int_exp_vals)
    int_exp_abs = abs(int_exp) if not pd.isna(int_exp) else np.nan

    gross_margin     = safe_div(gp,  rev)
    operating_margin = safe_div(opi, rev)
    net_margin       = safe_div(ni,  rev)
    roe = safe_div(ni, avg_equity)
    roa = safe_div(ni, avg_assets)

    eff_tax_rate = tax_rate if (not pd.isna(tax_rate) and 0 <= tax_rate <= 0.6) else 0.25
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan
    invested_capital = (total_debt + te - (0 if pd.isna(cash) else cash)) if not (pd.isna(total_debt) or pd.isna(te)) else np.nan
    roic = safe_div(nopat, invested_capital)

    current_ratio = safe_div(ca, cl)
    quick_ratio   = safe_div((ca - (0 if pd.isna(inv) else inv)), cl)
    cash_ratio    = safe_div((0 if pd.isna(cash) else cash) + (0 if pd.isna(sti) else sti), cl)

    debt_to_equity  = safe_div(total_debt, te)
    debt_to_assets  = safe_div(total_debt, ta)
    interest_coverage = safe_div(ebit, int_exp_abs)

    asset_turnover       = safe_div(rev,       avg_assets)
    inventory_turnover   = safe_div((cogs if not pd.isna(cogs) else rev), avg_inv)
    receivables_turnover = safe_div(rev,       avg_ar)
    payables_turnover    = safe_div((cogs if not pd.isna(cogs) else rev), avg_ap)

    dso = safe_div(365, receivables_turnover)
    dio = safe_div(365, inventory_turnover)
    dpo = safe_div(365, payables_turnover)
    ccc = dso + dio - dpo if not (pd.isna(dso) or pd.isna(dio) or pd.isna(dpo)) else np.nan

    fcf = ocf - capex if not (pd.isna(ocf) or pd.isna(capex)) else np.nan
    ocf_to_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(fcf, rev)

    price = data.get("price", np.nan); shares = data.get("shares", np.nan)
    pe = pb = ps = np.nan
    if not (pd.isna(price) or pd.isna(shares) or shares == 0):
        eps  = safe_div(ni, shares)
        pe   = safe_div(price, eps)
        sales_ps = safe_div(rev, shares)
        ps   = safe_div(price, sales_ps)
        bvps = safe_div(te, shares)
        pb   = safe_div(price, bvps)

    core = {
        "الهامش الإجمالي": gross_margin,
        "هامش التشغيل": operating_margin,
        "هامش صافي الربح": net_margin,
        "العائد على الأصول ROA": roa,
        "العائد على حقوق الملكية ROE": roe,
        "العائد على رأس المال المستثمر ROIC~": roic,
        "نسبة التداول Current": current_ratio,
        "نسبة السريعة Quick": quick_ratio,
        "نسبة النقد Cash": cash_ratio,
        "الدين/الحقوق D/E": debt_to_equity,
        "الدين/الأصول D/A": debt_to_assets,
        "تغطية الفوائد": interest_coverage,
        "دوران الأصول": asset_turnover,
        "دوران المخزون": inventory_turnover,
        "دوران الذمم المدينة": receivables_turnover,
        "دوران الدائنين": payables_turnover,
        "أيام التحصيل DSO": dso,
        "أيام المخزون DIO": dio,
        "أيام السداد DPO": dpo,
        "دورة التحويل النقدي CCC": ccc,
        "هامش التدفق التشغيلي OCF/NI": ocf_to_ni,
        "هامش التدفق الحر FCF Margin": fcf_margin,
        "P/E (اختياري)": pe,
        "P/B (اختياري)": pb,
        "P/S (اختياري)": ps
    }
    raw = {
        "Revenue": rev, "COGS": cogs, "GrossProfit": gp, "OperatingIncome": opi, "NetIncome": ni,
        "EBIT": ebit, "Tax": tax, "TaxRate": tax_rate,
        "TotalAssets": ta, "TotalEquity": te, "CurrentAssets": ca, "CurrentLiabilities": cl,
        "Inventory": inv, "AR": ar, "AP": ap, "Cash": cash, "STInvest": sti,
        "TotalDebt": total_debt,
        "AvgAssets": avg_assets, "AvgEquity": avg_equity, "AvgInv": avg_inv, "AvgAR": avg_ar, "AvgAP": avg_ap,
        "OCF": ocf, "Capex": capex, "FCF": fcf,
        "Price": price, "Shares": shares
    }
    return core, raw

def format_core_row(core: dict):
    view = {}
    percent_keys = [
        "الهامش الإجمالي","هامش التشغيل","هامش صافي الربح",
        "العائد على الأصول ROA","العائد على حقوق الملكية ROE","العائد على رأس المال المستثمر ROIC~",
        "هامش التدفق التشغيلي OCF/NI","هامش التدفق الحر FCF Margin"
    ]
    ratio_keys = ["نسبة التداول Current","نسبة السريعة Quick","نسبة النقد Cash","الدين/الحقوق D/E","الدين/الأصول D/A","تغطية الفوائد","دوران الأصول","دوران المخزون","دوران الذمم المدينة","دوران الدائنين"]
    days_keys  = ["أيام التحصيل DSO","أيام المخزون DIO","أيام السداد DPO","دورة التحويل النقدي CCC"]
    val_keys   = ["P/E (اختياري)","P/B (اختياري)","P/S (اختياري)"]
    for k,v in core.items():
        if k in percent_keys: view[k] = to_percent(v)
        elif k in days_keys:  view[k] = "—" if v is None or pd.isna(v) else f"{v:.1f} يوم"
        elif k in val_keys:   view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}x"
        elif k in ratio_keys: view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}x"
        else:                 view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}"
    return view

def generate_html_table(df: pd.DataFrame, buffett=False) -> str:
    html = """
    <style>
    table.buffett-table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
    .buffett-table th, .buffett-table td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    .buffett-table th {background-color: #0ea5e9; color: white;}
    .buffett-table tr:nth-child(even){background-color: #f8fafc;}
    .buffett-table tr:hover {background-color: #eef2ff;}
    .buffett-table td.green { color: #16a34a; font-weight: bold; }
    .buffett-table td.yellow { color: #d97706; font-weight: bold; }
    .buffett-table td.red { color: #dc2626; font-weight: bold; }
    </style>
    <table class="buffett-table">
    <thead><tr>"""
    for col in df.columns:
        html += f"<th>{escape(col)}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            cls = ""
            if buffett and col not in ("الرمز","الدرجة","التوصية"):
                cls = "green" if row[col] == "✅" else ("yellow" if row[col] == "⚠️" else ("red" if row[col] == "❌" else ""))
            html += f"<td class='{cls}'>{escape(str(row[col]))}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# واجهة المستخدم
# =============================
st.title("📊 التحليل الأساسي | Buffett Score")
st.caption("إطار وارن بوفيت: جودة الأعمال، القوة المالية، تحويل النقد، اقتصاديات التشغيل، التقييم.")

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1, help="TTM = مجموع 4 أرباع أخيرة؛ Annual = آخر سنة مالية.")
    show_buffett = st.checkbox("عرض Buffett Score", value=True)
    top_only = st.checkbox("عرض النِّسَب الأساسية فقط (جدول مختصر)", value=True)
    show_raw = st.checkbox("إظهار القيم الخام (Revenue/Assets/…)", value=False)
    st.markdown("---")
    st.markdown("#### 🧪 رموز تجريبية")
    if st.button("USA: AAPL MSFT NVDA"): st.session_state.syms = "AAPL MSFT NVDA"
    if st.button("KSA: 1120 2380 1050"): st.session_state.syms = "1120 2380 1050"

symbols_input = st.text_area(
    "أدخل الرموز (مسافة/سطر). للسوق السعودي ستُضاف اللاحقة .SR تلقائيًا.",
    st.session_state.get("syms","")
)

# تنظيف الرموز + إضافة .SR للأرقام والحروف ما لم تكن موجودة
raw = [s.strip().upper() for s in symbols_input.replace("\n"," ").split() if s.strip()]
clean = []
for s in raw:
    if suffix and not s.endswith(suffix):
        clean.append(s + suffix)
    else:
        clean.append(s)
symbols = sorted(set(clean))

if st.button("🚀 احسب"):
    if not symbols:
        st.warning("أدخل رمزًا واحدًا على الأقل.")
        st.stop()

    rows, raw_rows, buffett_rows, errors = [], [], [], []
    progress = st.progress(0, text=f"بدء الحساب... (0/{len(symbols)})")

    for i, code in enumerate(symbols, start=1):
        try:
            data = load_company_data(code)
            core, rawvals = compute_ratios(data, mode=mode)
            if core is None:
                errors.append(code); continue

            view = format_core_row(core)
            row = {"الرمز": code}; row.update(view); rows.append(row)

            if show_raw:
                rv = {"الرمز": code}
                for k,v in rawvals.items(): rv[k] = to_num(v, 2)
                raw_rows.append(rv)

            if show_buffett:
                bf = buffett_score(data, mode=mode)
                if bf:
                    bf_row = {"الرمز": code, "الدرجة": f"{bf['score']:.1f}/100"}
                    bf_row.update(bf['results'])
                    bf_row["التوصية"] = bf['recommendation']
                    buffett_rows.append(bf_row)

        except Exception as e:
            errors.append(f"{code} → {e}")
        finally:
            progress.progress(i/len(symbols), text=f"تم {i}/{len(symbols)}")

    if buffett_rows and show_buffett:
        st.subheader(f"Buffett Score ({mode}) — {len(buffett_rows)} شركة")
        df_buffett = pd.DataFrame(buffett_rows)
        st.markdown(generate_html_table(df_buffett, buffett=True), unsafe_allow_html=True)
        st.download_button("📥 تنزيل Buffett Score CSV", df_buffett.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"buffett_score_{mode}.csv", mime="text/csv")

    if rows:
        st.subheader(f"النِّسَب الأساسية ({mode}) — {len(rows)} شركة")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        html_out = generate_html_table(df)
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📥 تنزيل CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"fundamentals_{mode}.csv", mime="text/csv")
        with c2:
            st.download_button("📥 تنزيل HTML", html_out.encode("utf-8"),
                               file_name=f"fundamentals_{mode}.html", mime="text/html")

    if show_raw and raw_rows:
        st.markdown("---")
        st.subheader("القيم الخام (للتدقيق)")
        st.dataframe(pd.DataFrame(raw_rows), use_container_width=True)

    if errors:
        st.info("⚠️ ملاحظات:")
        for e in errors:
            st.write("• ", e)

# ملاحظات
with st.expander("📌 منهجية الحساب"):
    st.markdown("""
- **TTM**: مجموع آخر 4 أرباع للدخل/التدفقات، وأحدث ميزانية للأرصدة، ومتوسط (الحالية + السابقة) للأصول/الحقوق/AR/INV/AP.
- **Buffett Score**: 9 معايير (2 نقاط لكل معيار = 18 نقطة = 100%). الهوامش>20%، ROIC>20%، OCF/NI>1، FCF Margin>10%، CCC≤0، Net Debt≤0، تغطية الفوائد>10x، نمو EPS/FCF موجب، عائد المالك ≥8%.
- **CAGR** تم تصحيحه إلى (الآخر/الأول)^(1/n)-1 ويُتجاهل إذا كان أحد الطرفين ≤0.
- **COGS fallback** مصحح: نستخدم الإيراد بدلاً من معاملات منطقية خاطئة.
- **الفوائد**: نأخذ القيمة المطلقة لـ Interest Expense عند حساب التغطية.
- **السعودي**: تُضاف `.SR` تلقائيًا إن لم تكن موجودة.
""")
