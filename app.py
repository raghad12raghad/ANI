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
  .buffett-table td.green { color: green; font-weight: bold; }
  .buffett-table td.yellow { color: orange; font-weight: bold; }
  .buffett-table td.red { color: red; font-weight: bold; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# =============================
# أدوات مساعدة مشتركة
# =============================
def pct(x):
    return np.nan if x is None or pd.isna(x) else x

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
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.{digits}f}B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.{digits}f}M"
    if absx >= 1_000:
        return f"{x/1_000:.{digits}f}K"
    return f"{x:.{digits}f}"

def normalize_idx(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def build_index_map(df: pd.DataFrame):
    idx = {}
    for raw in df.index.astype(str):
        idx[normalize_idx(raw)] = raw
    return idx

def find_any(df: pd.DataFrame, keys: list[str], col):
    if df is None or df.empty: 
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
        cols = sorted(list(df.columns), key=lambda x: pd.to_datetime(str(x)), reverse=True)
        return cols
    except Exception:
        return list(df.columns)

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
EPS_KEYS = ["Diluted EPS", "EPS"]

# =============================
# تحميل بيانات شركة من Yahoo
# =============================
@st.cache_data(ttl=3600)
def load_company_data(ticker: str):
    t = yf.Ticker(ticker)
    try:
        inc_a = t.financials
    except Exception:
        inc_a = pd.DataFrame()
    try:
        inc_q = t.quarterly_financials
    except Exception:
        inc_q = pd.DataFrame()
    try:
        bal_a = t.balance_sheet
    except Exception:
        bal_a = pd.DataFrame()
    try:
        bal_q = t.quarterly_balance_sheet
    except Exception:
        bal_q = pd.DataFrame()
    try:
        cf_a  = t.cashflow
    except Exception:
        cf_a = pd.DataFrame()
    try:
        cf_q  = t.quarterly_cashflow
    except Exception:
        cf_q = pd.DataFrame()
    price = np.nan
    shares = np.nan
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        shares = float(fi.get("shares", np.nan))
    except Exception:
        try:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    # نمو EPS/FCF لـ 5 سنوات
    eps_5y = []
    fcf_5y = []
    try:
        for year in range(5):
            col = sorted_cols(inc_a)[year] if year < len(sorted_cols(inc_a)) else None
            if col:
                ni = find_any(inc_a, NI_KEYS, col)
                sh = find_any(bal_a, TE_KEYS, col) / find_any(bal_a, TE_KEYS, col) * shares if not pd.isna(shares) else np.nan
                eps = safe_div(ni, sh)
                if not pd.isna(eps):
                    eps_5y.append(eps)
                fcf = find_any(cf_a, OCF_KEYS, col) - find_any(cf_a, CAPEX_KEYS, col) if col in cf_a.columns else np.nan
                if not pd.isna(fcf):
                    fcf_5y.append(fcf)
    except Exception:
        pass
    return {
        "inc_a": inc_a, "inc_q": inc_q, "bal_a": bal_a, "bal_q": bal_q, "cf_a": cf_a, "cf_q": cf_q,
        "price": price, "shares": shares, "info_ok": True, "eps_5y": eps_5y, "fcf_5y": fcf_5y
    }

# =============================
# حساب Buffett Score
# =============================
def buffett_score(data: dict, mode: str = "TTM"):
    inc = data["inc_q"] if mode == "TTM" else data["inc_a"]
    bal = data["bal_q"] if mode == "TTM" else data["bal_a"]
    cf = data["cf_q"] if mode == "TTM" else data["cf_a"]
    quarterly = mode == "TTM"
    if inc.empty or bal.empty:
        return None
    inc_cols = sorted_cols(inc)
    bal_cols = sorted_cols(bal)
    cf_cols = sorted_cols(cf) if not cf.empty else []
    use_inc_cols = inc_cols[:4] if quarterly else inc_cols[:1]
    use_cf_cols = cf_cols[:4] if quarterly and cf_cols else cf_cols[:1] if cf_cols else []
    bal_curr = bal_cols[0] if bal_cols else None
    score = 0
    results = {}
    # 1. اتساق الهوامش
    op_margin = sum([find_any(inc, OPINC_KEYS, c) for c in use_inc_cols]) / sum([find_any(inc, REV_KEYS, c) for c in use_inc_cols])
    margin_stability = True  # تحتاج بيانات 5 سنوات للتحقق
    if not pd.isna(op_margin) and op_margin > 0.2 and margin_stability:
        results['الهوامش'] = '✅'
        score += 2
    else:
        results['الهوامش'] = '⚠️'
        score += 1
    # 2. ROIC
    ebit = sum([find_any(inc, EBIT_KEYS, c) for c in use_inc_cols])
    tax_rate = safe_div(sum([find_any(inc, TAX_KEYS, c) for c in use_inc_cols]), sum([find_any(inc, PBT_KEYS, c) for c in use_inc_cols]))
    eff_tax_rate = tax_rate if not pd.isna(tax_rate) and 0 <= tax_rate <= 0.6 else 0.25
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan
    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_curr)
    te = find_any(bal, TE_KEYS, bal_curr)
    cash = find_any(bal, CASH_KEYS, bal_curr)
    invested_capital = total_debt + te - (cash if not pd.isna(cash) else 0) if not (pd.isna(total_debt) or pd.isna(te)) else np.nan
    roic = safe_div(nopat, invested_capital)
    if not pd.isna(roic) and roic > 0.2:
        results['ROIC'] = '✅'
        score += 2
    else:
        results['ROIC'] = '⚠️'
        score += 1
    # 3. OCF/NI
    ocf = sum([find_any(cf, OCF_KEYS, c) for c in use_cf_cols]) if use_cf_cols else np.nan
    ni = sum([find_any(inc, NI_KEYS, c) for c in use_inc_cols])
    ocf_ni = safe_div(ocf, ni)
    if not pd.isna(ocf_ni) and ocf_ni > 1:
        results['تحويل النقد OCF/NI'] = '✅'
        score += 2
    else:
        results['تحويل النقد OCF/NI'] = '⚠️'
        score += 1
    # 4. FCF Margin
    capex = sum([find_any(cf, CAPEX_KEYS, c) for c in use_cf_cols]) if use_cf_cols else np.nan
    fcf = ocf - capex if not (pd.isna(ocf) or pd.isna(capex)) else np.nan
    rev = sum([find_any(inc, REV_KEYS, c) for c in use_inc_cols])
    fcf_margin = safe_div(fcf, rev)
    if not pd.isna(fcf_margin) and fcf_margin > 0.1:
        results['هامش التدفق الحر'] = '✅'
        score += 2
    else:
        results['هامش التدفق الحر'] = '⚠️'
        score += 1
    # 5. CCC
    dso = safe_div(365, safe_div(rev, np.nanmean([find_any(bal, AR_KEYS, bal_curr), find_any(bal, AR_KEYS, bal_cols[1] if len(bal_cols) > 1 else None)])))
    dio = safe_div(365, safe_div(sum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols]) or rev, np.nanmean([find_any(bal, INV_KEYS, bal_curr), find_any(bal, INV_KEYS, bal_cols[1] if len(bal_cols) > 1 else None)])))
    dpo = safe_div(365, safe_div(sum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols]) or rev, np.nanmean([find_any(bal, AP_KEYS, bal_curr), find_any(bal, AP_KEYS, bal_cols[1] if len(bal_cols) > 1 else None)])))
    ccc = dso + dio - dpo if not (pd.isna(dso) or pd.isna(dio) or pd.isna(dpo)) else np.nan
    if not pd.isna(ccc) and ccc <= 0:
        results['دورة التحويل النقدي CCC'] = '✅'
        score += 2
    else:
        results['دورة التحويل النقدي CCC'] = '⚠️'
        score += 1
    # 6. Net Debt
    sti = find_any(bal, STI_KEYS, bal_curr)
    net_debt = total_debt - (cash + sti if not (pd.isna(cash) or pd.isna(sti)) else cash if not pd.isna(cash) else 0) if not pd.isna(total_debt) else np.nan
    if not pd.isna(net_debt) and net_debt <= 0:
        results['صافي الدين'] = '✅'
        score += 2
    else:
        results['صافي الدين'] = '⚠️'
        score += 1
    # 7. Interest Coverage
    int_exp = abs(sum([find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols])) if sum([find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols]) else np.nan
    interest_coverage = safe_div(ebit, int_exp)
    if not pd.isna(interest_coverage) and interest_coverage > 10:
        results['تغطية الفوائد'] = '✅'
        score += 2
    else:
        results['تغطية الفوائد'] = '⚠️'
        score += 1
    # 8. نمو EPS/FCF
    eps_growth = np.nan
    fcf_growth = np.nan
    if data['eps_5y'] and len(data['eps_5y']) >= 2:
        eps_growth = (data['eps_5y'][0] / data['eps_5y'][-1]) ** (1/(len(data['eps_5y'])-1)) - 1 if data['eps_5y'][-1] != 0 else np.nan
    if data['fcf_5y'] and len(data['fcf_5y']) >= 2:
        fcf_growth = (data['fcf_5y'][0] / data['fcf_5y'][-1]) ** (1/(len(data['fcf_5y'])-1)) - 1 if data['fcf_5y'][-1] != 0 else np.nan
    if not (pd.isna(eps_growth) or pd.isna(fcf_growth)) and eps_growth > 0 and fcf_growth > 0:
        results['نمو EPS/FCF'] = '✅'
        score += 2
    else:
        results['نمو EPS/FCF'] = '⚠️'
        score += 1
    # 9. Owner Earnings Yield
    owner_earnings = fcf
    market_cap = data['price'] * data['shares'] if not (pd.isna(data['price']) or pd.isna(data['shares'])) else np.nan
    oey = safe_div(owner_earnings, market_cap)
    if not pd.isna(oey) and oey >= 0.08:
        results['عائد المالك'] = '✅'
        score += 2
    else:
        results['عائد المالك'] = '⚠️'
        score += 1
    final_score = (score / 18) * 100
    recommendation = 'Buy' if final_score >= 80 and results['عائد المالك'] == '✅' else 'Hold' if final_score >= 60 else 'Wait'
    return {
        'score': final_score,
        'results': results,
        'recommendation': recommendation,
        'metrics': {'roic': roic, 'ocf_ni': ocf_ni, 'fcf_margin': fcf_margin, 'ccc': ccc, 'net_debt': net_debt, 'interest_coverage': interest_coverage, 'oey': oey}
    }

# =============================
# حساب النسب — سنوي أو TTM
# =============================
def compute_ratios(data: dict, mode: str = "Annual"):
    inc = data["inc_a"]
    bal = data["bal_a"]
    cf = data["cf_a"]
    quarterly = False
    if mode == "TTM" and not data["inc_q"].empty:
        inc = data["inc_q"].copy()
        bal = data["bal_q"] if not data["bal_q"].empty else data["bal_a"]
        cf = data["cf_q"] if not data["cf_q"].empty else data["cf_a"]
        quarterly = True
    if inc is None or inc.empty or bal is None or bal.empty:
        return None, None
    inc_cols = sorted_cols(inc)
    bal_cols = sorted_cols(bal)
    cf_cols = sorted_cols(cf) if cf is not None and not cf.empty else []
    if quarterly:
        use_inc_cols = inc_cols[:4]
        use_cf_cols = cf_cols[:4] if cf_cols else []
    else:
        use_inc_cols = inc_cols[:1]
        use_cf_cols = cf_cols[:1] if cf_cols else []
    rev = sum([find_any(inc, REV_KEYS, c) for c in use_inc_cols])
    cogs = sum([find_any(inc, COGS_KEYS, c) for c in use_inc_cols])
    gp = sum([find_any(inc, GP_KEYS, c) for c in use_inc_cols])
    opi = sum([find_any(inc, OPINC_KEYS, c) for c in use_inc_cols])
    ni = sum([find_any(inc, NI_KEYS, c) for c in use_inc_cols])
    pbt = sum([find_any(inc, PBT_KEYS, c) for c in use_inc_cols])
    tax = sum([find_any(inc, TAX_KEYS, c) for c in use_inc_cols])
    tax_rate = safe_div(tax, pbt)
    ebit = sum([find_any(inc, EBIT_KEYS, c) for c in use_inc_cols])
    if pd.isna(ebit) or ebit == 0:
        ebit = opi
    bal_curr = bal_cols[0] if bal_cols else None
    bal_prev = bal_cols[1] if len(bal_cols) > 1 else None
    ta = find_any(bal, TA_KEYS, bal_curr)
    te = find_any(bal, TE_KEYS, bal_curr)
    ca = find_any(bal, CA_KEYS, bal_curr)
    cl = find_any(bal, CL_KEYS, bal_curr)
    inv = find_any(bal, INV_KEYS, bal_curr)
    ar = find_any(bal, AR_KEYS, bal_curr)
    ap = find_any(bal, AP_KEYS, bal_curr)
    cash = find_any(bal, CASH_KEYS, bal_curr)
    sti = find_any(bal, STI_KEYS, bal_curr)
    total_debt = find_any(bal, TOT_DEBT_KEYS, bal_curr)
    if pd.isna(total_debt) or total_debt == 0:
        ltd = find_any(bal, LTD_KEYS, bal_curr)
        sltd = find_any(bal, SLTD_KEYS, bal_curr)
        cdebt = find_any(bal, CUR_DEBT_KEYS, bal_curr)
        parts = [x for x in [ltd, sltd, cdebt] if not pd.isna(x)]
        total_debt = sum(parts) if parts else np.nan
    ta_prev = find_any(bal, TA_KEYS, bal_prev) if bal_prev else np.nan
    te_prev = find_any(bal, TE_KEYS, bal_prev) if bal_prev else np.nan
    inv_prev = find_any(bal, INV_KEYS, bal_prev) if bal_prev else np.nan
    ar_prev = find_any(bal, AR_KEYS, bal_prev) if bal_prev else np.nan
    ap_prev = find_any(bal, AP_KEYS, bal_prev) if bal_prev else np.nan
    avg_assets = np.nanmean([ta, ta_prev]) if not pd.isna(ta) else np.nan
    avg_equity = np.nanmean([te, te_prev]) if not pd.isna(te) else np.nan
    avg_inv = np.nanmean([inv, inv_prev]) if not pd.isna(inv) else np.nan
    avg_ar = np.nanmean([ar, ar_prev]) if not pd.isna(ar) else np.nan
    avg_ap = np.nanmean([ap, ap_prev]) if not pd.isna(ap) else np.nan
    if cf is not None and not cf.empty and use_cf_cols:
        ocf = sum([find_any(cf, OCF_KEYS, c) for c in use_cf_cols])
        capex_raw = [find_any(cf, CAPEX_KEYS, c) for c in use_cf_cols]
        capex = sum([x for x in capex_raw if not pd.isna(x)])
    else:
        ocf = np.nan
        capex = np.nan
    int_exp = sum([find_any(inc, INT_EXP_KEYS, c) for c in use_inc_cols])
    int_exp_abs = abs(int_exp) if not pd.isna(int_exp) else np.nan
    gross_margin = safe_div(gp, rev)
    operating_margin = safe_div(opi, rev)
    net_margin = safe_div(ni, rev)
    roe = safe_div(ni, avg_equity)
    roa = safe_div(ni, avg_assets)
    eff_tax_rate = tax_rate if (not pd.isna(tax_rate) and 0 <= tax_rate <= 0.6) else 0.25
    nopat = ebit * (1 - eff_tax_rate) if not pd.isna(ebit) else np.nan
    invested_capital = total_debt + te - (cash if not pd.isna(cash) else 0) if not (pd.isna(total_debt) or pd.isna(te)) else np.nan
    roic = safe_div(nopat, invested_capital)
    current_ratio = safe_div(ca, cl)
    quick_ratio = safe_div((ca - (inv if not pd.isna(inv) else 0)), cl)
    cash_ratio = safe_div((cash if not pd.isna(cash) else 0) + (sti if not pd.isna(sti) else 0), cl)
    debt_to_equity = safe_div(total_debt, te)
    debt_to_assets = safe_div(total_debt, ta)
    interest_coverage = safe_div(ebit, int_exp_abs)
    asset_turnover = safe_div(rev, avg_assets)
    inventory_turnover = safe_div((cogs if not pd.isna(cogs) else rev), avg_inv)
    receivables_turnover = safe_div(rev, avg_ar)
    payables_turnover = safe_div((cogs if not pd.isna(cogs) else rev), avg_ap)
    dso = safe_div(365, receivables_turnover)
    dio = safe_div(365, inventory_turnover)
    dpo = safe_div(365, payables_turnover)
    ccc = dso + dio - dpo if not (pd.isna(dso) or pd.isna(dio) or pd.isna(dpo)) else np.nan
    fcf = ocf - capex if not (pd.isna(ocf) or pd.isna(capex)) else np.nan
    ocf_to_ni = safe_div(ocf, ni)
    fcf_margin = safe_div(fcf, rev)
    price = data.get("price", np.nan)
    shares = data.get("shares", np.nan)
    pe = pb = ps = np.nan
    if not (pd.isna(price) or pd.isna(shares) or shares == 0):
        eps = safe_div(ni, shares)
        pe = safe_div(price, eps)
        sales_ps = safe_div(rev, shares)
        ps = safe_div(price, sales_ps)
        bvps = safe_div(te, shares)
        pb = safe_div(price, bvps)
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
        "Price": data.get("price", np.nan), "Shares": data.get("shares", np.nan)
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
    days_keys = ["أيام التحصيل DSO","أيام المخزون DIO","أيام السداد DPO","دورة التحويل النقدي CCC"]
    val_keys = ["P/E (اختياري)","P/B (اختياري)","P/S (اختياري)"]
    for k,v in core.items():
        if k in percent_keys:
            view[k] = to_percent(v)
        elif k in days_keys:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.1f} يوم"
        elif k in val_keys:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}x"
        elif k in ratio_keys:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}x"
        else:
            view[k] = "—" if v is None or pd.isna(v) else f"{v:.2f}"
    return view

def generate_html_table(df: pd.DataFrame, buffett=False) -> str:
    html = """
    <style>
    table.buffett-table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
    .buffett-table th, .buffett-table td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    .buffett-table th {background-color: #0ea5e9; color: white;}
    .buffett-table tr:nth-child(even){background-color: #f8fafc;}
    .buffett-table tr:hover {background-color: #eef2ff;}
    .buffett-table td.green { color: green; font-weight: bold; }
    .buffett-table td.yellow { color: orange; font-weight: bold; }
    .buffett-table td.red { color: red; font-weight: bold; }
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
            if buffett and col != "الرمز" and col != "الدرجة" and col != "التوصية":
                cls = "green" if row[col] == "✅" else "yellow" if row[col] == "⚠️" else "red" if row[col] == "❌" else ""
            html += f"<td class='{cls}'>{escape(str(row[col]))}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# واجهة المستخدم
# =============================
st.title("📊 التحليل الأساسي | Buffett Score")
st.caption("تحليل بناءً على إطار وارن بوفيت: جودة الأعمال، القوة المالية، تحويل النقد، اقتصاديات التشغيل، التقييم.")

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    market = st.selectbox("السوق", ["السوق الأمريكي", "السوق السعودي"])
    suffix = "" if market == "السوق الأمريكي" else ".SR"
    mode = st.radio("الفترة", ["Annual", "TTM"], index=1, help="TTM = مجموع 4 أرباع أخيرة؛ Annual = آخر سنة مالية.")
    show_buffett = st.checkbox("عرض Buffett Score", value=True)
    top_only = st.checkbox("عرض النِّسَب الأساسية فقط (جدول مختصر)", value=True)
    show_raw = st.checkbox("إظهار قيم البنود الخام (Revenue/Assets/…)", value=False)
    st.markdown("---")
    st.markdown("#### 🧪 رموز تجريبية")
    if st.button("USA: AAPL MSFT NVDA"):
        st.session_state.syms = "AAPL MSFT NVDA"
    if st.button("KSA: 1120 2380 1050"):
        st.session_state.syms = "1120 2380 1050"

symbols_input = st.text_area("أدخل الرموز (مسافة/سطر). استخدم اللاحقة .SR للسعودي أو اختر السوق وسأضيفها تلقائيًا.", 
                             st.session_state.get("syms",""))
raw = [s.strip().upper() for s in symbols_input.replace("\n"," ").split() if s.strip()]
clean = []
for s in raw:
    if suffix and not s.endswith(suffix) and s.isalpha():
        clean.append(s + suffix)
    else:
        clean.append(s)
symbols = sorted(set(clean))

if st.button("🚀 احسب النِّسَب"):
    if not symbols:
        st.warning("أدخل رمزًا واحدًا على الأقل.")
        st.stop()
    rows = []
    raw_rows = []
    buffett_rows = []
    errors = []
    progress = st.progress(0, text=f"بدء الحساب... (0/{len(symbols)})")
    for i, code in enumerate(symbols, start=1):
        try:
            data = load_company_data(code)
            core, rawvals = compute_ratios(data, mode=mode)
            if core is None:
                errors.append(code)
                continue
            view = format_core_row(core)
            row = {"الرمز": code}
            row.update(view if top_only else view)
            rows.append(row)
            if show_raw:
                rv = {"الرمز": code}
                for k,v in rawvals.items():
                    rv[k] = to_num(v, 2)
                raw_rows.append(rv)
            if show_buffett:
                buffett_data = buffett_score(data, mode=mode)
                if buffett_data:
                    buffett_row = {"الرمز": code, "الدرجة": f"{buffett_data['score']:.1f}/100"}
                    buffett_row.update(buffett_data['results'])
                    buffett_row['التوصية'] = buffett_data['recommendation']
                    buffett_rows.append(buffett_row)
        except Exception as e:
            errors.append(f"{code} → {e}")
        finally:
            progress.progress(i/len(symbols), text=f"تم {i}/{len(symbols)}")
    if buffett_rows and show_buffett:
        st.subheader(f"Buffett Score ({mode}) — {len(buffett_rows)} شركة")
        df_buffett = pd.DataFrame(buffett_rows)
        st.markdown(generate_html_table(df_buffett, buffett=True), unsafe_allow_html=True)
        csv_buffett = df_buffett.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 تنزيل Buffett Score CSV", csv_buffett, file_name=f"buffett_score_{mode}.csv", mime="text/csv")
    if rows:
        st.subheader(f"نتائج النِّسَب ({mode}) — {len(rows)} شركة")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        html_out = generate_html_table(df)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📥 تنزيل CSV", csv_bytes, file_name=f"fundamentals_{mode}.csv", mime="text/csv")
        with c2:
            st.download_button("📥 تنزيل HTML", html_out.encode("utf-8"), file_name=f"fundamentals_{mode}.html", mime="text/html")
    if show_raw and raw_rows:
        st.markdown("---")
        st.subheader("القيم الخام (للتدقيق/المراجعة)")
        df_raw = pd.DataFrame(raw_rows)
        st.dataframe(df_raw, use_container_width=True)
    if errors:
        st.info("⚠️ ملاحظات الرموز:")
        for e in errors:
            st.write("• ", e)

# تلميحات استخدام
with st.expander("📌 ملاحظات ومنهجية الحساب"):
    st.markdown("""
- **TTM**: نستخدم مجموع آخر 4 أرباع للبنود الدخلية، وأحدث ميزانية للأرصدة، ومتوسط (الحالية + السابقة) للأصول/الحقوق.
- **Buffett Score**: يقيّم بناءً على 9 معايير (هوامش >20%، ROIC >20%، OCF/NI >1، FCF Margin >10%، CCC ≤0، صافي الدين ≤0، تغطية الفوائد >10x، نمو EPS/FCF، عائد المالك ≥8%). الدرجة من 0-100.
- **المرونة**: بنود Yahoo قد تختلف أسماءها؛ نعتمد قائمة مرادفات، وإذا غاب البند تُعرض النسبة بـ "—".
- **ROIC~**: تقدير: NOPAT (≈ EBIT×(1–الضريبة)) على (الدين + حقوق المساهمين – النقد).
- **التقييم**: P/E وP/B وP/S تُحسب إذا وُجد السعر وعدد الأسهم.
- **السعودي**: أضف `.SR` للرمز أو اختر السوق.
""")
