# streamlit_dcf_fcfe_2stage_app.py
# -*- coding: utf-8 -*-
"""
Investor‑grade Streamlit app for a 2‑Stage FCFE DCF.

Run:
    pip install streamlit yfinance pandas numpy plotly kaleido
    streamlit run streamlit_dcf_fcfe_2stage_app.py
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports & Config
# ─────────────────────────────────────────────────────────────
import json, math, zipfile, io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="DCF – 2‑Stage FCFE | Investor App",
    layout="wide",
    page_icon="💹",
)

# Small CSS polish for professional look
st.markdown(
    """
    <style>
.kpi{background:#f7f9fc;color:#0f172a;padding:8px 10px;border-radius:12px;border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,.04);width:100%}
.kpi h4{margin:0;font-size:12px;color:#64748b}
.kpi .v{font-size:18px;font-weight:700;color:#0f172a}
.badge{padding:2px 8px;border-radius:6px;font-weight:800}
.badge.up{background:rgba(34,197,94,.15);color:#15803d}
.badge.down{background:rgba(239,68,68,.12);color:#b91c1c}
.subtle{color:#94a3b8;font-size:10px}
.section{margin-top:6px}
</style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def parse_percent(s) -> float:
    """Accepts 3, 3%, 3.0, -1, "2,5" -> 0.03, -0.01 etc."""
    x = float(str(s).strip().replace("%", "").replace(" ", "").replace(",", ".") or "0")
    return x / 100.0 if abs(x) >= 1.0 else x


def format_usd(x: float) -> str:
    return f"US${x:,.2f}"


def re_lever_beta(beta_u: float, tax, d_me, blume: float = 0.67) -> float:
    """Hamada on D/E then Blume shrink toward 1.0."""
    t = parse_percent(tax)
    de = parse_percent(d_me)
    beta_L_raw = beta_u * (1 + (1 - t) * de)
    return blume * beta_L_raw + (1 - blume) * 1.0


def cost_of_equity(rf, erp, beta) -> float:
    return parse_percent(rf) + beta * parse_percent(erp)


def build_fcfe(start_fcfe: float, growth_pa: float, years: int) -> List[float]:
    g = parse_percent(growth_pa)
    n = np.arange(1, int(years) + 1, dtype=float)
    return (start_fcfe * (1 + g) ** n).tolist()


def fetch_live_defaults(tkr: str) -> Tuple[float | None, float | None]:
    """Return (last_price, shares_m)."""
    try:
        y = yf.Ticker(tkr)
        price = None
        so = None
        fi = getattr(y, "fast_info", None)
        if fi:
            for k in ("last_price", "lastPrice", "regularMarketPrice", "last_close", "lastClose"):
                if k in fi and fi[k] is not None:
                    price = float(fi[k])
                    break
            so = fi.get("shares_outstanding") or fi.get("sharesOutstanding")
        if price is None:
            h = y.history(period="5d", auto_adjust=False)
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        if so is None:
            info = y.info or {}
            so = info.get("sharesOutstanding")
        shares_m = float(so) / 1e6 if so else None
        return (float(price) if price is not None else None, shares_m)
    except Exception:
        return (None, None)


# ─────────────────────────────────────────────────────────────
# FCFE fetch from Yahoo Finance
# ─────────────────────────────────────────────────────────────

def fetch_start_fcfe_from_yf(tkr: str, mode: str = "TTM") -> Tuple[Optional[float], Dict[str, float], str]:
    """Return (start_fcfe_m_USD, breakdown, reporting_currency).
    mode: "TTM" = sum last 4 quarters; "FY" = last fiscal year.
    FCFE ≈ CFO + CapEx + NetDebt (Issuance + Repayment). CapEx is usually negative.
    """
    try:
        y = yf.Ticker(tkr)
        # currency detection
        cur = None
        fi = getattr(y, "fast_info", None)
        if fi and isinstance(fi, dict):
            cur = fi.get("currency")
        if not cur:
            info = y.info or {}
            cur = info.get("currency", "USD")
        if str(cur).upper() == "GBP":
            cur = "GBP"  # normalize
        if str(cur).upper() == "GBP":
            cur = "GBP"
        if str(cur).upper() == "GBP":
            cur = "GBP"
        # load cashflow
        df = None
        if mode.upper() == "TTM":
            df = y.quarterly_cashflow
        else:
            df = y.cashflow
        if df is None or df.empty:
            return (None, {}, str(cur or "USD").upper())
        # aggregate series
        if mode.upper() == "TTM":
            cols = list(df.columns)[:4]
            s = df[cols].fillna(0.0).sum(axis=1)
        else:
            col = list(df.columns)[0]
            s = df[col].fillna(0.0)
        def pick(keys: List[str]) -> float:
            keys_n = [k.lower() for k in keys]
            for idx in s.index:
                name = str(idx).lower()
                if any(k in name for k in keys_n):
                    try:
                        return float(s.loc[idx])
                    except Exception:
                        return 0.0
            return 0.0
        cfo = pick(["total cash from operating activities", "operating cash flow", "cash from operating activities"])  # +
        capex = pick(["capital expenditures"])  # usually negative
        iss = pick(["issuance of debt", "debt issued"])  # +
        rep = pick(["repayment of debt", "debt repayment"])  # negative
        net_debt = iss + rep
        fcfe = cfo + capex + net_debt
        # FX to USD
        cur_up = str(cur or "USD").upper()
        if cur_up == "GBP":
            pair = "GBPUSD=X"
        else:
            pair = f"{cur_up}USD=X"
        rate = 1.0
        if cur_up != "USD":
            try:
                fx = yf.Ticker(pair)
                h = fx.history(period="5d")
                if not h.empty:
                    rate = float(h["Close"].iloc[-1])
            except Exception:
                rate = 1.0
        fcfe_usd = fcfe * rate
        brk = {
            "cfo_m": cfo / 1e6 * rate,
            "capex_m": capex / 1e6 * rate,
            "net_debt_m": net_debt / 1e6 * rate,
            "fcfe_m": fcfe_usd / 1e6,
            "fx": rate,
        }
        return (fcfe_usd / 1e6, brk, cur_up)
    except Exception:
        return (None, {}, "USD")

# ─────────────────────────────────────────────────────────────
# DCF Core
# ─────────────────────────────────────────────────────────────

@dataclass
class DCFInputs:
    years: List[int]  # calendar years t=1..N
    fcfe_m: List[float]  # USD million
    rf: float
    erp: float
    beta_u: float
    tax: float
    d_me: float
    g_perp: float
    shares_m: float
    price: Optional[float] = None
    beta_floor: float = 0.8
    beta_cap: float = 2.0

    def __post_init__(self):
        if self.shares_m <= 0:
            raise ValueError("shares_m must be > 0.")
        if self.beta_floor > self.beta_cap:
            raise ValueError("β floor > β cap.")
        self.tax = parse_percent(self.tax)
        self.d_me = parse_percent(self.d_me)
        self.rf = parse_percent(self.rf)
        self.erp = parse_percent(self.erp)
        self.g_perp = parse_percent(self.g_perp)


@dataclass
class DCFResult:
    schedule: pd.DataFrame
    beta: float
    r: float
    pv_stage1_m: float
    tv_m: float
    pv_tv_m: float
    eq_total_m: float
    fair_ps: float
    discount_vs_price: Optional[float]


def dcf_fcfe_2stage(inp: DCFInputs) -> DCFResult:
    if len(inp.years) != len(inp.fcfe_m):
        raise ValueError("years != fcfe length.")
    beta_rl = re_lever_beta(inp.beta_u, inp.tax, inp.d_me)
    beta = min(max(beta_rl, inp.beta_floor), inp.beta_cap)
    r = cost_of_equity(inp.rf, inp.erp, beta)
    g = inp.g_perp
    if g >= r:
        raise ValueError("Perpetual growth must be < discount rate.")

    t = np.arange(1, len(inp.fcfe_m) + 1, dtype=float)
    df = (1 + r) ** t
    pv = np.asarray(inp.fcfe_m, float) / df
    pv_stage1 = float(pv.sum())

    fcf_T = float(inp.fcfe_m[-1])
    tv = fcf_T * (1 + g) / (r - g)
    pv_tv = tv / ((1 + r) ** len(inp.fcfe_m))

    eq_total = pv_stage1 + pv_tv
    fair_ps = eq_total / float(inp.shares_m)
    disc = None if inp.price in (None, 0) else (fair_ps - float(inp.price)) / fair_ps

    sched = pd.DataFrame(
        {
            "Year": inp.years,
            "FCFE (USD m)": inp.fcfe_m,
            "Discount Factor": 1 / df,
            "PV FCFE (USD m)": pv,
        }
    )
    return DCFResult(sched, beta, r, pv_stage1, tv, pv_tv, eq_total, fair_ps, disc)


def solve_g_for_tv_share(pv_stage1_m: float, r: float, fcf_T_m: float, N: int, s: float) -> float:
    target = (s / (1 - s)) * pv_stage1_m
    T = target * ((1 + r) ** N) / fcf_T_m
    g = (T * r - 1.0) / (1.0 + T)
    return g


# ─────────────────────────────────────────────────────────────
# UI State Defaults
# ─────────────────────────────────────────────────────────────

DEFAULTS = dict(
    ticker="NOG",
    price=24.84,
    start_fcfe=650.0,
    horizon=5,
    growth=7.0,
    shares_m=95.0,
    rf="3.0",
    erp="5.0",
    beta_u=1.00,
    tax="23.0",
    dme="10",
    g_perp="2.0",
    beta_floor=0.8,
    beta_cap=2.0,
    tv_target=0.65,
    lock_tv=False,
)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────
# Sidebar Inputs
# ─────────────────────────────────────────────────────────────

export_now = False  # ensure defined

with st.sidebar:
    st.header("Inputs")

    col_t = st.columns([3, 1])
    with col_t[0]:
        st.text_input("Ticker", key="ticker")
    with col_t[1]:
        if st.button("↻ Live", help="Fetch price & shares via yfinance"):
            px, sh_m = fetch_live_defaults(st.session_state.ticker.strip())
            if px is not None:
                st.session_state.price = round(float(px), 2)
            if sh_m is not None:
                st.session_state.shares_m = round(float(sh_m), 2)
            st.rerun()

    st.number_input("Price", min_value=0.0, value=float(st.session_state.price), step=0.01, key="price")

    # Start FCFE (manual)
    st.number_input("Start FCFE (USD m)", min_value=0.0, value=float(st.session_state.start_fcfe), step=10.0, key="start_fcfe")

    # Core knobs
    st.select_slider("Horizon (years)", options=list(range(5, 11)), value=int(st.session_state.horizon), key="horizon")
    st.slider("Growth %/yr", min_value=-10.0, max_value=25.0, value=float(st.session_state.growth), step=0.1, key="growth")
    st.number_input("Shares (m)", min_value=0.01, value=float(st.session_state.shares_m), step=0.01, key="shares_m")

    with st.expander("Advanced", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Rf %", key="rf")
            st.number_input("β unlevered", min_value=0.0, max_value=5.0, value=float(st.session_state.beta_u), step=0.01, key="beta_u")
            st.slider("β floor", min_value=0.0, max_value=2.0, value=float(st.session_state.beta_floor), step=0.1, key="beta_floor")
        with c2:
            st.text_input("ERP %", key="erp")
            st.text_input("Tax %", key="tax")
            st.text_input("D/ME %", key="dme")
            st.slider("β cap", min_value=0.5, max_value=3.0, value=float(st.session_state.beta_cap), step=0.1, key="beta_cap")
        st.text_input("g perp %", key="g_perp")

    st.markdown("---")
    st.subheader("Terminal‑Value control")
    st.number_input("TV share target (0..1)", min_value=0.05, max_value=0.95, value=float(st.session_state.tv_target), step=0.01, key="tv_target")
    st.checkbox("Lock TV share (solve g)", value=bool(st.session_state.lock_tv), key="lock_tv")
    if st.button("Cap TV share now"):
        st.session_state["__cap_tv_now__"] = True
        st.rerun()

    st.markdown("---")
    st.subheader("Export")
    save_label = st.text_input("Label", value="run")
    do_zip = st.checkbox("ZIP bundle", value=True)
    export_now = st.button("Export current run")

# ─────────────────────────────────────────────────────────────
# Calculation
# ─────────────────────────────────────────────────────────────

error_msg = None
result: Optional[DCFResult] = None
inputs_obj: Optional[DCFInputs] = None

try:
    start_year = pd.Timestamp.today().year + 1
    years = list(range(start_year, start_year + int(st.session_state.horizon)))
    fcfe = build_fcfe(float(st.session_state.start_fcfe), growth_pa=st.session_state.growth, years=st.session_state.horizon)

    inputs_obj = DCFInputs(
        years=years,
        fcfe_m=fcfe,
        rf=st.session_state.rf,
        erp=st.session_state.erp,
        beta_u=float(st.session_state.beta_u),
        tax=st.session_state.tax,
        d_me=st.session_state.dme,
        g_perp=st.session_state.g_perp,
        shares_m=float(st.session_state.shares_m),
        price=float(st.session_state.price),
        beta_floor=float(st.session_state.beta_floor),
        beta_cap=float(st.session_state.beta_cap),
    )

    result = dcf_fcfe_2stage(inputs_obj)

    # Lock TV share logic: recompute g to hit target and rerun
    has_limit = 0 < float(st.session_state.tv_target) < 1
    if bool(st.session_state.lock_tv) and has_limit:
        s = float(st.session_state.tv_target)
        g_new = solve_g_for_tv_share(result.pv_stage1_m, result.r, inputs_obj.fcfe_m[-1], len(inputs_obj.fcfe_m), s)
        g_new = max(-0.02, min(result.r - 1e-4, g_new))  # −2% ≤ g < r
        if abs(g_new - inputs_obj.g_perp) > 1e-6:
            st.session_state.g_perp = f"{g_new * 100:.2f}"
            st.rerun()

    # Cap TV share button pressed
    if st.session_state.pop("__cap_tv_now__", False):
        if has_limit:
            s = float(st.session_state.tv_target)
            g_new = solve_g_for_tv_share(result.pv_stage1_m, result.r, inputs_obj.fcfe_m[-1], len(inputs_obj.fcfe_m), s)
            g_new = max(-0.02, min(result.r - 1e-4, g_new))
            st.session_state.g_perp = f"{g_new * 100:.2f}"
            st.rerun()

except Exception as e:
    error_msg = str(e)


# ─────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────

st.title("DCF – 2‑Stage FCFE")

if error_msg:
    st.error(f"Fehler: {error_msg}")
else:
    # Derived metrics
    disc = 0.0 if result.discount_vs_price is None else float(result.discount_vs_price)
    tag_color = "#22c55e" if disc > 0 else "#ef4444"
    tv_share = result.pv_tv_m / (result.pv_stage1_m + result.pv_tv_m) if (result.pv_stage1_m + result.pv_tv_m) > 0 else 0.0

    # ── KPI Top Row ─────────────────────────────────────
    top = st.container()
    with top:
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        with c1:
            st.markdown(f"<div class='kpi'><h4>Fair Value / Share</h4><div class='v'>{format_usd(result.fair_ps)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='kpi'><h4>Current Price</h4><div class='v'>{format_usd(float(st.session_state.price))}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='kpi'><h4>Cost of Equity</h4><div class='v'>{result.r*100:.2f}%</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='kpi'><h4>Levered Beta</h4><div class='v'>{result.beta:.2f}</div></div>", unsafe_allow_html=True)
        with c5:
            badge_class = 'up' if disc>0 else 'down'
            st.markdown(
                f"<div class='kpi'><h4>Valuation vs Price</h4><div class='v'><span class='badge {badge_class}'>"
                f"{('UNDERVALUED' if disc>0 else 'OVERVALUED')} {abs(disc)*100:.1f}%" 
                f"</span></div></div>",
                unsafe_allow_html=True,
            )

    # ── KPI Second Row ─────────────────────────────────
    sub = st.container()
    with sub:
        s1, s2, s3 = st.columns(3, gap="medium")
        with s1:
            st.markdown(f"<div class='kpi'><h4>PV Stage 1</h4><div class='v'>{format_usd(result.pv_stage1_m*1e6)}</div></div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div class='kpi'><h4>PV Terminal</h4><div class='v'>{format_usd(result.pv_tv_m*1e6)}</div></div>", unsafe_allow_html=True)
        with s3:
            color_tv = "#22c55e" if (0 < float(st.session_state.tv_target) < 1 and tv_share <= float(st.session_state.tv_target)) else "#ef4444"
            st.markdown(
                f"<div class='kpi'><h4>TV Share</h4><div class='v'><span style='color:{color_tv}'>{tv_share*100:.1f}%</span>"
                + (f" <span class='subtle'>(limit {float(st.session_state.tv_target)*100:.0f}%)</span>" if 0 < float(st.session_state.tv_target) < 1 else "")
                + "</div></div>",
                unsafe_allow_html=True,
            )

    # ── Charts Row ─────────────────────────────────────
    c1, c2 = st.columns([1.6, 1.0], gap="large")
    with c1:
        fig = px.line(
            result.schedule,
            x="Year",
            y="FCFE (USD m)",
            markers=True,
            title=f"Future FCFE path ({int(st.session_state.horizon)}y) | Terminal g {parse_percent(st.session_state.g_perp)*100:.2f}%",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        last_y = result.schedule["Year"].iat[-1]
        last_v = float(result.schedule["FCFE (USD m)"].iat[-1])
        fig.add_annotation(x=last_y, y=last_v, text=f"{last_v:,.1f} m", showarrow=True, arrowhead=2)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wf = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["PV Stage 1", "PV Terminal", "Equity Value"],
                textposition="outside",
                y=[result.pv_stage1_m, result.pv_tv_m, 0],
                text=[f"{result.pv_stage1_m:,.0f} m", f"{result.pv_tv_m:,.0f} m", f"{result.eq_total_m:,.0f} m"],
            )
        )
        wf.update_layout(title="PV Decomposition (USD m)", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(wf, use_container_width=True)

    st.markdown("### Cash‑flow Schedule")
    df_disp = result.schedule.copy()
    st.dataframe(
        df_disp,
        use_container_width=True,
        hide_index=True,
    )

# ── Export block ───────────────────────────────────────
    if export_now:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = (save_label or "run").strip()
        base_name = f"{st.session_state.ticker}_{label}_{ts}"
        out_dir = Path("exports") / base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect inputs as JSON
        inputs_payload: Dict[str, Any] = {
            "ticker": st.session_state.ticker,
            "price": float(st.session_state.price),
            "start_fcfe_m": float(st.session_state.start_fcfe),
            "horizon_years": int(st.session_state.horizon),
            "growth_pct_pa": float(st.session_state.growth),
            "shares_m": float(st.session_state.shares_m),
            "rf_pct": str(st.session_state.rf),
            "erp_pct": str(st.session_state.erp),
            "beta_unlevered": float(st.session_state.beta_u),
            "tax_pct": str(st.session_state.tax),
            "d_me_pct": str(st.session_state.dme),
            "g_perp_pct": str(st.session_state.g_perp),
            "beta_floor": float(st.session_state.beta_floor),
            "beta_cap": float(st.session_state.beta_cap),
        }

        # Files
        schedule_csv = out_dir / "schedule.csv"
        df_disp.to_csv(schedule_csv, index=False)

        summary = {
            "ticker": st.session_state.ticker,
            "fair_value_per_share": round(result.fair_ps, 6),
            "current_price": float(st.session_state.price),
            "discount_vs_price": None if result.discount_vs_price is None else round(result.discount_vs_price, 6),
            "cost_of_equity": round(result.r, 6),
            "levered_beta": round(result.beta, 6),
            "rf": round(inputs_obj.rf, 6),
            "erp": round(inputs_obj.erp, 6),
            "tax": round(inputs_obj.tax, 6),
            "D_over_E": round(inputs_obj.d_me, 6),
            "g_perp": round(inputs_obj.g_perp, 6),
            "pv_stage1_m": round(result.pv_stage1_m, 6),
            "terminal_value_m": round(result.tv_m, 6),
            "pv_terminal_value_m": round(result.pv_tv_m, 6),
            "equity_value_total_m": round(result.eq_total_m, 6),
            "shares_m": round(inputs_obj.shares_m, 6),
            "years": inputs_obj.years,
            "timestamp": ts,
        }
        (out_dir / "inputs.json").write_text(json.dumps(inputs_payload, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Figure PNG via kaleido
        fig_png = None
        try:
            fig_png = fig.to_image(format="png", scale=2)
            (out_dir / "fcfe_path.png").write_bytes(fig_png)
        except Exception:
            pass

        # Zip in memory for download
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{base_name}/inputs.json", json.dumps(inputs_payload, indent=2))
            z.writestr(f"{base_name}/summary.json", json.dumps(summary, indent=2))
            z.writestr(f"{base_name}/schedule.csv", df_disp.to_csv(index=False))
            if fig_png is not None:
                z.writestr(f"{base_name}/fcfe_path.png", fig_png)
        zip_buf.seek(0)

        if do_zip:
            st.download_button(
                label=f"Download {base_name}.zip",
                data=zip_buf,
                file_name=f"{base_name}.zip",
                mime="application/zip",
            )
        else:
            st.download_button(
                label="Download schedule.csv",
                data=df_disp.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}_schedule.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download summary.json",
                data=json.dumps(summary, indent=2).encode("utf-8"),
                file_name=f"{base_name}_summary.json",
                mime="application/json",
            )
            if fig_png is not None:
                st.download_button(
                    label="Download fcfe_path.png",
                    data=fig_png,
                    file_name=f"{base_name}_fcfe_path.png",
                    mime="image/png",
                )

st.caption("Model: FCFE DCF, levered β via Hamada + Blume, CAPM cost of equity. Perpetual g < r constraint enforced.")
