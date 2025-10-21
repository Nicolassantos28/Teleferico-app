# -*- coding: utf-8 -*-
# app.py — Modelo Cablebús (ejecutivo, claro y robusto)
# Ejecuta:  streamlit run app.py

import io
import zipfile
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# CONFIG & ESTILO
# =========================
st.set_page_config(page_title="Modelo Cablebús — Tablero Financiero", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 17px; }
h1 { font-size: 42px !important; margin-bottom: 0; }
h2 { font-size: 28px !important; margin: 10px 0 4px 0; }
h3 { font-size: 20px !important; margin: 8px 0 2px 0; }
div[data-testid="stMetricValue"] { font-size: 34px !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #9aa3af !important; }
.dataframe tbody td, .dataframe thead th { font-size: 15px !important; }
.rail-derecho { position: sticky; top: 72px; border-left: 1px solid #e5e7eb; padding-left: 14px; }
</style>
""", unsafe_allow_html=True)
alt.data_transformers.disable_max_rows()

# =========================
# UTILIDADES
# =========================
def _norm_num(s, default):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return float(default)

def money_input(label: str, key: str, default: float, help: str | None = None, decimals: int = 2) -> float:
    initial = st.session_state.get(key, default)
    shown = f"{float(initial):,.{decimals}f}"
    txt = st.text_input(label, value=shown, key=f"{key}__txt", help=help)
    val = _norm_num(txt, default)
    st.session_state[key] = val
    return val

def int_input_commas(label: str, key: str, default: int, help: str | None=None) -> int:
    initial = int(st.session_state.get(key, default))
    txt = st.text_input(label, value=f"{initial:,}", key=f"{key}__txt", help=help)
    val = int(round(_norm_num(txt, default)))
    st.session_state[key] = val
    return val

def percent_input(label: str, key: str, default_percent: float, help: str | None = None, decimals: int = 2) -> float:
    initial = st.session_state.get(key, default_percent)
    shown = f"{float(initial):.{decimals}f}"
    txt = st.text_input(f"{label} (%)", value=shown, key=f"{key}__pct", help=help)
    val_pct = _norm_num(txt, default_percent)
    st.session_state[key] = val_pct
    return max(0.0, float(val_pct)) / 100.0

def with_commas(n) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return ""

def pesos(n) -> str:
    try:
        return f"${int(round(float(n))):,}"
    except Exception:
        return ""

def df_wide(data, **kwargs):
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)

def bar_comp(df_name_value_pct, title):
    return (
        alt.Chart(df_name_value_pct)
        .mark_bar()
        .encode(
            x=alt.X("Nombre:N", sort="-y", title=None, axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
            y=alt.Y("Monto:Q", axis=alt.Axis(format=",", title="Monto (MXN)", labelFontSize=14, titleFontSize=16)),
            tooltip=[alt.Tooltip("Nombre:N"), alt.Tooltip("Monto:Q", format=","), alt.Tooltip("% del total:Q", format=".1%")],
            color=alt.Color("Nombre:N", legend=None),
        )
        .properties(height=300, title=alt.TitleParams(title, fontSize=16))
    )

def line_series(df_long, title):
    return (
        alt.Chart(df_long)
        .mark_line(point=False, strokeWidth=3)
        .encode(
            x=alt.X("Año:O", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
            y=alt.Y("Valor:Q", axis=alt.Axis(format=",", labelFontSize=14, titleFontSize=16)),
            color=alt.Color("Serie:N", legend=alt.Legend(title=None, labelFontSize=13)),
            tooltip=[alt.Tooltip("Año:O"), alt.Tooltip("Serie:N"), alt.Tooltip("Valor:Q", format=",")],
        )
        .properties(height=320, title=alt.TitleParams(title, fontSize=16))
    )

def annuity_payment_monthly(principal: float, annual_rate: float, years: int) -> float:
    P = max(float(principal), 0.0)
    rm = max(float(annual_rate), 0.0) / 12.0
    m = max(int(years * 12), 1)
    if rm == 0:
        return P / m
    return P * rm / (1 - (1 + rm) ** (-m))

def amort_yearly_from_monthly(P: float, annual_rate: float, years: int, horizonte: int):
    P = float(P)
    n_m = int(max(years, 0) * 12)
    rm  = float(max(annual_rate, 0.0) / 12.0)
    pm  = annuity_payment_monthly(P, annual_rate, years) if n_m > 0 else 0.0

    pagos_m = np.zeros(n_m, dtype=float)
    ints_m  = np.zeros(n_m, dtype=float)
    amort_m = np.zeros(n_m, dtype=float)
    saldo_m = np.zeros(n_m, dtype=float)
    saldo = P
    for i in range(n_m):
        interes = saldo * rm
        pago    = pm
        amort   = max(0.0, pago - interes)
        saldo   = max(0.0, saldo + interes - pago)
        pagos_m[i] = pago;  ints_m[i] = interes;  amort_m[i] = amort;  saldo_m[i] = saldo

    max_years = max(horizonte, years)
    pay_y = np.zeros(max_years); int_y = np.zeros(max_years); amo_y = np.zeros(max_years); bal_y = np.zeros(max_years)
    for t in range(1, max_years + 1):
        a, b = (t-1)*12, min(t*12, n_m)
        if a < b:
            pay_y[t-1] = pagos_m[a:b].sum(); int_y[t-1] = ints_m[a:b].sum()
            amo_y[t-1] = amort_m[a:b].sum(); bal_y[t-1] = saldo_m[b-1]
    if horizonte <= max_years:
        return pay_y[:horizonte], int_y[:horizonte], amo_y[:horizonte], bal_y[:horizonte]
    pad = horizonte - max_years
    return np.pad(pay_y,(0,pad)), np.pad(int_y,(0,pad)), np.pad(amo_y,(0,pad)), np.pad(bal_y,(0,pad))

def simulate_private_sweep(P, annual_rate, years, utilidad_anual, sweep_alpha, allow_prepay, horizonte):
    n_m = int(max(years, 0) * 12)
    rm = float(max(annual_rate, 0.0) / 12.0)
    pm = annuity_payment_monthly(P, annual_rate, years) if n_m > 0 else 0.0

    saldo = float(P)
    pagos_m = []; ints_m = []; amort_m = []; saldo_m = []
    for i in range(n_m):
        y = min(len(utilidad_anual)-1, i // 12)
        avail = float(sweep_alpha * max(0.0, utilidad_anual[y]) / 12.0)
        interes = saldo * rm
        pago_prog = pm
        pago_real = min(avail, pago_prog)
        extra = 0.0
        if allow_prepay and avail > pago_prog:
            extra = avail - pago_prog
            pago_real = avail
        amort = max(0.0, pago_real - interes) + max(0.0, extra)
        saldo = max(0.0, saldo + interes - pago_real - extra)
        pagos_m.append(pago_real + extra); ints_m.append(interes); amort_m.append(amort); saldo_m.append(saldo)

    max_years = max(horizonte, years)
    pay_y = np.zeros(max_years); int_y = np.zeros(max_years); amo_y = np.zeros(max_years); bal_y = np.zeros(max_years)
    for t in range(1, max_years + 1):
        a, b = (t-1)*12, min(t*12, n_m)
        if a < b:
            pay_y[t-1] = np.sum(pagos_m[a:b]); int_y[t-1] = np.sum(ints_m[a:b])
            amo_y[t-1] = np.sum(amort_m[a:b]); bal_y[t-1] = saldo_m[b-1]
    if horizonte <= max_years:
        return pay_y[:horizonte], int_y[:horizonte], amo_y[:horizonte], bal_y[:horizonte]
    pad = horizonte - max_years
    return np.pad(pay_y,(0,pad)), np.pad(int_y,(0,pad)), np.pad(amo_y,(0,pad)), np.pad(bal_y,(0,pad))

# Capitalización de intereses durante construcción (IDC)
def capitalized_principal(capex_actor: float, annual_rate: float, years_construction: int, share_year1: float) -> float:
    """Capex se desembolsa mensualmente: share_year1 en el 1er año y el resto en los siguientes.
    Intereses se capitalizan mes a mes sobre saldo."""
    months = max(1, int(years_construction) * 12)
    rm = max(float(annual_rate), 0.0) / 12.0
    months_y1 = min(12, months)
    months_rest = months - months_y1
    w1 = max(0.0, min(1.0, share_year1))
    disb_m_y1 = (capex_actor * w1) / months_y1
    disb_m_rest = (capex_actor * (1.0 - w1)) / max(1, months_rest)
    saldo = 0.0
    for i in range(months):
        disb = disb_m_y1 if i < months_y1 else disb_m_rest
        saldo += disb
        saldo += saldo * rm  # capitaliza interés del mes
    return float(saldo)

# IRR robusto (bisección)
def irr_bisection(cashflows, lo=-0.99, hi=5.0, tol=1e-6, maxit=200):
    def npv(r):
        return sum(cf / ((1+r)**t) for t, cf in enumerate(cashflows))
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(maxit):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return mid

# =========================
# PARÁMETROS FIJOS
# =========================
COSTO_BASE_POR_KM = 450_000_000   # 450 M por km
LONGITUD_KM = 10.1

# =========================
# LAYOUT base
# =========================
col_main, col_rail = st.columns([0.76, 0.24], gap="large")

with col_rail:
    st.markdown('<div class="rail-derecho">', unsafe_allow_html=True)
    st.subheader("Multiplicadores globales")
    st.caption("Aplican proporcionalmente sobre los supuestos base.")
    dem_x   = st.slider("Demanda (×)",      0.50, 1.50, st.session_state.get("dem_x",   1.00), 0.05, key="dem_x")
    capkm_x = st.slider("Costo por km (×)", 0.50, 1.50, st.session_state.get("capkm_x", 1.00), 0.05, key="capkm_x")
    opex_x  = st.slider("OPEX (×)",         0.50, 1.50, st.session_state.get("opex_x",  1.00), 0.05, key="opex_x")
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown("## 🚡 Modelo Cablebús — Tablero Financiero")
    T_INV, T_ING, T_OPEX, T_CAP, T_RES, T_GLOS = st.tabs(
        ["Inversión", "Ingresos", "Costos Operativos", "Estructura de Capital", "Resultados", "Glosario"]
    )

    # ----------------------- INVERSIÓN -----------------------
    with T_INV:
        st.markdown("### Inversión resumida (solo lo esencial)")
        st.write("Longitud del proyecto **10.1 km** (fija).")

        costo_km_aj = COSTO_BASE_POR_KM * capkm_x
        overrun_frac = percent_input("Sobrecosto por predios y otros (sobre CAPEX base)",
                                     key="overrun_pct", default_percent=10.0,
                                     help="Porcentaje sobre CAPEX base (costo/km ajustado × 10.1 km).")
        capex_base  = costo_km_aj * LONGITUD_KM
        sobrecosto  = capex_base * overrun_frac
        capex_total = capex_base + sobrecosto

        st.markdown("#### Construcción e IDC")
        cons_years = int(st.number_input("Años de construcción", min_value=1, max_value=4, value=2, step=1))
        share_y1   = st.slider("Distribución de CAPEX en Año 1 (%)", 30, 70, 50, 1) / 100.0
        include_idc = st.checkbox("Incluir **IDC** (capitalización de intereses durante construcción)", value=True,
                                  help="Calculado por actor según su tasa y la curva de desembolsos.")
        dsra_meses = int(st.number_input("**DSRA** — meses de servicio objetivo", min_value=0, max_value=12, value=6,
                                         help="Reserva de servicio de deuda; 0 para desactivar."))
        disc_rate  = percent_input("Tasa de descuento real para NPV", key="disc_rate", default_percent=10.0)

        # Persistimos para otras pestañas y exportación
        st.session_state["capex_total"] = float(capex_total)
        st.session_state["capex_base"]  = float(capex_base)
        st.session_state["costo_km_aj"] = float(costo_km_aj)
        st.session_state["cons_years"]  = cons_years
        st.session_state["share_y1"]    = share_y1
        st.session_state["include_idc"] = include_idc
        st.session_state["dsra_meses"]  = dsra_meses
        st.session_state["disc_rate"]   = disc_rate
        st.session_state["overrun_pct"] = overrun_frac * 100.0  # para exportar en %

        df_wide(pd.DataFrame({
            "Concepto": [
                "Costo por km (ajustado)", "Longitud (km)", "CAPEX base", "Sobrecosto (%)", "CAPEX total (sin IDC)"
            ],
            "Valor": [pesos(costo_km_aj), f"{LONGITUD_KM:.1f}", pesos(capex_base), f"{overrun_frac*100:.2f}%", pesos(capex_total)]
        }))

    # ----------------------- INGRESOS ------------------------
    with T_ING:
        st.markdown("### Supuestos de ingresos (por segmento)")
        c1, c2, c3 = st.columns(3)
        with c1:
            aforo_local = int_input_commas("Aforo diario — **Locales** (boletos)", "aforo_local", 19220)
            tarifa_local = money_input("Tarifa **Local** (MXN/boleto)", "tarifa_local", 11.0)
        with c2:
            aforo_tur   = int_input_commas("Aforo diario — **Turistas** (boletos)", "aforo_tur", 4845)
            tarifa_tur  = money_input("Tarifa **Turista** (MXN/boleto)", "tarifa_tur", 50.0)
        with c3:
            locales     = int_input_commas("Número de **establecimientos** en renta", "n_locales", 80)
            renta_mes   = money_input("Renta neta mensual por establecimiento", "renta_mes", 15_000.0)
        g_demand = percent_input("Crecimiento anual de **boletaje**", "g_demanda", 3.0)

        y1_local = (aforo_local * st.session_state["dem_x"]) * 365.0 * tarifa_local
        y1_tur   = (aforo_tur   * st.session_state["dem_x"]) * 365.0 * tarifa_tur
        y1_est   = locales * renta_mes * 12.0
        y1_total = y1_local + y1_tur + y1_est
        comp_ing_df = pd.DataFrame({"Nombre":["Usuarios Locales","Usuarios Turistas","Establecimientos"], "Monto":[y1_local,y1_tur,y1_est]})
        comp_ing_df["% del total"] = comp_ing_df["Monto"] / comp_ing_df["Monto"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Locales — Ingreso Año 1", f"{pesos(y1_local)}  ({comp_ing_df['% del total'][0]:.1%})")
        m2.metric("Turistas — Ingreso Año 1", f"{pesos(y1_tur)}  ({comp_ing_df['% del total'][1]:.1%})")
        m3.metric("Establecimientos — Año 1", f"{pesos(y1_est)}  ({comp_ing_df['% del total'][2]:.1%})")
        m4.metric("**Total ingreso Año 1**", pesos(y1_total))
        st.altair_chart(bar_comp(comp_ing_df, "Composición del ingreso (Año 1)"), use_container_width=True)

        st.session_state["ingresos_params"] = {
            "aforo_local": aforo_local, "tarifa_local": tarifa_local,
            "aforo_tur": aforo_tur, "tarifa_tur": tarifa_tur,
            "locales": locales, "renta_mes": renta_mes, "g_demand": g_demand
        }

    # ----------------------- OPEX ----------------------------
    with T_OPEX:
        st.markdown("### Costos Operativos (desglosados)")
        c1, c2, c3 = st.columns(3)
        with c1:
            opex_personal_y1 = money_input("Pago de **personal** (Año 1)", "opex_personal_y1", 20_000_000.0)
        with c2:
            opex_menor_y1    = money_input("**Mantenimiento menor** (Año 1)", "opex_menor_y1", 18_000_000.0)
        with c3:
            opex_mayor_monto = money_input("**Mantenimiento mayor** (monto por evento)", "opex_mayor_monto", 100_000_000.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            g_opex = percent_input("Crecimiento anual (personal + mant. menor)", "g_opex", 2.0)
        with c5:
            mayor_cada = int(st.number_input("Periodicidad de **mantenimiento mayor** (años)", min_value=5, max_value=30, value=10, step=1, key="mayor_cada"))
        with c6:
            use_provision = st.checkbox("Suavizar mant. mayor como **provisión anual equivalente**", value=True)

        y1_per = opex_personal_y1 * st.session_state["opex_x"]
        y1_men = opex_menor_y1   * st.session_state["opex_x"]
        y1_may_prov  = (opex_mayor_monto / max(1, mayor_cada)) * st.session_state["opex_x"]
        y1_ox_tot = y1_per + y1_men + (y1_may_prov if use_provision else 0.0)

        comp_ox_df = pd.DataFrame({
            "Nombre": ["Personal","Mant. menor","Mant. mayor" + (" (provisión)" if use_provision else " (evento)")],
            "Monto":  [y1_per, y1_men, (y1_may_prov if use_provision else 0.0)]
        })
        comp_ox_df["% del total"] = comp_ox_df["Monto"] / max(y1_ox_tot, 1e-9)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Personal — Año 1", f"{pesos(y1_per)}  ({comp_ox_df['% del total'][0]:.1%})")
        n2.metric("Mant. menor — Año 1", f"{pesos(y1_men)}  ({comp_ox_df['% del total'][1]:.1%})")
        n3.metric(("Mant. mayor (prov.)" if use_provision else "Mant. mayor (evento)") + " — Año 1",
                  f"{pesos(comp_ox_df['Monto'][2])}  ({comp_ox_df['% del total'][2]:.1%})")
        n4.metric("**OPEX total Año 1**", pesos(y1_ox_tot))
        st.altair_chart(bar_comp(comp_ox_df, "Composición de OPEX (Año 1)"), use_container_width=True)

        st.session_state["opex_params"] = {
            "personal_y1": opex_personal_y1, "menor_y1": opex_menor_y1,
            "mayor_monto": opex_mayor_monto, "g_opex": g_opex,
            "mayor_cada": mayor_cada, "use_provision": use_provision
        }

    # ------------------ ESTRUCTURA DE CAPITAL ----------------
    def _rebalance_public(changed: str):
        keys = ["fonatur", "fonadin", "shcp", "edo"]
        vals = {k: float(st.session_state.get(k, 25.0)) for k in keys}
        for k in keys:
            vals[k] = min(100.0, max(0.0, vals[k]))
        target_others = max(0.0, 100.0 - vals[changed])
        others = [k for k in keys if k != changed]
        others_sum = sum(vals[k] for k in others)
        if others_sum <= 1e-9:
            share = target_others / len(others) if len(others) else 0.0
            for k in others: vals[k] = share
        else:
            f = target_others / others_sum
            for k in others: vals[k] *= f
        for k in keys: st.session_state[k] = round(vals[k], 2)

    with T_CAP:
        st.markdown("### Bloques y reparto")
        publico = st.slider("Participación **pública** (base)", 0.0, 1.0, value=0.5, step=0.01, key="pub_pct")
        privado_base = 1.0 - publico
        st.write(f"**Público (base):** {publico:.0%}  |  **Privado (base):** {privado_base:.0%}")

        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("FONATUR (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonatur", on_change=_rebalance_public, args=("fonatur",))
        c2.number_input("BANOBRAS-FONADIN (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonadin", on_change=_rebalance_public, args=("fonadin",))
        c3.number_input("SHCP (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="shcp", on_change=_rebalance_public, args=("shcp",))
        c4.number_input("Gobierno del Estado (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="edo", on_change=_rebalance_public, args=("edo",))

        st.markdown("### Mecanismo de pago al privado")
        sweep_alpha = percent_input("Utilidad operativa destinada a **deuda privada**", "sweep_pct", 40.0,
                                    help="Porcentaje de la utilidad operativa anual (cash sweep).")
        allow_prepay = st.checkbox("Permitir **prepagos** si el sweep excede el pago programado", value=True)
        auto_fit_priv = st.checkbox("**Auto-ajustar** tamaño del crédito privado con base en cobertura del sweep (año crítico)", value=True)

        st.markdown("### Crédito por actor (tasa base + ajuste por plazo)")
        with st.expander("Tasa anual (% con punto) y plazo (años)."):
            slope = percent_input("Ajuste de tasa por plazo (p.p. por año)", key="slope_bps", default_percent=0.10,
                                  help="Ej.: 0.10 = +0.10 puntos porcentuales por cada año adicional.")
            d1, d2, d3, d4, d5 = st.columns(5)
            r_fon = percent_input("FONATUR — Tasa base", key="tasa_fon_base", default_percent=8.00);  n_fon = d1.number_input("FONATUR — Años", 1, 40, 15, 1, key="n_fon")
            r_ban = percent_input("BANOBRAS-FONADIN — Tasa base", key="tasa_ban_base", default_percent=9.50); n_ban = d2.number_input("BANOBRAS — Años", 1, 40, 20, 1, key="n_ban")
            r_shc = percent_input("SHCP — Tasa base", key="tasa_shcp_base", default_percent=9.00);   n_shc = d3.number_input("SHCP — Años", 1, 40, 18, 1, key="n_shc")
            r_edo = percent_input("Estado — Tasa base", key="tasa_edo_base", default_percent=11.00); n_edo = d4.number_input("Estado — Años", 1, 40, 15, 1, key="n_edo")
            r_pri = percent_input("Privado — Tasa base", key="tasa_pri_base", default_percent=14.00); n_pri = d5.number_input("Privado — Años", 1, 40, 12, 1, key="n_pri")

            r_fon_eff = r_fon + slope * n_fon
            r_ban_eff = r_ban + slope * n_ban
            r_shc_eff = r_shc + slope * n_shc
            r_edo_eff = r_edo + slope * n_edo
            r_pri_eff = r_pri + slope * n_pri

            st.session_state["rates_years"] = {
                "FONATUR": (r_fon_eff, n_fon),
                "BANOBRAS-FONADIN": (r_ban_eff, n_ban),
                "SHCP": (r_shc_eff, n_shc),
                "Gobierno del Estado": (r_edo_eff, n_edo),
                "Privado": (r_pri_eff, n_pri),
            }
        st.session_state["sweep_alpha"] = sweep_alpha
        st.session_state["allow_prepay"] = allow_prepay
        st.session_state["auto_fit_priv"] = auto_fit_priv

    # ------------------------ CÁLCULO ------------------------
    @st.cache_data(show_spinner=False)
    def calc_projection(horizonte: int, dem_x: float, opex_x: float,
                        ingresos_params: dict, opex_params: dict,
                        capex_total: float, publico_base: float, pub_weights: dict,
                        rates_years: dict, sweep_alpha: float, allow_prepay: bool, auto_fit_priv: bool,
                        cons_years: int, share_y1: float, include_idc: bool, dsra_meses: int):
        years = np.arange(1, horizonte + 1)

        # Ingresos
        ing_local_y1 = (ingresos_params["aforo_local"] * dem_x) * 365.0 * ingresos_params["tarifa_local"]
        ing_tur_y1   = (ingresos_params["aforo_tur"]   * dem_x) * 365.0 * ingresos_params["tarifa_tur"]
        ing_locals   = ingresos_params["locales"] * ingresos_params["renta_mes"] * 12.0
        g = ingresos_params["g_demand"]
        ing_local = np.array([ing_local_y1 * ((1 + g) ** (t - 1)) for t in years], dtype=float)
        ing_tur   = np.array([ing_tur_y1   * ((1 + g) ** (t - 1)) for t in years], dtype=float)
        ing_est   = np.full_like(ing_local, ing_locals, dtype=float)
        ingresos_totales = ing_local + ing_tur + ing_est

        # OPEX
        use_prov = bool(opex_params["use_provision"])
        g_ox = opex_params["g_opex"]; cada = max(1, int(opex_params["mayor_cada"]))
        per0 = opex_params["personal_y1"] * opex_x
        men0 = opex_params["menor_y1"]   * opex_x
        maym = opex_params["mayor_monto"]* opex_x
        opex_personal = np.array([per0 * ((1 + g_ox) ** (t - 1)) for t in years], dtype=float)
        opex_menor    = np.array([men0 * ((1 + g_ox) ** (t - 1)) for t in years], dtype=float)
        opex_mayor_evento = np.array([ (maym if (t % cada == 0) else 0.0) for t in years ], dtype=float)
        opex_mayor_prov   = np.full_like(opex_menor, (maym / cada), dtype=float)
        opex_total        = opex_personal + opex_menor + (opex_mayor_prov if use_prov else opex_mayor_evento)

        utilidad_op   = ingresos_totales - opex_total

        # Financiamiento base (sobre CAPEX total SIN IDC)
        capex = float(capex_total)
        privado_base = 1.0 - publico_base
        rf, nf = rates_years["FONATUR"]; rb, nb = rates_years["BANOBRAS-FONADIN"]
        rs, ns = rates_years["SHCP"];     re, ne = rates_years["Gobierno del Estado"]
        rp, npv= rates_years["Privado"]

        # Auto-ajuste tamaño privado (conservador): usa percentil 30 de utilidad
        if auto_fit_priv:
            util_target = np.percentile(utilidad_op, 30)
            pm_unit = annuity_payment_monthly(1.0, rp, npv)*12
            allowed_principal = max(0.0, min(capex, (sweep_alpha*util_target)/pm_unit))
            privado_frac = (allowed_principal / capex) if capex > 0 else 0.0
        else:
            privado_frac = privado_base

        contrib_privada = capex * privado_frac
        contrib_publica = capex - contrib_privada

        # IDC por actor (si aplica)
        def principal_con_idc(base, r, yrs):
            return capitalized_principal(base, r, cons_years, share_y1) if include_idc else base

        base_fon  = principal_con_idc(contrib_publica * pub_weights["fonatur"], rf, nf)
        base_ban  = principal_con_idc(contrib_publica * pub_weights["fonadin"], rb, nb)
        base_shcp = principal_con_idc(contrib_publica * pub_weights["shcp"],    rs, ns)
        base_edo  = principal_con_idc(contrib_publica * pub_weights["edo"],     re, ne)
        base_pri  = principal_con_idc(contrib_privada,                           rp, npv)

        idc_total = (base_fon + base_ban + base_shcp + base_edo + base_pri) - capex

        # Pagos programados (sin sweep)
        serv_fon, _, _, sal_fon = amort_yearly_from_monthly(base_fon,  rf, nf, horizonte)
        serv_ban, _, _, sal_ban = amort_yearly_from_monthly(base_ban,  rb, nb, horizonte)
        serv_shc, _, _, sal_shc = amort_yearly_from_monthly(base_shcp, rs, ns, horizonte)
        serv_edo, _, _, sal_edo = amort_yearly_from_monthly(base_edo,  re, ne, horizonte)
        serv_pri, _, _, sal_pri = amort_yearly_from_monthly(base_pri,  rp, npv, horizonte)

        serv_pub = serv_fon + serv_ban + serv_shc + serv_edo
        serv_tot_prog = serv_pub + serv_pri

        # DSRA objetivo y flujos de fondeo/liberación
        dsra_target = (dsra_meses/12.0) * serv_tot_prog if dsra_meses > 0 else np.zeros_like(serv_tot_prog)
        dsra_flow = np.zeros_like(dsra_target)
        dsra_bal  = np.zeros_like(dsra_target)
        if dsra_meses > 0:
            dsra_flow[0] = dsra_target[0]    # fondeo en Año 1 (al inicio de operaciones)
            dsra_bal[0]  = dsra_target[0]
            for t in range(1, len(dsra_target)):
                dsra_flow[t] = dsra_target[t] - dsra_bal[t-1]
                dsra_bal[t]  = dsra_target[t]

        # “Cash sweep” al privado (sobre utilidad operativa)
        serv_pri_sweep, _, _, sal_pri_sweep = simulate_private_sweep(
            base_pri, rp, npv, utilidad_op, sweep_alpha, allow_prepay, horizonte
        )

        # Flujos y métricas
        flujo_post_deuda_prog = utilidad_op - serv_tot_prog
        flujo_post_deuda_y_dsra = flujo_post_deuda_prog - dsra_flow
        dscr_prog = np.divide(utilidad_op, serv_tot_prog, out=np.full_like(utilidad_op, np.nan), where=serv_tot_prog>0)

        # Caja (series acumuladas)
        efectivo_op              = -capex + np.cumsum(utilidad_op)                                  # sin deuda
        efectivo_post_prog       = -capex - idc_total + np.cumsum(flujo_post_deuda_prog)            # con deuda programada + IDC
        efectivo_post_prog_dsra  = -capex - idc_total + np.cumsum(flujo_post_deuda_y_dsra)          # con DSRA integrado

        saldo_dict_prog = {
            "FONATUR": sal_fon, "BANOBRAS-FONADIN": sal_ban, "SHCP": sal_shc,
            "Gobierno del Estado": sal_edo, "Privado (programado)": sal_pri,
            "Privado (sweep)": sal_pri_sweep
        }

        # KPIs: NPV/IRR (anual)
        r = st.session_state.get("disc_rate", 0.10)
        cf_proj = np.concatenate(([-capex], utilidad_op))                         # Proyecto (sin deuda)
        cf_post = np.concatenate(([-capex - idc_total], utilidad_op - serv_tot_prog - dsra_flow))
        npv_proj = sum(cf_proj[t] / ((1+r)**t) for t in range(len(cf_proj)))
        npv_post = sum(cf_post[t] / ((1+r)**t) for t in range(len(cf_post)))
        irr_post = irr_bisection(cf_post)

        return {
            "years": years,
            "ing_local": ing_local, "ing_tur": ing_tur, "ing_est": ing_est,
            "ingresos_totales": ingresos_totales,
            "opex_personal": opex_personal, "opex_menor": opex_menor,
            "opex_mayor_evento": opex_mayor_evento, "opex_mayor_prov": opex_mayor_prov,
            "opex_total": opex_total,
            "utilidad_op": utilidad_op,
            "servicios_prog": {"pub": serv_pub, "pri": serv_pri, "tot": serv_tot_prog},
            "serv_pri_sweep": serv_pri_sweep,
            "saldo_por_actor": saldo_dict_prog,
            "dscr_prog": dscr_prog,
            "efectivo_op": efectivo_op,
            "efectivo_post_prog": efectivo_post_prog,
            "efectivo_post_prog_dsra": efectivo_post_prog_dsra,
            "idc_total": idc_total,
            "dsra_target": dsra_target, "dsra_flow": dsra_flow, "dsra_bal": dsra_bal,
            "npv_proj": npv_proj, "npv_post": npv_post, "irr_post": irr_post,
            "priv_frac_final": privado_frac
        }

    # ----------------------- RESULTADOS -----------------------
    with T_RES:
        st.markdown("### Horizonte y cálculo")
        horizonte = int(st.number_input("Horizonte (años)", min_value=5, value=20, step=1, key="horiz"))

        capex_total = float(st.session_state.get("capex_total", 0.0))
        publico_base = float(st.session_state.get("pub_pct", 0.5))
        pub_w = {
            "fonatur": float(st.session_state.get("fonatur", 25.0))/100.0,
            "fonadin": float(st.session_state.get("fonadin", 25.0))/100.0,
            "shcp":    float(st.session_state.get("shcp",    25.0))/100.0,
            "edo":     float(st.session_state.get("edo",     25.0))/100.0,
        }
        rates_years = st.session_state.get("rates_years", {
            "FONATUR": (0.08, 15), "BANOBRAS-FONADIN": (0.095, 20),
            "SHCP": (0.09, 18), "Gobierno del Estado": (0.11, 15), "Privado": (0.14, 12)
        })
        sweep_alpha = float(st.session_state.get("sweep_alpha", 0.40))
        allow_prepay = bool(st.session_state.get("allow_prepay", True))
        auto_fit_priv = bool(st.session_state.get("auto_fit_priv", True))
        cons_years = int(st.session_state.get("cons_years", 2))
        share_y1   = float(st.session_state.get("share_y1", 0.5))
        include_idc = bool(st.session_state.get("include_idc", True))
        dsra_meses = int(st.session_state.get("dsra_meses", 6))

        out = calc_projection(
            horizonte, float(st.session_state.get("dem_x",1.0)), float(st.session_state.get("opex_x",1.0)),
            st.session_state.get("ingresos_params", {"aforo_local":19220,"tarifa_local":11.0,"aforo_tur":4845,"tarifa_tur":50.0,"locales":80,"renta_mes":15000.0,"g_demand":0.03}),
            st.session_state.get("opex_params", {"personal_y1":20_000_000.0,"menor_y1":18_000_000.0,"mayor_monto":100_000_000.0,"g_opex":0.02,"mayor_cada":10,"use_provision":True}),
            capex_total, publico_base, pub_w, rates_years, sweep_alpha, allow_prepay, auto_fit_priv,
            cons_years, share_y1, include_idc, dsra_meses
        )
        years = out["years"]

        # Resumen ejecutivo
        y1_ing  = int(round(out["ingresos_totales"][0]))
        y1_ox   = int(round(out["opex_total"][0]))
        y1_uop  = int(round(out["utilidad_op"][0]))
        y1_deuda= int(round(out["servicios_prog"]["tot"][0]))
        y1_dscr = float(out["dscr_prog"][0]) if out["servicios_prog"]["tot"][0] > 0 else np.nan

        R1, R2, R3, R4, R5, R6 = st.columns(6)
        R1.metric("CAPEX total (sin IDC)", pesos(capex_total))
        R2.metric("Ingresos Año 1", pesos(y1_ing))
        R3.metric("OPEX Año 1", pesos(y1_ox))
        R4.metric("Utilidad operativa Año 1", pesos(y1_uop))
        R5.metric("Serv. deuda Año 1 (programado)", pesos(y1_deuda))
        R6.metric("DSCR Año 1 (programado)", f"{y1_dscr:.2f}×" if not np.isnan(y1_dscr) else "—")
        st.caption(f"**Participación privada final**: **{out['priv_frac_final']:.1%}** | **IDC estimado capitalizado:** {pesos(out['idc_total'])} | **DSRA objetivo:** {st.session_state['dsra_meses']} meses.")

        st.divider()
        SUB_OP, SUB_DEUDA, SUB_CAJA, SUB_VAL = st.tabs(["Operación", "Deuda & Financiamiento", "Caja & DSRA", "Valor (NPV/IRR)"])

        # Operación
        with SUB_OP:
            df_ing = pd.DataFrame({
                "Año": years,
                "Usuarios Locales": np.rint(out["ing_local"]).astype("int64"),
                "Usuarios Turistas": np.rint(out["ing_tur"]).astype("int64"),
                "Establecimientos": np.rint(out["ing_est"]).astype("int64")
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(df_ing, "Ingresos por segmento"), use_container_width=True)

            df_ox = pd.DataFrame({
                "Año": years,
                "Personal": np.rint(out["opex_personal"]).astype("int64"),
                "Mant. menor": np.rint(out["opex_menor"]).astype("int64"),
                "Mant. mayor (evento)": np.rint(out["opex_mayor_evento"]).astype("int64"),
                "Mant. mayor (provisión)": np.rint(out["opex_mayor_prov"]).astype("int64")
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(df_ox, "OPEX por componente"), use_container_width=True)

            df_u = pd.DataFrame({"Año": years, "Utilidad operativa": np.rint(out["utilidad_op"]).astype("int64")}) \
                .melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(df_u, "Utilidad operativa"), use_container_width=True)

        # Deuda
        with SUB_DEUDA:
            df_sd = pd.DataFrame({
                "Año": years,
                "Servicio de deuda (programado)": np.rint(out["servicios_prog"]["tot"]).astype("int64"),
                "Utilidad operativa": np.rint(out["utilidad_op"]).astype("int64")
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(df_sd, "Servicio de deuda programado vs Utilidad operativa"), use_container_width=True)

            saldo_df = pd.DataFrame({"Año": years, **{k: np.rint(v).astype("int64") for k,v in out["saldo_por_actor"].items()}}) \
                .melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(saldo_df, "Saldo de deuda por institución"), use_container_width=True)

        # Caja & DSRA
        with SUB_CAJA:
            deuda_total_prog = np.rint(
                np.sum(np.vstack([v for k,v in out["saldo_por_actor"].items() if "Privado (sweep)" not in k]), axis=0)
            ).astype("int64")
            deuda_total_con_sweep = np.rint(
                np.sum(np.vstack([
                    out["saldo_por_actor"]["FONATUR"],
                    out["saldo_por_actor"]["BANOBRAS-FONADIN"],
                    out["saldo_por_actor"]["SHCP"],
                    out["saldo_por_actor"]["Gobierno del Estado"],
                    out["saldo_por_actor"]["Privado (sweep)"]
                ]), axis=0)
            ).astype("int64")

            df_cash = pd.DataFrame({
                "Año": years,
                "Efectivo acum. (operativo)": np.rint(out["efectivo_op"]).astype("int64"),
                "Efectivo acum. (post-deuda)": np.rint(out["efectivo_post_prog"]).astype("int64"),
                "Efectivo acum. (post-deuda + DSRA)": np.rint(out["efectivo_post_prog_dsra"]).astype("int64"),
                "Deuda total (programada)": deuda_total_prog,
                "Deuda total (con sweep privado)": deuda_total_con_sweep
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(line_series(df_cash, "Caja y Deuda total (incluye DSRA)"), use_container_width=True)

            if st.session_state["dsra_meses"] > 0:
                dsra_df = pd.DataFrame({"Año": years, "Objetivo": np.rint(out["dsra_target"]).astype("int64"),
                                        "Flujo DSRA (top-up positivo)": np.rint(out["dsra_flow"]).astype("int64"),
                                        "Saldo DSRA": np.rint(out["dsra_bal"]).astype("int64")}) \
                          .melt(id_vars="Año", var_name="Serie", value_name="Valor")
                st.altair_chart(line_series(dsra_df, "Reserva DSRA — objetivo, flujos y saldo"), use_container_width=True)

        # Valor (NPV/IRR)
        with SUB_VAL:
            v1, v2, v3 = st.columns(3)
            v1.metric("NPV del Proyecto (sin deuda)", pesos(out["npv_proj"]))
            v2.metric("NPV Post-Deuda + DSRA", pesos(out["npv_post"]))
            v3.metric("IRR Post-Deuda", f"{out['irr_post']*100:.2f}%" if not np.isnan(out["irr_post"]) else "—")

        # -------- Exportación --------
        def build_export(sheets: dict[str, pd.DataFrame]) -> tuple[bytes, str, str]:
            engine = None
            try:
                import xlsxwriter  # noqa: F401
                engine = "xlsxwriter"
            except Exception:
                try:
                    import openpyxl  # noqa: F401
                    engine = "openpyxl"
                except Exception:
                    engine = None
            if engine:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine=engine) as writer:
                    for name, df in sheets.items():
                        df.to_excel(writer, sheet_name=name, index=False)
                return buf.getvalue(), "modelo_cablebus_financiero.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, df in sheets.items():
                    zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8"))
            return zbuf.getvalue(), "modelo_cablebus_financiero.zip", "application/zip"

        years_df = pd.DataFrame({"Año": years})
        inv_sheet = pd.DataFrame({
            "Variable": ["Costo por km (ajustado)", "Longitud (km)", "CAPEX base", "Sobrecosto (%)",
                         "CAPEX total (sin IDC)", "Años construcción", "Distribución Año 1 (%)", "IDC total capitalizado"],
            "Valor": [st.session_state.get("costo_km_aj", COSTO_BASE_POR_KM), LONGITUD_KM,
                      st.session_state.get("capex_base", 0.0), st.session_state.get("overrun_pct", 10.0),
                      st.session_state.get("capex_total", 0.0), st.session_state.get("cons_years", 2),
                      st.session_state.get("share_y1", 0.5)*100, out["idc_total"]]
        })
        ingresos_sheet = years_df.assign(
            Locales=out["ing_local"], Turistas=out["ing_tur"], Establecimientos=out["ing_est"], Total=out["ingresos_totales"]
        )
        opex_sheet = years_df.assign(
            Personal=out["opex_personal"], Mant_menor=out["opex_menor"],
            Mant_mayor_evento=out["opex_mayor_evento"], Mant_mayor_provision=out["opex_mayor_prov"],
            Total_usado=out["opex_total"]
        )
        operativo_sheet = years_df.assign(
            Ingresos=out["ingresos_totales"], OPEX=out["opex_total"], Utilidad_operativa=out["utilidad_op"]
        )
        deuda_prog_sheet = years_df.assign(
            Servicio_deuda_total_programado=out["servicios_prog"]["tot"],
            Servicio_privado_programado=out["servicios_prog"]["pri"],
            Servicio_privado_con_sweep=out["serv_pri_sweep"]
        )
        saldos_sheet = years_df.assign(**out["saldo_por_actor"])
        dsra_sheet = years_df.assign(DSRA_objetivo=out["dsra_target"], DSRA_flujo=out["dsra_flow"], DSRA_saldo=out["dsra_bal"])
        valor_sheet = pd.DataFrame({
            "KPI": ["NPV Proyecto (sin deuda)", "NPV Post-Deuda + DSRA", "IRR Post-Deuda", "Tasa descuento (real)"],
            "Valor": [out["npv_proj"], out["npv_post"], out["irr_post"], st.session_state.get("disc_rate", 0.10)]
        })

        data_bytes, fname, mime = build_export({
            "Inversion": inv_sheet,
            "Ingresos": ingresos_sheet,
            "OPEX": opex_sheet,
            "Operativo": operativo_sheet,
            "Deuda_programada_y_sweep": deuda_prog_sheet,
            "Saldos_por_institucion": saldos_sheet,
            "DSRA": dsra_sheet,
            "KPIs_valor": valor_sheet
        })
        st.download_button("⬇️ Descargar modelo (XLSX o ZIP)", data=data_bytes, file_name=fname, mime=mime)

    # -------------------- GLOSARIO / METODOLOGÍA --------------------
    with T_GLOS:
        st.markdown(textwrap.dedent("""
        ### Metodología (compacta)
        - **CAPEX** = costo/km ajustado × 10.1 + **Sobrecosto** (%).
        - **IDC**: se capitaliza mensualmente con curva de desembolso: % Año 1, resto subsecuente.
        - **Ingresos** = Locales + Turistas (crecen a *g*) + Establecimientos (renta fija).
        - **OPEX** = Personal + Mant. menor + Mant. mayor (**evento** cada *N* años o **provisión** = monto/N).
        - **Utilidad operativa** = Ingresos − OPEX (pesos constantes).
        - **Deuda**: anualidad a tasa efectiva por actor; **privado** permite *cash sweep* (α% de utilidad).
        - **DSRA**: objetivo de *m* meses del servicio programado. Se fondea al inicio y se ajusta anualmente.
        - **DSCR** (programado) = Utilidad operativa / Servicio programado.
        - **NPV**: flujo anual descontado con la tasa real indicada. **IRR**: sobre flujo post-deuda+DSRA.
        > Todas las gráficas son **reactivas** a cualquier cambio (ingresos, OPEX, IDC, DSRA, tasas, sweep).
        """))
