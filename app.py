# -*- coding: utf-8 -*-
# app.py — Modelo Cablebús (ejecutivo + robusto)
# Ejecutar:  streamlit run app.py

import io, zipfile, textwrap
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# --------------------------- CONFIG / ESTILO ---------------------------
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

# --------------------------- UTILIDADES ---------------------------
def _norm_num(s, default):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return float(default)

def money_input(label, key, default, help=None, decimals=2):
    initial = st.session_state.get(key, default)
    shown = f"{float(initial):,.{decimals}f}"
    txt = st.text_input(label, value=shown, key=f"{key}__txt", help=help)
    val = _norm_num(txt, default)
    st.session_state[key] = val
    return val

def int_input_commas(label, key, default, help=None):
    initial = int(st.session_state.get(key, default))
    txt = st.text_input(label, value=f"{initial:,}", key=f"{key}__txt", help=help)
    val = int(round(_norm_num(txt, default)))
    st.session_state[key] = val
    return val

def percent_input(label, key, default_percent, help=None, decimals=2):
    initial = st.session_state.get(key, default_percent)
    shown = f"{float(initial):.{decimals}f}"
    txt = st.text_input(f"{label} (%)", value=shown, key=f"{key}__pct", help=help)
    val_pct = _norm_num(txt, default_percent)
    st.session_state[key] = val_pct
    return max(0.0, float(val_pct)) / 100.0

def pesos(n):
    try:
        return f"${int(round(float(n))):,}"
    except Exception:
        return ""

def with_commas(n):
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return ""

def df_wide(df, **kw):
    try:
        return st.dataframe(df, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kw)

def chart_bar(df, title):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Nombre:N", sort="-y", axis=alt.Axis(labelFontSize=15, title=None)),
            y=alt.Y("Monto:Q", axis=alt.Axis(labelFontSize=15, title="Monto (MXN)", format=",")),
            tooltip=[
                alt.Tooltip("Nombre:N"),
                alt.Tooltip("Monto:Q", format=","),
                alt.Tooltip("% del total:Q", format=".1%"),
            ],
            color=alt.Color("Nombre:N", legend=None),
        )
        .properties(height=320, title=alt.TitleParams(title, fontSize=18))
    )

def chart_line(df_long, title):
    return (
        alt.Chart(df_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Año:O", axis=alt.Axis(labelFontSize=15, titleFontSize=16)),
            y=alt.Y("Valor:Q", axis=alt.Axis(labelFontSize=15, title=None, format=",")),
            color=alt.Color("Serie:N", legend=alt.Legend(title=None, labelFontSize=13)),
            tooltip=[
                alt.Tooltip("Año:O"),
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Valor:Q", format=","),
            ],
        )
        .properties(height=340, title=alt.TitleParams(title, fontSize=18))
    )

def chart_area_stacked(df_long, title):
    return (
        alt.Chart(df_long)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("Año:O", axis=alt.Axis(labelFontSize=15)),
            y=alt.Y("Valor:Q", stack=True, axis=alt.Axis(labelFontSize=15, title=None, format=",")),
            color=alt.Color("Serie:N", legend=alt.Legend(title=None, labelFontSize=13)),
            tooltip=[
                alt.Tooltip("Año:O"),
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Valor:Q", format=","),
            ],
        )
        .properties(height=340, title=alt.TitleParams(title, fontSize=18))
    )

def annuity_payment_monthly(P, annual_rate, years):
    P = max(float(P), 0.0)
    rm = max(float(annual_rate), 0.0) / 12.0
    m = max(int(years * 12), 1)
    if rm == 0:
        return P / m
    return P * rm / (1 - (1 + rm) ** (-m))

def amort_yearly_from_monthly(P, annual_rate, years, horizonte):
    P = float(P)
    n_m = int(max(years, 0) * 12)
    rm = float(max(annual_rate, 0.0) / 12.0)
    pm = annuity_payment_monthly(P, annual_rate, years) if n_m > 0 else 0.0
    pagos_m = np.zeros(n_m)
    ints_m = np.zeros(n_m)
    amort_m = np.zeros(n_m)
    saldo_m = np.zeros(n_m)
    saldo = P
    for i in range(n_m):
        interes = saldo * rm
        pago = pm
        amort = max(0.0, pago - interes)
        saldo = max(0.0, saldo + interes - pago)
        pagos_m[i] = pago
        ints_m[i] = interes
        amort_m[i] = amort
        saldo_m[i] = saldo
    max_y = max(horizonte, years)
    pay_y = np.zeros(max_y)
    int_y = np.zeros(max_y)
    amo_y = np.zeros(max_y)
    bal_y = np.zeros(max_y)
    for t in range(1, max_y + 1):
        a, b = (t - 1) * 12, min(t * 12, n_m)
        if a < b:
            pay_y[t - 1] = pagos_m[a:b].sum()
            int_y[t - 1] = ints_m[a:b].sum()
            amo_y[t - 1] = amort_m[a:b].sum()
            bal_y[t - 1] = saldo_m[b - 1]
    if horizonte <= max_y:
        return pay_y[:horizonte], int_y[:horizonte], amo_y[:horizonte], bal_y[:horizonte]
    pad = horizonte - max_y
    return (
        np.pad(pay_y, (0, pad)),
        np.pad(int_y, (0, pad)),
        np.pad(amo_y, (0, pad)),
        np.pad(bal_y, (0, pad)),
    )

def simulate_private_sweep(P, rate, years, utilidad_anual, sweep_alpha, allow_prepay, horizonte):
    n_m = int(max(years, 0) * 12)
    rm = float(max(rate, 0.0) / 12.0)
    pm = annuity_payment_monthly(P, rate, years) if n_m > 0 else 0.0
    saldo = float(P)
    pagos_m, ints_m, amort_m, saldo_m = [], [], [], []
    for i in range(n_m):
        y = min(len(utilidad_anual) - 1, i // 12)
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
        pagos_m.append(pago_real + extra)
        ints_m.append(interes)
        amort_m.append(amort)
        saldo_m.append(saldo)
    max_y = max(horizonte, years)
    pay_y = np.zeros(max_y)
    int_y = np.zeros(max_y)
    amo_y = np.zeros(max_y)
    bal_y = np.zeros(max_y)
    for t in range(1, max_y + 1):
        a, b = (t - 1) * 12, min(t * 12, n_m)
        if a < b:
            pay_y[t - 1] = np.sum(pagos_m[a:b])
            int_y[t - 1] = np.sum(ints_m[a:b])
            amo_y[t - 1] = np.sum(amort_m[a:b])
            bal_y[t - 1] = saldo_m[b - 1]
    if horizonte <= max_y:
        return pay_y[:horizonte], int_y[:horizonte], amo_y[:horizonte], bal_y[:horizonte]
    pad = horizonte - max_y
    return (
        np.pad(pay_y, (0, pad)),
        np.pad(int_y, (0, pad)),
        np.pad(amo_y, (0, pad)),
        np.pad(bal_y, (0, pad)),
    )

def capitalized_principal(capex_actor, annual_rate, years_construction, share_year1):
    months = max(1, int(years_construction) * 12)
    rm = max(float(annual_rate), 0.0) / 12.0
    m1 = min(12, months)
    mrest = months - m1
    w1 = max(0.0, min(1.0, share_year1))
    disb_y1 = (capex_actor * w1) / m1
    disb_rest = (capex_actor * (1.0 - w1)) / max(1, mrest)
    saldo = 0.0
    for i in range(months):
        saldo += disb_y1 if i < m1 else disb_rest
        saldo += saldo * rm
    return float(saldo)

def irr_bisection(cashflows, lo=-0.99, hi=5.0, tol=1e-6, maxit=200):
    def npv(r):
        return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(maxit):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return mid

# --------------------------- PARÁMETROS FIJOS ---------------------------
COSTO_BASE_POR_KM = 450_000_000
LONGITUD_KM = 10.1

# --------------------------- LAYOUT: RIEL Y TABS ---------------------------
col_main, col_rail = st.columns([0.76, 0.24], gap="large")

with col_rail:
    st.markdown('<div class="rail-derecho">', unsafe_allow_html=True)
    st.subheader("Multiplicadores globales")
    st.caption("Aplican proporcionalmente sobre supuestos base.")
    dem_x = st.slider("Demanda (×)", 0.50, 1.50, st.session_state.get("dem_x", 1.00), 0.05, key="dem_x")
    capkm_x = st.slider("Costo por km (×)", 0.50, 1.50, st.session_state.get("capkm_x", 1.00), 0.05, key="capkm_x")
    opex_x = st.slider("OPEX (×)", 0.50, 1.50, st.session_state.get("opex_x", 1.00), 0.05, key="opex_x")
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown("## 🚡 Modelo Cablebús — Tablero Financiero")
    T_INV, T_ING, T_OPEX, T_CAP, T_RES, T_GLOS = st.tabs(
        ["Inversión", "Ingresos", "Costos Operativos", "Estructura de Capital", "Resultados", "Glosario"]
    )

    # --------------------------- INVERSIÓN ---------------------------
    with T_INV:
        st.markdown("### Inversión resumida")
        st.write("Longitud fija **10.1 km**.")

        costo_km_aj = COSTO_BASE_POR_KM * capkm_x
        overrun = percent_input("Sobrecosto por predios y otros (sobre CAPEX base)", "overrun_pct", 10.0)
        capex_base = costo_km_aj * LONGITUD_KM
        capex_total = capex_base * (1 + overrun)

        cons_years = int(st.number_input("Años de construcción", 1, 4, 2, 1))
        share_y1 = st.slider("Distribución CAPEX en Año 1 (%)", 30, 70, 50, 1) / 100.0
        include_idc = st.checkbox("Incluir **IDC** (intereses capitalizados durante construcción)", value=True)
        dsra_meses = int(st.number_input("**DSRA** — meses de servicio objetivo", 0, 12, 6))
        disc_rate = percent_input("Tasa de descuento real (NPV)", "disc_rate", 10.0)

        st.session_state.update({
            "capex_total": float(capex_total),
            "capex_base": float(capex_base),
            "costo_km_aj": float(costo_km_aj),
            "cons_years": cons_years,
            "share_y1": share_y1,
            "include_idc": include_idc,
            "dsra_meses": dsra_meses,
            "disc_rate": disc_rate,
            "overrun_pct": overrun * 100.0,
        })

        df_wide(pd.DataFrame({
            "Concepto": ["Costo por km (ajustado)", "Longitud (km)", "CAPEX base", "Sobrecosto (%)", "CAPEX total (sin IDC)"],
            "Valor": [pesos(costo_km_aj), f"{LONGITUD_KM:.1f}", pesos(capex_base), f"{overrun*100:.2f}%", pesos(capex_total)],
        }))

    # --------------------------- INGRESOS ---------------------------
    with T_ING:
        st.markdown("### Supuestos de ingresos")
        c1, c2, c3 = st.columns(3)
        with c1:
            aforo_local = int_input_commas("Aforo diario — **Locales**", "aforo_local", 19220)
            tarifa_local = money_input("Tarifa **Local** (MXN)", "tarifa_local", 11.0)
        with c2:
            aforo_tur = int_input_commas("Aforo diario — **Turistas**", "aforo_tur", 4845)
            tarifa_tur = money_input("Tarifa **Turista** (MXN)", "tarifa_tur", 50.0)
        with c3:
            locales = int_input_commas("**Establecimientos** en renta", "n_locales", 80)
            renta_mes = money_input("Renta neta mensual por establec.", "renta_mes", 15_000.0)
        g_demand = percent_input("Crecimiento anual de **boletaje**", "g_demanda", 3.0)

        y1_local = (aforo_local * dem_x) * 365.0 * tarifa_local
        y1_tur = (aforo_tur * dem_x) * 365.0 * tarifa_tur
        y1_est = locales * renta_mes * 12.0

        comp_ing_df = pd.DataFrame({
            "Nombre": ["Usuarios Locales", "Usuarios Turistas", "Establecimientos"],
            "Monto": [y1_local, y1_tur, y1_est],
        })
        comp_ing_df["% del total"] = comp_ing_df["Monto"] / comp_ing_df["Monto"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Locales — Año 1", f"{pesos(y1_local)} ({comp_ing_df['% del total'][0]:.1%})")
        m2.metric("Turistas — Año 1", f"{pesos(y1_tur)} ({comp_ing_df['% del total'][1]:.1%})")
        m3.metric("Establecimientos — Año 1", f"{pesos(y1_est)} ({comp_ing_df['% del total'][2]:.1%})")
        m4.metric("**Ingreso total Año 1**", pesos(y1_local + y1_tur + y1_est))
        st.altair_chart(chart_bar(comp_ing_df, "Composición del ingreso (Año 1)"), use_container_width=True)

        st.session_state["ingresos_params"] = {
            "aforo_local": aforo_local,
            "tarifa_local": tarifa_local,
            "aforo_tur": aforo_tur,
            "tarifa_tur": tarifa_tur,
            "locales": locales,
            "renta_mes": renta_mes,
            "g_demand": g_demand,
        }

    # --------------------------- OPEX ---------------------------
    with T_OPEX:
        st.markdown("### Costos Operativos")
        c1, c2, c3 = st.columns(3)
        with c1:
            opex_personal_y1 = money_input("**Personal** (Año 1)", "opex_personal_y1", 20_000_000.0)
        with c2:
            opex_menor_y1 = money_input("**Mant. menor** (Año 1)", "opex_menor_y1", 18_000_000.0)
        with c3:
            opex_mayor_monto = money_input("**Mant. mayor** (por evento)", "opex_mayor_monto", 100_000_000.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            g_opex = percent_input("Crecimiento anual (personal+menor)", "g_opex", 2.0)
        with c5:
            mayor_cada = int(st.number_input("Periodicidad **mant. mayor** (años)", 5, 30, 10, 1, key="mayor_cada"))
        with c6:
            use_provision = st.checkbox("Suavizar mant. mayor como **provisión anual**", value=True)

        y1_per = opex_personal_y1 * opex_x
        y1_men = opex_menor_y1 * opex_x
        y1_may_prov = (opex_mayor_monto / mayor_cada) * opex_x

        comp_ox_df = pd.DataFrame({
            "Nombre": ["Personal", "Mant. menor", "Mant. mayor" + (" (prov.)" if use_provision else " (evento)")],
            "Monto": [y1_per, y1_men, (y1_may_prov if use_provision else 0.0)],
        })
        total_ox = y1_per + y1_men + (y1_may_prov if use_provision else 0.0)
        comp_ox_df["% del total"] = comp_ox_df["Monto"] / max(total_ox, 1e-9)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Personal — Año 1", f"{pesos(y1_per)} ({comp_ox_df['% del total'][0]:.1%})")
        n2.metric("Mant. menor — Año 1", f"{pesos(y1_men)} ({comp_ox_df['% del total'][1]:.1%})")
        n3.metric(
            ("Mant. mayor (prov.)" if use_provision else "Mant. mayor (evento)") + " — Año 1",
            f"{pesos(comp_ox_df['Monto'][2])} ({comp_ox_df['% del total'][2]:.1%})",
        )
        n4.metric("**OPEX total Año 1**", pesos(total_ox))
        st.altair_chart(chart_bar(comp_ox_df, "Composición de OPEX (Año 1)"), use_container_width=True)

        st.session_state["opex_params"] = {
            "personal_y1": opex_personal_y1,
            "menor_y1": opex_menor_y1,
            "mayor_monto": opex_mayor_monto,
            "g_opex": g_opex,
            "mayor_cada": mayor_cada,
            "use_provision": use_provision,
        }

    # --------------------------- CAPITAL ---------------------------
    def _rebalance_public(changed):
        keys = ["fonatur", "fonadin", "shcp", "edo"]
        vals = {k: float(st.session_state.get(k, 25.0)) for k in keys}
        for k in keys:
            vals[k] = min(100.0, max(0.0, vals[k]))
        target = max(0.0, 100.0 - vals[changed])
        others = [k for k in keys if k != changed]
        s = sum(vals[k] for k in others)
        if s <= 1e-9:
            share = target / len(others)
            for k in others:
                st.session_state[k] = round(share, 2)
        else:
            f = target / s
            for k in others:
                st.session_state[k] = round(vals[k] * f, 2)

    with T_CAP:
        st.markdown("### Bloques y reparto público")
        publico = st.slider("Participación **pública** (base)", 0.0, 1.0, value=0.5, step=0.01, key="pub_pct")
        st.write(f"**Público:** {publico:.0%} | **Privado:** {1.0 - publico:.0%}")

        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("FONATUR (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonatur",
                        on_change=_rebalance_public, args=("fonatur",))
        c2.number_input("BANOBRAS-FONADIN (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonadin",
                        on_change=_rebalance_public, args=("fonadin",))
        c3.number_input("SHCP (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="shcp",
                        on_change=_rebalance_public, args=("shcp",))
        c4.number_input("Gobierno del Estado (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="edo",
                        on_change=_rebalance_public, args=("edo",))

        st.markdown("### Pago al privado")
        sweep_alpha = percent_input("Utilidad destinada a **deuda privada**", "sweep_pct", 40.0)
        allow_prepay = st.checkbox("Permitir **prepagos** si el sweep excede el pago programado", value=True)
        auto_fit_priv = st.checkbox("**Auto-ajustar** tamaño del crédito privado según cobertura del sweep", value=True)

        st.markdown("### Tasas y plazos (tasa base + slope por plazo)")
        with st.expander("Parámetros por actor"):
            slope = percent_input("Ajuste de tasa por plazo (p.p. por año)", "slope_bps", 0.10,
                                  help="Ej. 0.10 = +0.10 puntos porcentuales por cada año adicional.")
            d1, d2, d3, d4, d5 = st.columns(5)
            r_fon = percent_input("FONATUR — Tasa base", "tasa_fon_base", 8.00)
            n_fon = d1.number_input("FONATUR — Años", 1, 40, 15, 1, key="n_fon")
            r_ban = percent_input("BANOBRAS-FONADIN — Tasa base", "tasa_ban_base", 9.50)
            n_ban = d2.number_input("BANOBRAS — Años", 1, 40, 20, 1, key="n_ban")
            r_shc = percent_input("SHCP — Tasa base", "tasa_shcp_base", 9.00)
            n_shc = d3.number_input("SHCP — Años", 1, 40, 18, 1, key="n_shc")
            r_edo = percent_input("Estado — Tasa base", "tasa_edo_base", 11.00)
            n_edo = d4.number_input("Estado — Años", 1, 40, 15, 1, key="n_edo")
            r_pri = percent_input("Privado — Tasa base", "tasa_pri_base", 14.00)
            n_pri = d5.number_input("Privado — Años", 1, 40, 12, 1, key="n_pri")
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
        st.session_state.update({
            "sweep_alpha": sweep_alpha,
            "allow_prepay": allow_prepay,
            "auto_fit_priv": auto_fit_priv,
        })

    # --------------------------- CÁLCULO ---------------------------
    @st.cache_data(show_spinner=False)
    def calc_projection(horizonte, dem_x, opex_x, ingresos_params, opex_params, capex_total,
                        publico_base, pub_weights, rates_years, sweep_alpha, allow_prepay, auto_fit_priv,
                        cons_years, share_y1, include_idc, dsra_meses):

        years = np.arange(1, horizonte + 1)

        # Ingresos
        ing_local_y1 = (ingresos_params["aforo_local"] * dem_x) * 365.0 * ingresos_params["tarifa_local"]
        ing_tur_y1 = (ingresos_params["aforo_tur"] * dem_x) * 365.0 * ingresos_params["tarifa_tur"]
        ing_est_y = ingresos_params["locales"] * ingresos_params["renta_mes"] * 12.0
        g = ingresos_params["g_demand"]
        ing_local = np.array([ing_local_y1 * ((1 + g) ** (t - 1)) for t in years])
        ing_tur = np.array([ing_tur_y1 * ((1 + g) ** (t - 1)) for t in years])
        ing_est = np.full_like(ing_local, ing_est_y)
        ingresos_tot = ing_local + ing_tur + ing_est

        # OPEX
        per0 = opex_params["personal_y1"] * opex_x
        men0 = opex_params["menor_y1"] * opex_x
        may = opex_params["mayor_monto"] * opex_x
        g_ox = opex_params["g_opex"]
        cad = max(1, int(opex_params["mayor_cada"]))
        use_prov = bool(opex_params["use_provision"])

        opex_personal = np.array([per0 * ((1 + g_ox) ** (t - 1)) for t in years])
        opex_menor = np.array([men0 * ((1 + g_ox) ** (t - 1)) for t in years])
        opex_may_evento = np.array([(may if (t % cad == 0) else 0.0) for t in years])
        opex_may_prov = np.full_like(opex_menor, may / cad)
        opex_tot = opex_personal + opex_menor + (opex_may_prov if use_prov else opex_may_evento)

        utilidad = ingresos_tot - opex_tot

        # Financiamiento
        capex = float(capex_total)
        privado_base = 1.0 - publico_base
        rf, nf = rates_years["FONATUR"]
        rb, nb = rates_years["BANOBRAS-FONADIN"]
        rs, ns = rates_years["SHCP"]
        re, ne = rates_years["Gobierno del Estado"]
        rp, npv = rates_years["Privado"]

        if auto_fit_priv:
            util_target = np.percentile(utilidad, 30)
            pm_unit = annuity_payment_monthly(1.0, rp, npv) * 12
            allowed_principal = max(0.0, min(capex, (sweep_alpha * util_target) / pm_unit))
            priv_frac = (allowed_principal / capex) if capex > 0 else 0.0
        else:
            priv_frac = privado_base

        contrib_priv = capex * priv_frac
        contrib_pub = capex - contrib_priv

        def P_idc(base, r):
            return capitalized_principal(base, r, cons_years, share_y1) if include_idc else base

        base_fon = P_idc(contrib_pub * pub_weights["fonatur"], rf)
        base_ban = P_idc(contrib_pub * pub_weights["fonadin"], rb)
        base_shc = P_idc(contrib_pub * pub_weights["shcp"], rs)
        base_edo = P_idc(contrib_pub * pub_weights["edo"], re)
        base_pri = P_idc(contrib_priv, rp)
        idc_total = (base_fon + base_ban + base_shc + base_edo + base_pri) - capex

        serv_fon, _, _, sal_fon = amort_yearly_from_monthly(base_fon, rf, nf, horizonte)
        serv_ban, _, _, sal_ban = amort_yearly_from_monthly(base_ban, rb, nb, horizonte)
        serv_shc, _, _, sal_shc = amort_yearly_from_monthly(base_shc, rs, ns, horizonte)
        serv_edo, _, _, sal_edo = amort_yearly_from_monthly(base_edo, re, ne, horizonte)
        serv_pri, _, _, sal_pri = amort_yearly_from_monthly(base_pri, rp, npv, horizonte)
        serv_pub = serv_fon + serv_ban + serv_shc + serv_edo
        serv_tot = serv_pub + serv_pri

        # DSRA
        dsra_target = (dsra_meses / 12.0) * serv_tot if dsra_meses > 0 else np.zeros_like(serv_tot)
        dsra_flow = np.zeros_like(dsra_target)
        dsra_bal = np.zeros_like(dsra_target)
        if dsra_meses > 0:
            dsra_flow[0] = dsra_target[0]
            dsra_bal[0] = dsra_target[0]
            for t in range(1, len(dsra_target)):
                dsra_flow[t] = dsra_target[t] - dsra_bal[t - 1]
                dsra_bal[t] = dsra_target[t]

        # Sweep privado
        serv_pri_sweep, _, _, sal_pri_sweep = simulate_private_sweep(
            base_pri, rp, npv, utilidad, sweep_alpha, allow_prepay, horizonte
        )

        # Coberturas
        safe_div = lambda num, den: np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        dscr_tot = safe_div(utilidad, serv_tot)
        cov_fon = safe_div(utilidad, serv_fon)
        cov_ban = safe_div(utilidad, serv_ban)
        cov_shc = safe_div(utilidad, serv_shc)
        cov_edo = safe_div(utilidad, serv_edo)
        cov_pri = safe_div(utilidad, serv_pri)

        cover_total = float(np.sum(np.minimum(utilidad, serv_tot)) / max(np.sum(serv_tot), 1e-9))
        cover_by_actor = {
            "FONATUR": float(np.sum(np.minimum(utilidad, serv_fon)) / max(np.sum(serv_fon), 1e-9)),
            "BANOBRAS-FONADIN": float(np.sum(np.minimum(utilidad, serv_ban)) / max(np.sum(serv_ban), 1e-9)),
            "SHCP": float(np.sum(np.minimum(utilidad, serv_shc)) / max(np.sum(serv_shc), 1e-9)),
            "Gobierno del Estado": float(np.sum(np.minimum(utilidad, serv_edo)) / max(np.sum(serv_edo), 1e-9)),
            "Privado": float(np.sum(np.minimum(utilidad, serv_pri)) / max(np.sum(serv_pri), 1e-9)),
        }

        # Caja
        efectivo_op = -capex + np.cumsum(utilidad)
        efectivo_post_prog = -capex - idc_total + np.cumsum(utilidad - serv_tot)
        efectivo_post_prog_dsra = -capex - idc_total + np.cumsum(utilidad - serv_tot - dsra_flow)

        saldo_por_actor = {
            "FONATUR": sal_fon,
            "BANOBRAS-FONADIN": sal_ban,
            "SHCP": sal_shc,
            "Gobierno del Estado": sal_edo,
            "Privado (programado)": sal_pri,
            "Privado (sweep)": sal_pri_sweep,
        }

        r = st.session_state.get("disc_rate", 0.10)
        cf_proj = np.concatenate(([-capex], utilidad))
        cf_post = np.concatenate(([-capex - idc_total], utilidad - serv_tot - dsra_flow))
        npv_proj = sum(cf_proj[t] / ((1 + r) ** t) for t in range(len(cf_proj)))
        npv_post = sum(cf_post[t] / ((1 + r) ** t) for t in range(len(cf_post)))
        irr_post = irr_bisection(cf_post)

        return {
            "years": years,
            "ing_local": ing_local,
            "ing_tur": ing_tur,
            "ing_est": ing_est,
            "ing_tot": ingresos_tot,
            "ox_personal": opex_personal,
            "ox_menor": opex_menor,
            "ox_may_evento": opex_may_evento,
            "ox_may_prov": opex_may_prov,
            "ox_tot": opex_tot,
            "uop": utilidad,
            "serv": {
                "FONATUR": serv_fon,
                "BANOBRAS-FONADIN": serv_ban,
                "SHCP": serv_shc,
                "Gobierno del Estado": serv_edo,
                "Privado": serv_pri,
                "Total": serv_tot,
            },
            "serv_pri_sweep": serv_pri_sweep,
            "saldo": saldo_por_actor,
            "dscr_tot": dscr_tot,
            "cov_actor_series": {
                "FONATUR": cov_fon,
                "BANOBRAS-FONADIN": cov_ban,
                "SHCP": cov_shc,
                "Gobierno del Estado": cov_edo,
                "Privado": cov_pri,
            },
            "cover_total": cover_total,
            "cover_by_actor": cover_by_actor,
            "ef_op": efectivo_op,
            "ef_post": efectivo_post_prog,
            "ef_post_dsra": efectivo_post_prog_dsra,
            "idc_total": idc_total,
            "dsra_target": dsra_target,
            "dsra_flow": dsra_flow,
            "dsra_bal": dsra_bal,
            "npv_proj": npv_proj,
            "npv_post": npv_post,
            "irr_post": irr_post,
            "priv_frac_final": priv_frac,
        }

    # --------------------------- RESULTADOS ---------------------------
    with T_RES:
        st.markdown("### Horizonte y cálculo")
        H = int(st.number_input("Horizonte (años)", 5, 40, 20, 1, key="horiz"))

        capex_total = float(st.session_state.get("capex_total", 0.0))
        publico = float(st.session_state.get("pub_pct", 0.5))
        pub_w = {
            "fonatur": st.session_state.get("fonatur", 25.0) / 100.0,
            "fonadin": st.session_state.get("fonadin", 25.0) / 100.0,
            "shcp": st.session_state.get("shcp", 25.0) / 100.0,
            "edo": st.session_state.get("edo", 25.0) / 100.0,
        }
        rates = st.session_state.get(
            "rates_years",
            {
                "FONATUR": (0.08, 15),
                "BANOBRAS-FONADIN": (0.095, 20),
                "SHCP": (0.09, 18),
                "Gobierno del Estado": (0.11, 15),
                "Privado": (0.14, 12),
            },
        )
        out = calc_projection(
            H,
            st.session_state.get("dem_x", 1.0),
            st.session_state.get("opex_x", 1.0),
            st.session_state.get(
                "ingresos_params",
                {
                    "aforo_local": 19220,
                    "tarifa_local": 11.0,
                    "aforo_tur": 4845,
                    "tarifa_tur": 50.0,
                    "locales": 80,
                    "renta_mes": 15000.0,
                    "g_demand": 0.03,
                },
            ),
            st.session_state.get(
                "opex_params",
                {
                    "personal_y1": 20_000_000.0,
                    "menor_y1": 18_000_000.0,
                    "mayor_monto": 100_000_000.0,
                    "g_opex": 0.02,
                    "mayor_cada": 10,
                    "use_provision": True,
                },
            ),
            capex_total,
            publico,
            pub_w,
            rates,
            st.session_state.get("sweep_alpha", 0.40),
            st.session_state.get("allow_prepay", True),
            st.session_state.get("auto_fit_priv", True),
            st.session_state.get("cons_years", 2),
            st.session_state.get("share_y1", 0.5),
            st.session_state.get("include_idc", True),
            st.session_state.get("dsra_meses", 6),
        )
        years = out["years"]

        # Executive strip
        y1_ing = int(out["ing_tot"][0])
        y1_ox = int(out["ox_tot"][0])
        y1_uop = int(out["uop"][0])
        y1_srv = int(out["serv"]["Total"][0])
        y1_dscr = float(out["dscr_tot"][0]) if out["serv"]["Total"][0] > 0 else np.nan
        R1, R2, R3, R4, R5, R6 = st.columns(6)
        R1.metric("CAPEX total (sin IDC)", pesos(capex_total))
        R2.metric("Ingresos Año 1", pesos(y1_ing))
        R3.metric("OPEX Año 1", pesos(y1_ox))
        R4.metric("Utilidad operativa Año 1", pesos(y1_uop))
        R5.metric("Servicio deuda Año 1", pesos(y1_srv))
        R6.metric("DSCR Año 1", f"{y1_dscr:.2f}×" if not np.isnan(y1_dscr) else "—")
        st.caption(
            f"**Privado final**: {out['priv_frac_final']:.1%} | **IDC capitalizado**: {pesos(out['idc_total'])} | **DSRA objetivo**: {st.session_state['dsra_meses']} meses."
        )

        st.divider()
        SUB_OP, SUB_DEUDA, SUB_CAJA, SUB_VAL = st.tabs(["Operación", "Deuda & Cobertura", "Caja & DSRA", "Valor (NPV/IRR)"])

        # -------- Operación
        with SUB_OP:
            df_ing = pd.DataFrame({
                "Año": years,
                "Usuarios Locales": np.rint(out["ing_local"]).astype("int64"),
                "Usuarios Turistas": np.rint(out["ing_tur"]).astype("int64"),
                "Establecimientos": np.rint(out["ing_est"]).astype("int64"),
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(chart_line(df_ing, "Ingresos por segmento"), use_container_width=True)

            df_ox = pd.DataFrame({
                "Año": years,
                "Personal": np.rint(out["ox_personal"]).astype("int64"),
                "Mant. menor": np.rint(out["ox_menor"]).astype("int64"),
                "Mant. mayor (evento)": np.rint(out["ox_may_evento"]).astype("int64"),
                "Mant. mayor (provisión)": np.rint(out["ox_may_prov"]).astype("int64"),
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(chart_line(df_ox, "OPEX por componente"), use_container_width=True)

            df_u = pd.DataFrame({
                "Año": years,
                "Utilidad operativa": np.rint(out["uop"]).astype("int64"),
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(chart_line(df_u, "Utilidad operativa"), use_container_width=True)

        # -------- Deuda & Cobertura
        with SUB_DEUDA:
            serv_actor_df = pd.DataFrame({
                "Año": years,
                "FONATUR": np.rint(out["serv"]["FONATUR"]).astype("int64"),
                "BANOBRAS-FONADIN": np.rint(out["serv"]["BANOBRAS-FONADIN"]).astype("int64"),
                "SHCP": np.rint(out["serv"]["SHCP"]).astype("int64"),
                "Gobierno del Estado": np.rint(out["serv"]["Gobierno del Estado"]).astype("int64"),
                "Privado (programado)": np.rint(out["serv"]["Privado"]).astype("int64"),
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(
                chart_area_stacked(serv_actor_df, "Servicio de deuda por institución (programado)"),
                use_container_width=True,
            )
            util_layer = alt.Chart(pd.DataFrame({
                "Año": years, "Serie": ["Utilidad operativa"] * len(years), "Valor": np.rint(out["uop"]).astype("int64")
            })).mark_line(strokeDash=[6, 4], strokeWidth=3).encode(
                x="Año:O", y=alt.Y("Valor:Q", axis=alt.Axis(format=",")), color=alt.value("black")
            )
            st.altair_chart(util_layer, use_container_width=True)

            saldo_df = pd.DataFrame({"Año": years, **{k: np.rint(v).astype("int64") for k, v in out["saldo"].items()}})\
                .melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(chart_line(saldo_df, "Saldo de deuda por institución (fin de año)"), use_container_width=True)

            cov_series_df = pd.DataFrame({"Año": years, "Total (DSCR)": out["dscr_tot"]})\
                .melt(id_vars="Año", var_name="Serie", value_name="Valor")
            for k, v in out["cov_actor_series"].items():
                cov_series_df = pd.concat([
                    cov_series_df,
                    pd.DataFrame({"Año": years, "Serie": [f"Cov. {k}"] * len(years), "Valor": v})
                ], ignore_index=True)
            cov_series_df["Valor"] = np.where(cov_series_df["Valor"] > 2.5, 2.5, cov_series_df["Valor"])
            st.altair_chart(chart_line(cov_series_df, "Cobertura: Utilidad / Servicio (cap 2.5× para lectura)"),
                            use_container_width=True)

            tot_coverage = out["cover_total"]
            cov_y1 = float(out["uop"][0] / out["serv"]["Total"][0]) if out["serv"]["Total"][0] > 0 else np.nan
            cov_min = float(np.nanmin(out["dscr_tot"]))
            idx_crit = int(np.nanargmin(out["dscr_tot"])) if np.any(~np.isnan(out["dscr_tot"])) else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cobertura total del horizonte", f"{tot_coverage * 100:.1f}%")
            c2.metric("Cobertura Año 1", f"{(cov_y1 * 100):.1f}%" if not np.isnan(cov_y1) else "—")
            c3.metric("Cobertura mínima anual", f"{cov_min:.2f}×")
            c4.metric("Año más exigente (mín DSCR)", f"{years[idx_crit]}")

            def payoff_year(saldo):
                idx = np.where(np.array(saldo) > 0)[0]
                if len(idx) == 0:
                    return 0
                last = idx[-1] + 1
                return last if saldo[last - 1] > 0 else max(1, last)

            resumen_actor = []
            for actor, key in [
                ("FONATUR", "FONATUR"),
                ("BANOBRAS-FONADIN", "BANOBRAS-FONADIN"),
                ("SHCP", "SHCP"),
                ("Gobierno del Estado", "Gobierno del Estado"),
                ("Privado (prog.)", "Privado (programado)"),
                ("Privado (sweep)", "Privado (sweep)"),
            ]:
                sal = out["saldo"][key]
                if "Privado (prog.)" in actor:
                    serv_vec = out["serv"]["Privado"]
                    cover = out["cover_by_actor"]["Privado"]
                elif "Privado (sweep)" in actor:
                    serv_vec = out["serv"]["Privado"]
                    cover = out["cover_by_actor"]["Privado"]
                else:
                    serv_vec = out["serv"].get(actor, np.zeros_like(years))
                    cover = out["cover_by_actor"].get(actor, np.nan)

                resumen_actor.append({
                    "Actor": actor,
                    "Servicio total (MXN)": int(np.sum(serv_vec)) if np.size(serv_vec) > 0 else 0,
                    "Cobertura posible con utilidad": f"{(cover * 100):.1f}%" if not np.isnan(cover) else "—",
                    "Año de liquidación (fin de saldo)": payoff_year(sal) if np.any(sal > 0) else 0,
                })
            df_act = pd.DataFrame(resumen_actor)
            df_act["Servicio total (MXN)"] = df_act["Servicio total (MXN)"].map(with_commas)
            df_wide(df_act)

        # -------- Caja & DSRA
        with SUB_CAJA:
            deuda_prog_total = np.rint(np.sum(np.vstack([
                out["saldo"]["FONATUR"],
                out["saldo"]["BANOBRAS-FONADIN"],
                out["saldo"]["SHCP"],
                out["saldo"]["Gobierno del Estado"],
                out["saldo"]["Privado (programado)"],
            ]), axis=0)).astype("int64")
            deuda_sweep_total = np.rint(np.sum(np.vstack([
                out["saldo"]["FONATUR"],
                out["saldo"]["BANOBRAS-FONADIN"],
                out["saldo"]["SHCP"],
                out["saldo"]["Gobierno del Estado"],
                out["saldo"]["Privado (sweep)"],
            ]), axis=0)).astype("int64")

            df_cash = pd.DataFrame({
                "Año": years,
                "Efectivo (operativo)": np.rint(out["ef_op"]).astype("int64"),
                "Efectivo (post-deuda)": np.rint(out["ef_post"]).astype("int64"),
                "Efectivo (post-deuda+DSRA)": np.rint(out["ef_post_dsra"]).astype("int64"),
                "Deuda total (programada)": deuda_prog_total,
                "Deuda total (con sweep privado)": deuda_sweep_total,
            }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
            st.altair_chart(chart_line(df_cash, "Caja y Deuda total"), use_container_width=True)

            if st.session_state["dsra_meses"] > 0:
                dsra_df = pd.DataFrame({
                    "Año": years,
                    "Objetivo": np.rint(out["dsra_target"]).astype("int64"),
                    "Flujo DSRA": np.rint(out["dsra_flow"]).astype("int64"),
                    "Saldo DSRA": np.rint(out["dsra_bal"]).astype("int64"),
                }).melt(id_vars="Año", var_name="Serie", value_name="Valor")
                st.altair_chart(chart_line(dsra_df, "Reserva DSRA — objetivo, flujos y saldo"), use_container_width=True)

        # -------- Valor (NPV/IRR)
        with SUB_VAL:
            v1, v2, v3 = st.columns(3)
            v1.metric("NPV Proyecto (sin deuda)", pesos(out["npv_proj"]))
            v2.metric("NPV Post-Deuda + DSRA", pesos(out["npv_post"]))
            v3.metric("IRR Post-Deuda", f"{out['irr_post'] * 100:.2f}%" if not np.isnan(out["irr_post"]) else "—")

        # -------- Exportación --------
        def build_export(sheets):
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
                with pd.ExcelWriter(buf, engine=engine) as w:
                    for name, df in sheets.items():
                        df.to_excel(w, sheet_name=name, index=False)
                return buf.getvalue(), "modelo_cablebus_financiero.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, df in sheets.items():
                    zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8"))
            return zbuf.getvalue(), "modelo_cablebus_financiero.zip", "application/zip"

        Y = pd.DataFrame({"Año": years})
        inv_sheet = pd.DataFrame({
            "Variable": [
                "Costo/km (aj)",
                "Longitud (km)",
                "CAPEX base",
                "Sobrecosto (%)",
                "CAPEX total (sin IDC)",
                "Años cons.",
                "Dist. Año 1 (%)",
                "IDC total cap.",
            ],
            "Valor": [
                st.session_state.get("costo_km_aj", COSTO_BASE_POR_KM),
                LONGITUD_KM,
                st.session_state.get("capex_base", 0.0),
                st.session_state.get("overrun_pct", 10.0),
                capex_total,
                st.session_state.get("cons_years", 2),
                st.session_state.get("share_y1", 0.5) * 100,
                out["idc_total"],
            ],
        })
        ingresos_sheet = Y.assign(
            Locales=out["ing_local"],
            Turistas=out["ing_tur"],
            Establecimientos=out["ing_est"],
            Total=out["ing_tot"],
        )
        opex_sheet = Y.assign(
            Personal=out["ox_personal"],
            Mant_menor=out["ox_menor"],
            Mant_mayor_evento=out["ox_may_evento"],
            Mant_mayor_prov=out["ox_may_prov"],
            Total=out["ox_tot"],
        )
        operativo_sheet = Y.assign(
            Ingresos=out["ing_tot"], OPEX=out["ox_tot"], Utilidad_operativa=out["uop"]
        )
        serv_actor_sheet = Y.assign(
            FONATUR=out["serv"]["FONATUR"],
            BANOBRAS_FONADIN=out["serv"]["BANOBRAS-FONADIN"],
            SHCP=out["serv"]["SHCP"],
            GOB_EDO=out["serv"]["Gobierno del Estado"],
            PRIVADO=out["serv"]["Privado"],
            TOTAL=out["serv"]["Total"],
        )
        cobertura_series_sheet = Y.assign(
            DSCR_total=out["dscr_tot"],
            Cov_FONATUR=out["cov_actor_series"]["FONATUR"],
            Cov_BANOBRAS=out["cov_actor_series"]["BANOBRAS-FONADIN"],
            Cov_SHCP=out["cov_actor_series"]["SHCP"],
            Cov_Estado=out["cov_actor_series"]["Gobierno del Estado"],
            Cov_Privado=out["cov_actor_series"]["Privado"],
        )
        saldo_sheet = Y.assign(**out["saldo"])
        dsra_sheet = Y.assign(DSRA_obj=out["dsra_target"], DSRA_flujo=out["dsra_flow"], DSRA_saldo=out["dsra_bal"])
        cobertura_resumen_sheet = pd.DataFrame({
            "Métrica": ["Cobertura total del horizonte (%)"] + list(out["cover_by_actor"].keys()),
            "Valor": [out["cover_total"] * 100.0] + [v * 100.0 for v in out["cover_by_actor"].values()],
        })
        valor_sheet = pd.DataFrame({
            "KPI": ["NPV Proyecto (sin deuda)", "NPV Post-Deuda + DSRA", "IRR Post-Deuda", "Tasa descuento (real)"],
            "Valor": [out["npv_proj"], out["npv_post"], out["irr_post"], st.session_state.get("disc_rate", 0.10)],
        })

        data_bytes, fname, mime = build_export({
            "Inversion": inv_sheet,
            "Ingresos": ingresos_sheet,
            "OPEX": opex_sheet,
            "Operativo": operativo_sheet,
            "Servicio_por_actor": serv_actor_sheet,
            "Cobertura_series": cobertura_series_sheet,
            "Saldos_por_institucion": saldo_sheet,
            "DSRA": dsra_sheet,
            "Cobertura_resumen": cobertura_resumen_sheet,
            "KPIs_valor": valor_sheet,
        })
        st.download_button("⬇️ Descargar modelo (XLSX o ZIP)", data=data_bytes, file_name=fname, mime=mime)

    # --------------------------- GLOSARIO / METODO ---------------------------
    with T_GLOS:
        st.markdown(textwrap.dedent("""
        ### Metodología (compacta)
        - **CAPEX** = costo/km ajustado × 10.1 + **Sobrecosto** (%).
        - **IDC**: capitalización mensual según curva de desembolsos (Año 1 vs subsecuentes).
        - **Ingresos** = Locales + Turistas (crecen a *g*) + Establecimientos (renta fija).
        - **OPEX** = Personal + Mant. menor + Mant. mayor (evento cada *N* años o **provisión** = monto/N).
        - **Utilidad operativa** = Ingresos − OPEX (pesos constantes).
        - **Deuda**: anualidad a tasa efectiva por actor; **Privado** admite *cash sweep* (α% utilidad) y prepagos.
        - **DSRA**: meta de *m* meses del servicio programado; se fondea y ajusta con cambios.
        - **Cobertura**: series **Utilidad/Servicio** total y por actor; cobertura del horizonte = Σmin(Utilidad, Servicio)/ΣServicio.
        - **Valor**: NPV/IRR con tasa real de descuento indicada.
        """))
