# app.py — Modelo Cablebús (ejecutivo, atado y con provisión de mantenimiento)
# Ejecuta: streamlit run app.py

import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# CONFIG & ESTILO
# ============================================================
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
.small-note { color:#94a3b8; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILIDADES
# ============================================================
def _parse_num(txt, default):
    try:
        s = str(txt).strip().replace(" ", "").replace(",", "")
        return float(s)
    except Exception:
        return float(default)

def money_input(label: str, key: str, default: float, help: str | None = None, decimals: int = 2) -> float:
    initial = st.session_state.get(key, default)
    shown = f"{float(initial):,.{decimals}f}"
    txt = st.text_input(label, value=shown, key=f"{key}__txt", help=help)
    val = _parse_num(txt, default)
    st.session_state[key] = val
    return val

def int_input_commas(label: str, key: str, default: int, help: str | None=None) -> int:
    initial = int(st.session_state.get(key, default))
    txt = st.text_input(label, value=f"{initial:,}", key=f"{key}__txt", help=help)
    val = int(round(_parse_num(txt, default)))
    st.session_state[key] = val
    return val

def percent_input(label: str, key: str, default_percent: float, help: str | None = None, decimals: int = 2) -> float:
    initial = st.session_state.get(key, default_percent)
    shown = f"{float(initial):.{decimals}f}"
    txt = st.text_input(f"{label} (%)", value=shown, key=f"{key}__pct", help=help)
    val_pct = _parse_num(txt, default_percent)
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

# --- Gráficas con Altair (números grandes) ---
def bar_comp(df_name_value_pct, title):
    chart = (
        alt.Chart(df_name_value_pct)
        .mark_bar()
        .encode(
            x=alt.X("Nombre:N", sort="-y", title=None, axis=alt.Axis(labelFontSize=14)),
            y=alt.Y("Monto:Q", axis=alt.Axis(format=",", title="Monto (MXN)", labelFontSize=14, titleFontSize=16)),
            tooltip=[alt.Tooltip("Nombre:N"), alt.Tooltip("Monto:Q", format=","), alt.Tooltip("% del total:Q", format=".1%")],
            color=alt.Color("Nombre:N", legend=None),
        )
        .properties(height=300, title=alt.TitleParams(title, fontSize=16))
    )
    return chart

def line_series(df_long, title):
    chart = (
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
    return chart

# Amortización mensual → series anuales
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

def last_credit_year(saldo_dict) -> int | None:
    last = 0
    for arr in saldo_dict.values():
        a = np.array(arr, dtype=float)
        idx = np.where(a > 0)[0]
        if idx.size > 0:
            last = max(last, int(idx.max() + 1))
    return last or None

# ============================================================
# PARÁMETROS ESTRUCTURALES
# ============================================================
COSTO_BASE_POR_KM = 450_000_000  # 450 M por km
LONGITUD_KM = 10.1               # fija

# ============================================================
# RAIL DERECHO: Multiplicadores globales
# ============================================================
col_main, col_rail = st.columns([0.76, 0.24], gap="large")

with col_rail:
    st.markdown('<div class="rail-derecho">', unsafe_allow_html=True)
    st.subheader("Ajustes globales (multiplicadores)")
    st.caption("Aplican proporcionalmente sobre los supuestos base.")
    dem_x   = st.slider("Escala de Demanda (×)",      0.50, 1.50, st.session_state.get("dem_x",   1.00), 0.05, key="dem_x")
    capkm_x = st.slider("Escala de Costo por km (×)", 0.50, 1.50, st.session_state.get("capkm_x", 1.00), 0.05, key="capkm_x")
    opex_x  = st.slider("Escala de OPEX (×)",         0.50, 1.50, st.session_state.get("opex_x",  1.00), 0.05, key="opex_x")
    st.caption("- **Demanda ×** multiplica aforos de Local y Turista.\n- **Costo por km ×** multiplica $450M/km.\n- **OPEX ×** multiplica Personal, Mant. menor y el monto de Mant. mayor.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CONTENIDO
# ============================================================
with col_main:
    st.markdown("## 🚡 Modelo Cablebús — Tablero Financiero")

    T_INV, T_ING, T_OPEX, T_CAP, T_RES, T_GLOS = st.tabs(
        ["Inversión", "Ingresos", "Costos Operativos", "Estructura de Capital", "Resultados", "Glosario"]
    )

    # ----------------------- INVERSIÓN ------------------------
    with T_INV:
        st.markdown("### Inversión resumida")
        st.write("Longitud del proyecto **10.1 km** (fija).")

        costo_km_aj = COSTO_BASE_POR_KM * capkm_x
        overrun_frac = percent_input(
            "Sobrecosto estimado por predios y otros (sobre CAPEX base)",
            key="overrun_pct", default_percent=10.0,
            help="Ej.: 10% sobre el CAPEX base (costo por km ajustado × 10.1 km)."
        )

        capex_base  = costo_km_aj * LONGITUD_KM
        sobrecosto  = capex_base * overrun_frac
        capex_total = capex_base + sobrecosto

        st.session_state["capex_total"] = float(capex_total)
        st.session_state["capex_base"]  = float(capex_base)
        st.session_state["costo_km_aj"] = float(costo_km_aj)

        df_wide(pd.DataFrame({
            "Concepto": [
                "Costo por km (ajustado)", "Longitud (km)", "CAPEX base (sin sobrecosto)",
                "Sobrecosto (predios + otros)", "CAPEX total (a financiar)"
            ],
            "Valor": [pesos(costo_km_aj), f"{LONGITUD_KM:.1f}", pesos(capex_base), pesos(sobrecosto), pesos(capex_total)]
        }))

    # ----------------------- INGRESOS -------------------------
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

        # Totales Año 1 por fuente (con multiplicador de demanda)
        y1_local = (aforo_local * st.session_state["dem_x"]) * 365.0 * tarifa_local
        y1_tur   = (aforo_tur   * st.session_state["dem_x"]) * 365.0 * tarifa_tur
        y1_est   = locales * renta_mes * 12.0
        y1_total = y1_local + y1_tur + y1_est

        # % del total
        comp_ing_df = pd.DataFrame({
            "Nombre": ["Usuarios Locales","Usuarios Turistas","Establecimientos"],
            "Monto":  [y1_local, y1_tur, y1_est]
        })
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

    # ----------------------- OPEX -----------------------------
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
            use_provision = st.checkbox("Suavizar mant. mayor como **provisión anual equivalente**", value=True, help="Usa monto/periodicidad cada año. Los flujos 'evento' se conservan para análisis.")

        # Totales Año 1 (con multiplicador OPEX ×)
        y1_per = opex_personal_y1 * st.session_state["opex_x"]
        y1_men = opex_menor_y1   * st.session_state["opex_x"]
        y1_may_event = (opex_mayor_monto * st.session_state["opex_x"]) if (1 % max(1, mayor_cada) == 0) else 0.0
        y1_may_prov  = (opex_mayor_monto / max(1, mayor_cada)) * st.session_state["opex_x"]
        y1_ox_tot = y1_per + y1_men + (y1_may_prov if use_provision else y1_may_event)

        comp_ox_df = pd.DataFrame({
            "Nombre": ["Personal","Mant. menor","Mant. mayor" + (" (provisión)" if use_provision else " (evento)")],
            "Monto":  [y1_per, y1_men, (y1_may_prov if use_provision else y1_may_event)]
        })
        comp_ox_df["% del total"] = comp_ox_df["Monto"] / max(y1_ox_tot, 1e-9)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Personal — Año 1", f"{pesos(y1_per)}  ({comp_ox_df['% del total'][0]:.1%})")
        n2.metric("Mant. menor — Año 1", f"{pesos(y1_men)}  ({comp_ox_df['% del total'][1]:.1%})")
        n3.metric(("Mant. mayor (prov.)" if use_provision else "Mant. mayor (evento)") + " — Año 1",
                  f"{pesos(comp_ox_df['Monto'][2])}  ({comp_ox_df['% del total'][2]:.1%})")
        n4.metric("**OPEX total Año 1**", pesos(y1_ox_tot))

        st.altair_chart(bar_comp(comp_ox_df, "Composición de OPEX (Año 1)"), use_container_width=True)

        st.caption("La **provisión anual equivalente** reparte el monto de mantenimiento mayor entre los años (monto/periodicidad). Para flujos de caja, se conserva también la serie de **eventos**.")

        st.session_state["opex_params"] = {
            "personal_y1": opex_personal_y1, "menor_y1": opex_menor_y1,
            "mayor_monto": opex_mayor_monto, "g_opex": g_opex, "mayor_cada": mayor_cada,
            "use_provision": use_provision
        }

    # ----------------- ESTRUCTURA DE CAPITAL ------------------
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
        publico = st.slider("Participación **pública**", 0.0, 1.0, value=0.5, step=0.01, key="pub_pct")
        privado = 1.0 - publico
        st.write(f"**Público:** {publico:.0%}  |  **Privado:** {privado:.0%}")

        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("FONATUR (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonatur", on_change=_rebalance_public, args=("fonatur",))
        c2.number_input("BANOBRAS-FONADIN (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonadin", on_change=_rebalance_public, args=("fonadin",))
        c3.number_input("SHCP (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="shcp", on_change=_rebalance_public, args=("shcp",))
        c4.number_input("Gobierno del Estado (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="edo", on_change=_rebalance_public, args=("edo",))
        st.caption(f"Suma pública actual: **{st.session_state['fonatur'] + st.session_state['fonadin'] + st.session_state['shcp'] + st.session_state['edo']:.2f}%**")

        st.markdown("### Crédito por actor (tasa base + ajuste por plazo)")
        with st.expander("Tasa anual (% con punto) y plazo (años)."):
            slope = percent_input("Ajuste de tasa por plazo (p.p. por año)", key="slope_bps", default_percent=0.10,
                                  help="Ej.: 0.10 = +0.10% por cada año adicional de plazo (puntos porcentuales).")
            c1, c2, c3, c4, c5 = st.columns(5)
            r_fon = percent_input("FONATUR — Tasa base", key="tasa_fon_base", default_percent=8.00);  n_fon = c1.number_input("FONATUR — Años", 1, 40, 15, 1, key="n_fon")
            r_ban = percent_input("BANOBRAS-FONADIN — Tasa base", key="tasa_ban_base", default_percent=9.50); n_ban = c2.number_input("BANOBRAS — Años", 1, 40, 20, 1, key="n_ban")
            r_shc = percent_input("SHCP — Tasa base", key="tasa_shcp_base", default_percent=9.00);   n_shc = c3.number_input("SHCP — Años", 1, 40, 18, 1, key="n_shc")
            r_edo = percent_input("Estado — Tasa base", key="tasa_edo_base", default_percent=11.00); n_edo = c4.number_input("Estado — Años", 1, 40, 15, 1, key="n_edo")
            r_pri = percent_input("Privado — Tasa base", key="tasa_pri_base", default_percent=14.00); n_pri = c5.number_input("Privado — Años", 1, 40, 12, 1, key="n_pri")

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

    # ------------------------ CÁLCULO -------------------------
    @st.cache_data(show_spinner=False)
    def calc_projection(horizonte: int, dem_x: float, opex_x: float,
                        ingresos_params: dict, opex_params: dict,
                        capex_total: float, publico: float, pub_weights: dict,
                        rates_years: dict):
        years = np.arange(1, horizonte + 1)

        # ===== Ingresos por segmento
        ing_local_y1 = (ingresos_params["aforo_local"] * dem_x) * 365.0 * ingresos_params["tarifa_local"]
        ing_tur_y1   = (ingresos_params["aforo_tur"]   * dem_x) * 365.0 * ingresos_params["tarifa_tur"]
        ing_locals   = ingresos_params["locales"] * ingresos_params["renta_mes"] * 12.0

        g = ingresos_params["g_demand"]
        ing_local = np.array([ing_local_y1 * ((1 + g) ** (t - 1)) for t in years], dtype=float)
        ing_tur   = np.array([ing_tur_y1   * ((1 + g) ** (t - 1)) for t in years], dtype=float)
        ing_est   = np.full_like(ing_local, ing_locals, dtype=float)
        ingresos_totales = ing_local + ing_tur + ing_est

        # ===== OPEX (desglosado)
        use_prov = bool(opex_params["use_provision"])
        g_ox = opex_params["g_opex"]; cada = max(1, int(opex_params["mayor_cada"]))
        per0 = opex_params["personal_y1"] * opex_x
        men0 = opex_params["menor_y1"]   * opex_x
        maym = opex_params["mayor_monto"]* opex_x

        opex_personal = np.array([per0 * ((1 + g_ox) ** (t - 1)) for t in years], dtype=float)
        opex_menor    = np.array([men0 * ((1 + g_ox) ** (t - 1)) for t in years], dtype=float)
        # Evento vs Provisión
        opex_mayor_evento = np.array([ (maym if (t % cada == 0) else 0.0) for t in years ], dtype=float)
        opex_mayor_prov   = np.full_like(opex_menor, (maym / cada), dtype=float)
        opex_total        = opex_personal + opex_menor + (opex_mayor_prov if use_prov else opex_mayor_evento)

        utilidad_op   = ingresos_totales - opex_total

        # ===== Financiamiento (anualidad mensual → anual)
        capex = float(capex_total)
        contrib_publica = capex * publico
        contrib_privada = capex * (1.0 - publico)
        base_fon  = contrib_publica * pub_weights["fonatur"]
        base_ban  = contrib_publica * pub_weights["fonadin"]
        base_shcp = contrib_publica * pub_weights["shcp"]
        base_edo  = contrib_publica * pub_weights["edo"]
        base_pri  = contrib_privada

        rf, nf = rates_years["FONATUR"]; rb, nb = rates_years["BANOBRAS-FONADIN"]
        rs, ns = rates_years["SHCP"];     re, ne = rates_years["Gobierno del Estado"]
        rp, npv= rates_years["Privado"]

        serv_fon, int_fon, amo_fon, sal_fon = amort_yearly_from_monthly(base_fon,  rf, nf, horizonte)
        serv_ban, int_ban, amo_ban, sal_ban = amort_yearly_from_monthly(base_ban,  rb, nb, horizonte)
        serv_shc, int_shc, amo_shc, sal_shc = amort_yearly_from_monthly(base_shcp, rs, ns, horizonte)
        serv_edo, int_edo, amo_edo, sal_edo = amort_yearly_from_monthly(base_edo,  re, ne, horizonte)
        serv_pri, int_pri, amo_pri, sal_pri = amort_yearly_from_monthly(base_pri,  rp, npv, horizonte)

        serv_pub = serv_fon + serv_ban + serv_shc + serv_edo
        serv_tot = serv_pub + serv_pri
        saldo_dict = {
            "FONATUR": sal_fon, "BANOBRAS-FONADIN": sal_ban, "SHCP": sal_shc,
            "Gobierno del Estado": sal_edo, "Privado": sal_pri
        }

        flujo_post_deuda = utilidad_op - serv_tot
        dscr = np.divide(utilidad_op, serv_tot, out=np.full_like(utilidad_op, np.nan), where=serv_tot>0)

        efectivo_op   = -capex + np.cumsum(utilidad_op)      # sin deuda
        efectivo_post = -capex + np.cumsum(flujo_post_deuda) # con deuda

        # Costos vida completa por actor
        def costo_total(P, r, n):
            pm = annuity_payment_monthly(P, r, n)
            total_pagado = pm * 12 * n
            return int(round(P)), int(round(max(0.0, total_pagado - P))), n

        costo_rows = []
        for nombre, (P, r, n) in {
            "FONATUR": (base_fon, rf, nf),
            "BANOBRAS-FONADIN": (base_ban, rb, nb),
            "SHCP": (base_shcp, rs, ns),
            "Gobierno del Estado": (base_edo, re, ne),
            "Privado": (base_pri, rp, npv),
        }.items():
            principal_i, interes_i, fin = costo_total(P, r, n)
            costo_rows.append({
                "Actor": nombre,
                "Principal": principal_i,
                "Intereses (vida del crédito)": interes_i,
                "Costo total (Principal+Intereses)": principal_i + interes_i,
                "Fin de pago (Año)": fin
            })
        costo_df = pd.DataFrame(costo_rows)

        return {
            "years": years,
            "ing_local": ing_local, "ing_tur": ing_tur, "ing_est": ing_est,
            "ingresos_totales": ingresos_totales,
            "opex_personal": opex_personal, "opex_menor": opex_menor,
            "opex_mayor_evento": opex_mayor_evento, "opex_mayor_prov": opex_mayor_prov,
            "opex_total": opex_total,
            "utilidad_op": utilidad_op,
            "servicios": {"pub": serv_pub, "pri": serv_pri, "tot": serv_tot,
                          "FONATUR": serv_fon, "BANOBRAS-FONADIN": serv_ban, "SHCP": serv_shc,
                          "Gobierno del Estado": serv_edo, "Privado": serv_pri},
            "saldo_por_actor": saldo_dict,
            "flujo_post_deuda": flujo_post_deuda,
            "dscr": dscr,
            "efectivo_op": efectivo_op,
            "efectivo_post": efectivo_post,
            "costo_df": costo_df
        }

    # ----------------------- RESULTADOS -----------------------
    with T_RES:
        st.markdown("### Horizonte y cálculo")
        horizonte = int(st.number_input("Horizonte (años)", min_value=5, value=20, step=1, key="horiz"))

        capex_total = float(st.session_state.get("capex_total", 0.0))
        publico = float(st.session_state.get("pub_pct", 0.5))
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

        out = calc_projection(
            horizonte, float(st.session_state.get("dem_x",1.0)), float(st.session_state.get("opex_x",1.0)),
            st.session_state.get("ingresos_params", {"aforo_local":19220,"tarifa_local":11.0,"aforo_tur":4845,"tarifa_tur":50.0,"locales":80,"renta_mes":15000.0,"g_demand":0.03}),
            st.session_state.get("opex_params", {"personal_y1":20_000_000.0,"menor_y1":18_000_000.0,"mayor_monto":100_000_000.0,"g_opex":0.02,"mayor_cada":10,"use_provision":True}),
            capex_total, publico, pub_w, rates_years
        )
        years = out["years"]

        # ---------- RESUMEN EJECUTIVO ----------
        y1_ing  = int(round(out["ingresos_totales"][0]))
        y1_ox   = int(round(out["opex_total"][0]))
        y1_uop  = int(round(out["utilidad_op"][0]))
        y1_deuda= int(round(out["servicios"]["tot"][0]))
        y1_dscr = float(out["dscr"][0]) if out["servicios"]["tot"][0] > 0 else np.nan

        # Paybacks
        br_op_idx   = np.argmax(out["efectivo_op"]   >= 0) if np.any(out["efectivo_op"]   >= 0) else None
        br_post_idx = np.argmax(out["efectivo_post"] >= 0) if np.any(out["efectivo_post"] >= 0) else None
        pay_op   = int(years[br_op_idx])   if br_op_idx   is not None else None
        pay_post = int(years[br_post_idx]) if br_post_idx is not None else None

        R1, R2, R3, R4, R5, R6 = st.columns(6)
        R1.metric("CAPEX total", pesos(capex_total))
        R2.metric("Ingresos Año 1", pesos(y1_ing))
        R3.metric("OPEX Año 1", pesos(y1_ox))
        R4.metric("Utilidad operativa Año 1", pesos(y1_uop))
        R5.metric("Servicio de deuda Año 1", pesos(y1_deuda))
        R6.metric("DSCR Año 1", f"{y1_dscr:.2f}×" if not np.isnan(y1_dscr) else "—")

        Q1, Q2, Q3 = st.columns(3)
        Q1.metric("Payback **operativo** (años)", f"{pay_op}" if pay_op else "—")
        Q2.metric("Payback **post-deuda** (años)", f"{pay_post}" if pay_post else "—")
        last_year = last_credit_year(out["saldo_por_actor"])
        Q3.metric("Fin del último crédito (año)", f"{last_year}" if last_year else "—")

        st.divider()

        # ---------- SUBTABS ORGANIZADOS ----------
        SUB_OP, SUB_DEUDA, SUB_CAJA = st.tabs(["Operación", "Deuda & Financiamiento", "Caja"])

        # ----- Operación -----
        with SUB_OP:
            df_ing = pd.DataFrame({
                "Año": years,
                "Usuarios Locales": np.rint(out["ing_local"]).astype("int64"),
                "Usuarios Turistas": np.rint(out["ing_tur"]).astype("int64"),
                "Establecimientos": np.rint(out["ing_est"]).astype("int64"),
            })
            st.altair_chart(line_series(df_ing.melt("Año", var_name="Serie", value_name="Valor"), "Ingresos por segmento"), use_container_width=True)

            df_ox = pd.DataFrame({
                "Año": years,
                "Personal": np.rint(out["opex_personal"]).astype("int64"),
                "Mant. menor": np.rint(out["opex_menor"]).astype("int64"),
                "Mant. mayor (evento)": np.rint(out["opex_mayor_evento"]).astype("int64"),
                "Mant. mayor (provisión)": np.rint(out["opex_mayor_prov"]).astype("int64"),
            })
            st.altair_chart(line_series(df_ox.melt("Año", var_name="Serie", value_name="Valor"), "OPEX por componente"), use_container_width=True)

            st.altair_chart(line_series(
                pd.DataFrame({"Año": years, "Utilidad operativa": np.rint(out["utilidad_op"]).astype("int64")}).melt("Año", var_name="Serie", value_name="Valor"),
                "Utilidad operativa"), use_container_width=True)

        # ----- Deuda & Financiamiento -----
        with SUB_DEUDA:
            serv_tot = np.rint(out["servicios"]["tot"]).astype("int64")
            df_sd = pd.DataFrame({"Año": years, "Servicio de deuda": serv_tot, "Utilidad operativa": np.rint(out["utilidad_op"]).astype("int64")})
            st.altair_chart(line_series(df_sd.melt("Año", var_name="Serie", value_name="Valor"), "Servicio de deuda vs Utilidad operativa"), use_container_width=True)

            saldo = out["saldo_por_actor"]
            saldo_df = pd.DataFrame({"Año": years, **{k: np.rint(v).astype("int64") for k,v in saldo.items()}})
            st.altair_chart(line_series(saldo_df.melt("Año", var_name="Serie", value_name="Valor"), "Saldo de deuda por institución"), use_container_width=True)

            st.altair_chart(line_series(
                pd.DataFrame({"Año": years, "DSCR": out["dscr"]}).melt("Año", var_name="Serie", value_name="Valor"),
                "Cobertura de servicio de deuda (DSCR)"), use_container_width=True)

            show_cost = out["costo_df"].copy()
            for c in ["Principal", "Intereses (vida del crédito)", "Costo total (Principal+Intereses)"]:
                show_cost[c] = show_cost[c].map(with_commas)
            st.markdown("**Costo por actor (vida completa del crédito)**")
            df_wide(show_cost)

        # ----- Caja -----
        with SUB_CAJA:
            deuda_total_saldo = np.rint(np.sum(np.vstack(list(out["saldo_por_actor"].values())), axis=0)).astype("int64")
            df_cash = pd.DataFrame({
                "Año": years,
                "Deuda total (saldo insoluto)": deuda_total_saldo,
                "Efectivo acumulado (operativo)": np.rint(out["efectivo_op"]).astype("int64"),
                "Efectivo acumulado (post-deuda)": np.rint(out["efectivo_post"]).astype("int64"),
            })
            st.altair_chart(line_series(df_cash.melt("Año", var_name="Serie", value_name="Valor"), "Deuda vs Efectivo acumulado"), use_container_width=True)

        # ------------------ Exportación ------------------
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
            "Variable": ["Costo por km (ajustado)", "Longitud (km)", "CAPEX base", "Sobrecosto (%)", "CAPEX total"],
            "Valor": [st.session_state.get("costo_km_aj", COSTO_BASE_POR_KM), LONGITUD_KM,
                      st.session_state.get("capex_base", 0.0), st.session_state.get("overrun_pct", 10.0),
                      st.session_state.get("capex_total", 0.0)]
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
            Ingresos=out["ingresos_totales"], OPEX=out["opex_total"], Utilidad_operativa=out["utilidad_op"], Efectivo_acum_operativo=out["efectivo_op"]
        )
        deuda_sheet = years_df.assign(
            Servicio_deuda_total=out["servicios"]["tot"], DSCR=out["dscr"], Efectivo_acum_post_deuda=out["efectivo_post"]
        )
        saldos_sheet = years_df.assign(**out["saldo_por_actor"])

        # Parámetros globales (para trazabilidad)
        params_global = pd.DataFrame({
            "Parámetro": ["Demanda ×","Costo por km ×","OPEX ×","Crec. boletaje (g)","Crec. OPEX (g)","Mant. mayor (monto)","Periodicidad (años)","OPEX usa provisión"],
            "Valor": [st.session_state.get("dem_x",1.0), st.session_state.get("capkm_x",1.0), st.session_state.get("opex_x",1.0),
                      st.session_state.get("ingresos_params",{}).get("g_demand",0.03),
                      st.session_state.get("opex_params",{}).get("g_opex",0.02),
                      st.session_state.get("opex_params",{}).get("mayor_monto",100_000_000.0),
                      st.session_state.get("opex_params",{}).get("mayor_cada",10),
                      st.session_state.get("opex_params",{}).get("use_provision",True)]
        })

        data_bytes, fname, mime = build_export({
            "Parametros_globales": params_global,
            "Inversion": inv_sheet,
            "Ingresos": ingresos_sheet,
            "OPEX": opex_sheet,
            "Operativo": operativo_sheet,
            "Deuda": deuda_sheet,
            "Saldos_por_actor": saldos_sheet,
            "Costo_por_actor": out["costo_df"]
        })
        st.download_button("⬇️ Descargar modelo (XLSX o ZIP)", data=data_bytes, file_name=fname, mime=mime)

    # ------------------------- GLOSARIO ------------------------
    with T_GLOS:
        st.markdown("""
### Términos clave
- **CAPEX**: Inversión inicial (costo por km × 10.1 + sobrecosto).
- **OPEX**: Suma anual de **personal** + **mantenimiento menor** + **mantenimiento mayor**.
- **Mantenimiento mayor (evento)**: Desembolso puntual cada *N* años por un monto fijo.
- **Provisión anual equivalente**: Reparto uniforme del costo del mantenimiento mayor entre los *N* años (monto/N). Facilita lectura comparativa; no sustituye el análisis de flujo por **evento**.
- **Ingresos**: Locales + Turistas + Establecimientos (rentas). Locales y Turistas crecen a **g** anual; rentas fijas.
- **Utilidad operativa**: Ingresos − OPEX (según se elija evento o provisión).
- **Servicio de deuda**: Pagos **mensuales** de anualidad convertidos a suma **anual**.
- **Saldo de deuda**: Monto pendiente por institución al cierre de cada año.
- **DSCR**: Utilidad operativa / Servicio de deuda (≥ 1 indica cobertura).
- **Efectivo acumulado (operativo / post-deuda)**: −CAPEX + Σ(Utilidad) / −CAPEX + Σ(Utilidad − Deuda).

### Nota metodológica (estructura del modelo)
1. **Entradas**  
   a) Multiplicadores globales: Demanda×, Costo/km×, OPEX×.  
   b) Inversión: costo por km ajustado × 10.1 km + sobrecosto %.  
   c) Ingresos: aforos diarios (Locales/Turistas), tarifas, locales y renta; crecimiento de boletaje *g*.  
   d) OPEX: personal y mant. menor con crecimiento *g_opex*; mant. mayor como **evento** cada *N* años y/o como **provisión** (monto/N).  
   e) Financiamiento: bloque público/privado; reparto entre FONATUR, BANOBRAS-FONADIN, SHCP, Estado; **tasa base + ajuste por plazo**; anualidad mensual.

2. **Cálculo**  
   - Ingresos por segmento → total por año.  
   - OPEX por rubro → total por año (según *evento* o *provisión* seleccionada).  
   - **Utilidad operativa** = Ingresos − OPEX.  
   - Servicio de deuda por actor (mensual → anual) y **saldo anual**.  
   - **DSCR** = Utilidad operativa / Servicio de deuda.  
   - **Efectivo acumulado** operativo y post-deuda (= −CAPEX + Σ(flujos)).  

3. **Reporte y descarga**  
   - KPIs ejecutivos (Año 1), series y saldos por actor.  
   - Excel con hojas de parámetros, ingresos, OPEX (evento y provisión), operativo, deuda, saldos y costos por actor.

> Sugerencia para mayor rigor: si necesitas ver **ambas visiones a la vez**, deja activada la **provisión** para comparabilidad y consulta en *Operación* la curva **“Mant. mayor (evento)”** para impactos de caja puntuales.
        """)
