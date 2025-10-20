# app.py — Modelo Cablebús (pesos constantes, sin archivos externos)
# Ejecuta: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configuración base
# ------------------------------------------------------------
st.set_page_config(page_title="Modelo Cablebús — Tablero Financiero", layout="wide")

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def with_commas(n) -> str:
    """String con comas para miles (sin símbolo $)."""
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return ""

def pesos(n) -> str:
    """MXN con comas y símbolo $ para métricas (texto)."""
    try:
        return f"${int(round(float(n))):,}"
    except Exception:
        return ""

def _to_float(s, default: float) -> float:
    """Convierte texto a float aceptando coma o punto."""
    if s is None:
        return float(default)
    if isinstance(s, (int, float, np.floating)):
        return float(s)
    s = str(s).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float(default)

def float_input(label: str, key: str, default: float, help: str | None = None, decimals: int = 2) -> float:
    """
    Text input para números con punto. Evita conflictos de claves.
    - El widget usa key=f"{key}__txt"
    - El valor limpio se guarda en st.session_state[key]
    """
    initial = st.session_state.get(key, default)
    txt = st.text_input(label, value=f"{float(initial):.{decimals}f}", key=f"{key}__txt", help=help)
    val = _to_float(txt, default)
    st.session_state[key] = val
    return val

def percent_input(label: str, key: str, default_percent: float, help: str | None = None, decimals: int = 2) -> float:
    """
    Pide % (con punto) y devuelve FRACCIÓN (0.085 si pones 8.5).
    - Guarda visible el % en st.session_state[key]
    """
    initial = st.session_state.get(key, default_percent)
    txt = st.text_input(f"{label} (%)", value=f"{float(initial):.{decimals}f}", key=f"{key}__pct", help=help)
    val_pct = _to_float(txt, default_percent)
    st.session_state[key] = val_pct
    return max(0.0, float(val_pct)) / 100.0

def annuity_payment_monthly(principal: float, annual_rate: float, years: int) -> float:
    """Pago mensual (sistema francés). Tasa anual -> mensual."""
    P = max(float(principal), 0.0)
    rm = max(float(annual_rate), 0.0) / 12.0
    m = max(int(years * 12), 1)
    if rm == 0:
        return P / m
    return P * rm / (1 - (1 + rm) ** (-m))

# ------------------------------------------------------------
# Defaults del modelo (sin archivos externos)
# ------------------------------------------------------------
INV_TOTAL_MODELO = 5_310_002_376
INV_ANUAL_PROG   = INV_TOTAL_MODELO / 2  # 2025 y 2026 iguales

INV_DEFAULTS = [
    {"Concepto":"Costo por KM", "SUMA":5175000000, "2025":2587500000, "2026":2587500000},
    {"Concepto":"Adquisición de terrenos", "SUMA":36225000, "2025":18112500, "2026":18112500},
    {"Concepto":"Elevación de torres", "SUMA":600000, "2025":300000, "2026":300000},
    {"Concepto":"Otros", "SUMA":np.nan, "2025":0, "2026":0},
    {"Concepto":"SUMA TOTAL DE CONSTRUCCIÓN", "SUMA":5211825000, "2025":2605912500, "2026":2605912500},
    {"Concepto":"Gastos administrativos, permisos, entre otros", "SUMA":10000000, "2025":5000000, "2026":5000000},
    {"Concepto":"Derecho de vía", "SUMA":26059125, "2025":13029563, "2026":13029563},
    {"Concepto":"Estudios y Proyectos", "SUMA":5000000, "2025":2500000, "2026":2500000},
    {"Concepto":"Dictámenes", "SUMA":5000000, "2025":2500000, "2026":2500000},
    {"Concepto":"Obras de mitigación", "SUMA":52118250, "2025":26059125, "2026":26059125},
    {"Concepto":"Seguros", "SUMA":np.nan, "2025":np.nan, "2026":np.nan},
    {"Concepto":"Otros gastos", "SUMA":98177375, "2025":49088688, "2026":49088688},
]

AforoDefaults = {
    "Local":   {"tarifa": 11.0, "aforo_dia": 19220},
    "Turista": {"tarifa": 50.0, "aforo_dia": 4845},
    "Locales": {"cantidad": 80,  "renta_mes": 15000.0},
}

OpexDefaults = [
    {"Categoría": "Operación y Mantenimiento",    "Concepto": "Operación",                         "Monto": 12000000},
    {"Categoría": "Operación y Mantenimiento",    "Concepto": "Conservación",                      "Monto": 15000000},
    {"Categoría": "Operación y Mantenimiento",    "Concepto": "Seguros",                           "Monto": 387215},
    {"Categoría": "Operación y Mantenimiento",    "Concepto": "Fianzas",                           "Monto": 54489},
    {"Categoría": "Operación y Mantenimiento",    "Concepto": "Mantenimiento mayor (cada 10 años)","Monto": np.nan},
    {"Categoría": "Administración y Fiducia",     "Concepto": "Administración",                    "Monto": 790483},
    {"Categoría": "Administración y Fiducia",     "Concepto": "Administración I+D",                "Monto": 163982},
    {"Categoría": "Administración y Fiducia",     "Concepto": "Comisiones Bancarias",              "Monto": 109346},
    {"Categoría": "Administración y Fiducia",     "Concepto": "Honorarios fiducarios",             "Monto": 71042},
    {"Categoría": "Auditorías y Estudios",        "Concepto": "Ingeniero Independiente",           "Monto": 60141},
    {"Categoría": "Auditorías y Estudios",        "Concepto": "Auditor Operativo",                 "Monto": 14392},
    {"Categoría": "Auditorías y Estudios",        "Concepto": "Auditor Estados Financieros",       "Monto": 3340},
    {"Categoría": "Auditorías y Estudios",        "Concepto": "Otros estudios",                    "Monto": 3000},
    {"Categoría": "Contraprestaciones/Regulatorio","Concepto": "Contraprestación Guerrero (5%)",    "Monto": 9500000},
    {"Categoría": "I+D y Arrendamientos",         "Concepto": "Rentas I+D",                        "Monto": 47144},
]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom:12px">
      <h1 style="margin:0;font-size:44px">🚡 Modelo Cablebús — Tablero Financiero</h1>
      <p style="margin:4px 0 0;color:#9aa3af">
        Proyección operativa, estructura de capital y servicio de deuda — <b>pesos constantes</b>.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("**Nota:** todos los montos están en **pesos constantes de hoy** (deflactados).")

# ------------------------------------------------------------
# Pestañas
# ------------------------------------------------------------
T_INV, T_CAP, T_AF, T_OPEX, T_RES = st.tabs(
    ["Inversión", "Estructura de Capital", "Aforo e Ingresos", "Costos Operativos", "Resultados"]
)

# ------------------------------------------------------------
# Inversión (tabla única amarrada al total)
# ------------------------------------------------------------
with T_INV:
    st.markdown("### Inversión normalizada (amarrada al modelo)")

    base = pd.DataFrame(INV_DEFAULTS).copy()
    for col in ["2025", "2026"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    s25 = float(base["2025"].sum()); s26 = float(base["2026"].sum())
    k25 = (INV_ANUAL_PROG / s25) if s25 > 0 else 0.0
    k26 = (INV_ANUAL_PROG / s26) if s26 > 0 else 0.0

    used = base[["Concepto"]].copy()
    used["2025"] = (base["2025"] * k25).round(0)
    used["2026"] = (base["2026"] * k26).round(0)
    used["SUMA"]  = (used["2025"] + used["2026"]).round(0)

    total_row = pd.DataFrame([{
        "Concepto": "TOTAL PROYECTO (control)",
        "2025": INV_ANUAL_PROG,
        "2026": INV_ANUAL_PROG,
        "SUMA":  INV_TOTAL_MODELO
    }])
    used = pd.concat([total_row, used], ignore_index=True)

    # Ajustes de redondeo para asegurar igualdad exacta año con año
    adj25 = int(INV_ANUAL_PROG - used.loc[1:, "2025"].sum())
    adj26 = int(INV_ANUAL_PROG - used.loc[1:, "2026"].sum())
    if len(used) > 1:
        used.loc[1, "2025"] += adj25
        used.loc[1, "2026"] += adj26
        used.loc[1, "SUMA"]  = used.loc[1, "2025"] + used.loc[1, "2026"]

    used.loc[0, "SUMA"] = used.loc[0, "2025"] + used.loc[0, "2026"]

    # Vista legible con comas (como texto)
    tabla_inv = used.set_index("Concepto").copy()
    for c in ["2025", "2026", "SUMA"]:
        tabla_inv[c] = tabla_inv[c].round(0).astype("int64")
    show_inv = tabla_inv.copy()
    for c in ["2025", "2026", "SUMA"]:
        show_inv[c] = show_inv[c].map(with_commas)

    st.dataframe(show_inv, use_container_width=True)
    st.caption("La primera fila es el **control**; las filas siguientes suman exactamente ese total por año.")
    st.session_state["capex_modelo"] = INV_TOTAL_MODELO

# ------------------------------------------------------------
# Estructura de Capital
# ------------------------------------------------------------
def _rebalance_public(changed: str):
    """Mantiene suma pública=100% rebalanceando el resto."""
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
    st.markdown("### 1) Bloques")
    publico = st.slider("Participación **pública**", 0.0, 1.0, value=0.5, step=0.01, key="pub_pct")
    privado = 1.0 - publico
    st.write(f"**Público:** {publico:.0%}  |  **Privado:** {privado:.0%}")

    st.markdown("### 2) Distribución del **bloque público** (suma 100%)")
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("FONATUR (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonatur", on_change=_rebalance_public, args=("fonatur",))
    c2.number_input("BANOBRAS-FONADIN (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="fonadin", on_change=_rebalance_public, args=("fonadin",))
    c3.number_input("SHCP (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="shcp", on_change=_rebalance_public, args=("shcp",))
    c4.number_input("Gobierno del Estado (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="edo", on_change=_rebalance_public, args=("edo",))
    st.caption(f"Suma pública actual: **{st.session_state['fonatur'] + st.session_state['fonadin'] + st.session_state['shcp'] + st.session_state['edo']:.2f}%**")

    # Aportaciones por bloque/actor
    capex_modelo = float(st.session_state.get("capex_modelo", INV_TOTAL_MODELO))
    contrib_publica = capex_modelo * publico
    contrib_privada = capex_modelo * privado
    pub_w = {k: st.session_state.get(k, 25.0)/100.0 for k in ["fonatur", "fonadin", "shcp", "edo"]}

    desglose_publico = {
        "FONATUR": contrib_publica * pub_w["fonatur"],
        "BANOBRAS-FONADIN": contrib_publica * pub_w["fonadin"],
        "SHCP": contrib_publica * pub_w["shcp"],
        "Gobierno del Estado": contrib_publica * pub_w["edo"],
    }

    df_ap = pd.DataFrame(
        [{"Bloque/Actor": "Público (Total)", "Aportación": contrib_publica}] +
        [{"Bloque/Actor": k, "Aportación": v} for k, v in desglose_publico.items()] +
        [{"Bloque/Actor": "Privado (Total)", "Aportación": contrib_privada}]
    )
    df_ap["Aportación"] = df_ap["Aportación"].round(0).astype("int64")
    show_ap = df_ap.copy()
    show_ap["Aportación (MXN)"] = show_ap["Aportación"].map(with_commas)
    show_ap = show_ap[["Bloque/Actor", "Aportación (MXN)"]]
    st.dataframe(show_ap, use_container_width=True)

    # 3) Crédito por actor (simple, atado a plazo)
    st.markdown("### 3) Crédito por actor (simple)")
    with st.expander("Tasa anual (% con punto) y plazo (años). Ajuste opcional por plazo."):
        slope = percent_input("Ajuste de tasa por plazo (p.p. por año)", key="slope_bps", default_percent=0.10,
                              help="Ejemplo: 0.10 = +0.10% por cada año adicional de plazo.")
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        r_fon = percent_input("FONATUR — Tasa base", key="tasa_fon_base", default_percent=8.00);  n_fon = cc1.number_input("FONATUR — Años", 1, 40, 15, 1, key="n_fon")
        r_ban = percent_input("BANOBRAS-FONADIN — Tasa base", key="tasa_ban_base", default_percent=9.50); n_ban = cc2.number_input("BANOBRAS — Años", 1, 40, 20, 1, key="n_ban")
        r_shc = percent_input("SHCP — Tasa base", key="tasa_shcp_base", default_percent=9.00);   n_shc = cc3.number_input("SHCP — Años", 1, 40, 18, 1, key="n_shc")
        r_edo = percent_input("Estado — Tasa base", key="tasa_edo_base", default_percent=11.00); n_edo = cc4.number_input("Estado — Años", 1, 40, 15, 1, key="n_edo")
        r_pri = percent_input("Privado — Tasa base", key="tasa_pri_base", default_percent=14.00); n_pri = cc5.number_input("Privado — Años", 1, 40, 12, 1, key="n_pri")

        # r_aplicada = r_base + slope * años  (todas en fracción)
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

# ------------------------------------------------------------
# Aforo e Ingresos (compacto)
# ------------------------------------------------------------
with T_AF:
    st.subheader("Parámetro de proyección")
    g = percent_input(
        "Crecimiento **anual** de demanda (boletaje)",
        key="g_demanda",
        default_percent=3.00,
        help="Afecta Local + Turista; rentas se mantienen constantes."
    )
    st.session_state["g_frac"] = g  # fracción 0.xx

    st.subheader("Aforo e Ingresos (compacto)")
    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        st.markdown("**Local**")
        tarifa_local = float_input("Tarifa Local (MXN)", key="tarifa_local", default=AforoDefaults["Local"]["tarifa"], decimals=2)
        aforo_local  = st.number_input("Aforo diario Local", value=AforoDefaults["Local"]["aforo_dia"], step=100, key="aforo_local")
        ingresos_local = aforo_local * tarifa_local * 365
        st.metric("Ingresos anuales", pesos(ingresos_local))

    with c2:
        st.markdown("**Turista**")
        tarifa_tur = float_input("Tarifa Turista (MXN)", key="tarifa_tur", default=AforoDefaults["Turista"]["tarifa"], decimals=2)
        aforo_tur  = st.number_input("Aforo diario Turista", value=AforoDefaults["Turista"]["aforo_dia"], step=50, key="aforo_tur")
        ingresos_tur = aforo_tur * tarifa_tur * 365
        st.metric("Ingresos anuales", pesos(ingresos_tur))

    with c3:
        st.markdown("**Locales comerciales**")
        locales   = st.number_input("Número de locales", value=AforoDefaults["Locales"]["cantidad"], step=1, key="locales")
        renta_mes = float_input("Renta neta mensual por local (MXN)", key="renta_mes", default=AforoDefaults["Locales"]["renta_mes"], decimals=2)
        ingresos_rentas = locales * renta_mes * 12
        st.metric("Ingresos anuales", pesos(ingresos_rentas))

    total_ingresos = float(ingresos_local + ingresos_tur + ingresos_rentas)
    st.metric("**Total ingresos anuales**", pesos(total_ingresos))

# ------------------------------------------------------------
# Costos Operativos
# ------------------------------------------------------------
with T_OPEX:
    st.subheader("Costos Operativos (anuales)")

    # Editor sin formatos (evitamos bug de sprintf)
    opex_df = st.data_editor(
        pd.DataFrame(OpexDefaults),
        key="opex_editor",
        num_rows="dynamic",
        use_container_width=True
    )

    costos_operativos = float(np.nansum(pd.to_numeric(opex_df["Monto"], errors="coerce")))
    st.session_state["costos_operativos"] = costos_operativos

    st.markdown("**Resumen por categoría (MXN/año):**")
    grp = opex_df.copy()
    grp["Monto"] = pd.to_numeric(grp["Monto"], errors="coerce").fillna(0)
    resumen = grp.groupby("Categoría", dropna=False)["Monto"].sum().astype("int64")
    show_resumen = resumen.map(with_commas).to_frame("Total categoría (MXN)")
    st.dataframe(show_resumen, use_container_width=True)

    st.metric("**Total costos operativos (anuales)**", pesos(costos_operativos))

# ------------------------------------------------------------
# Resultados (proyección + deuda)
# ------------------------------------------------------------
with T_RES:
    st.subheader("Resultados — Proyección (Ingresos, Costos, Utilidad) y Deuda")

    anos = int(st.number_input("Horizonte (años)", value=15, min_value=1, step=1, key="horizonte"))

    # Insumos
    g = float(st.session_state.get("g_frac", 0.03))
    tarifa_local  = float(st.session_state.get("tarifa_local", AforoDefaults["Local"]["tarifa"]))
    aforo_local   = int(st.session_state.get("aforo_local",  AforoDefaults["Local"]["aforo_dia"]))
    tarifa_tur    = float(st.session_state.get("tarifa_tur",  AforoDefaults["Turista"]["tarifa"]))
    aforo_tur     = int(st.session_state.get("aforo_tur",     AforoDefaults["Turista"]["aforo_dia"]))
    locales       = int(st.session_state.get("locales",       AforoDefaults["Locales"]["cantidad"]))
    renta_mes     = float(st.session_state.get("renta_mes",   AforoDefaults["Locales"]["renta_mes"]))
    costos0       = float(st.session_state.get("costos_operativos", float(np.nansum(pd.DataFrame(OpexDefaults)["Monto"]))))

    years = np.arange(1, anos + 1)

    ing_local0  = aforo_local * tarifa_local * 365
    ing_tur0    = aforo_tur  * tarifa_tur  * 365
    ing_rentas0 = locales * renta_mes * 12

    ingresos_boletaje = np.array([(ing_local0 + ing_tur0) * ((1 + g) ** (t - 1)) for t in years], dtype=float)
    ingresos_rentas   = np.full_like(ingresos_boletaje, ing_rentas0, dtype=float)
    ingresos_totales  = ingresos_boletaje + ingresos_rentas
    costos_series     = np.full_like(ingresos_boletaje, costos0, dtype=float)
    utilidad_anual    = ingresos_totales - costos_series

    # Estructura y deuda
    capex_modelo = float(st.session_state.get("capex_modelo", INV_TOTAL_MODELO))
    publico = float(st.session_state.get("pub_pct", 0.5)); privado = 1.0 - publico
    contrib_publica = capex_modelo * publico
    contrib_privada = capex_modelo * privado

    pub_w = {
        "fonatur": float(st.session_state.get("fonatur", 25.0))/100.0,
        "fonadin": float(st.session_state.get("fonadin", 25.0))/100.0,
        "shcp":    float(st.session_state.get("shcp",    25.0))/100.0,
        "edo":     float(st.session_state.get("edo",     25.0))/100.0,
    }
    base_fon  = contrib_publica * pub_w["fonatur"]
    base_ban  = contrib_publica * pub_w["fonadin"]
    base_shcp = contrib_publica * pub_w["shcp"]
    base_edo  = contrib_publica * pub_w["edo"]
    base_pri  = contrib_privada

    rates_years = st.session_state.get("rates_years", {
        "FONATUR": (0.08, 15), "BANOBRAS-FONADIN": (0.095, 20),
        "SHCP": (0.09, 18), "Gobierno del Estado": (0.11, 15), "Privado": (0.14, 12)
    })
    rf, nf  = rates_years["FONATUR"]
    rb, nb  = rates_years["BANOBRAS-FONADIN"]
    rs, ns  = rates_years["SHCP"]
    re, ne  = rates_years["Gobierno del Estado"]
    rp, npv = rates_years["Privado"]

    pm_fon  = annuity_payment_monthly(base_fon,  rf, nf);  serv_fon  = np.array([pm_fon*12  if t <= nf  else 0.0 for t in years])
    pm_ban  = annuity_payment_monthly(base_ban,  rb, nb);  serv_ban  = np.array([pm_ban*12  if t <= nb  else 0.0 for t in years])
    pm_shcp = annuity_payment_monthly(base_shcp, rs, ns);  serv_shcp = np.array([pm_shcp*12 if t <= ns  else 0.0 for t in years])
    pm_edo  = annuity_payment_monthly(base_edo,  re, ne);  serv_edo  = np.array([pm_edo*12  if t <= ne  else 0.0 for t in years])
    pm_pri  = annuity_payment_monthly(base_pri,  rp, npv); serv_pri  = np.array([pm_pri*12  if t <= npv else 0.0 for t in years])

    # Series enteras para mostrar tabulares y métricas
    to_int = lambda a: np.rint(a).astype("int64")
    ingresos_totales_i = to_int(ingresos_totales)
    costos_series_i    = to_int(costos_series)
    utilidad_anual_i   = to_int(utilidad_anual)
    serv_fon_i = to_int(serv_fon); serv_ban_i = to_int(serv_ban); serv_shcp_i = to_int(serv_shcp); serv_edo_i = to_int(serv_edo); serv_pri_i = to_int(serv_pri)
    serv_pub_i = to_int(serv_fon_i + serv_ban_i + serv_shcp_i + serv_edo_i)
    serv_prv_i = serv_pri_i
    serv_tot_i = to_int(serv_pub_i + serv_prv_i)
    flujo_neto_i    = to_int(utilidad_anual_i - serv_tot_i)
    efectivo_acum_i = to_int(np.cumsum(flujo_neto_i))

    # Gráfica principal
    st.line_chart(pd.DataFrame({
        "Año": years,
        "Ingresos": ingresos_totales_i,
        "Costos operativos": costos_series_i,
        "Utilidad": utilidad_anual_i,
    }).set_index("Año"))

    # Métricas clave
    util_y1 = int(utilidad_anual_i[0]) if len(utilidad_anual_i) else 0
    util_acum_oper = int(utilidad_anual_i.sum())
    deuda_y1 = int(serv_tot_i[0]) if len(serv_tot_i) else 0
    deuda_acum = int(serv_tot_i.sum())
    pct_cubierta_y1 = (util_y1 / deuda_y1) if deuda_y1 > 0 else np.nan
    gap_y1 = deuda_y1 - util_y1

    m1, m2, m3 = st.columns(3)
    m1.metric("Utilidad **Año 1** (Ing − Costos)", pesos(util_y1))
    m2.metric("Utilidad **acumulada** (operativa)", pesos(util_acum_oper))
    m3.metric("% deuda cubierta con la **utilidad Año 1**", f"{(pct_cubierta_y1*100):.1f}%" if not np.isnan(pct_cubierta_y1) else "—")

    n1, n2, n3 = st.columns(3)
    n1.metric("Servicio de deuda **Año 1**", pesos(deuda_y1))
    n2.metric("Brecha Año 1 (Deuda − Utilidad)", pesos(gap_y1))
    n3.metric("Servicio de deuda **acumulado**", pesos(deuda_acum))

    # Calendario anual — versión legible con comas (texto)
    st.subheader("Calendario anual — Servicio de deuda y caja (resumen)")
    resumen_df = pd.DataFrame({
        "Año": years,
        "Ingresos": ingresos_totales_i,
        "Costos": costos_series_i,
        "Utilidad": utilidad_anual_i,
        "Deuda pública": serv_pub_i,
        "Deuda privada": serv_prv_i,
        "Deuda total": serv_tot_i,
        "Flujo neto": flujo_neto_i,
        "Efectivo acumulado": efectivo_acum_i,
    }).set_index("Año")

    show_res = resumen_df.copy()
    for c in show_res.columns:
        show_res[c] = show_res[c].map(with_commas)
    st.dataframe(show_res, use_container_width=True)

    with st.expander("Detalle por actor (anual)"):
        detalle_df = pd.DataFrame({
            "Año": years,
            "FONATUR": serv_fon_i,
            "BANOBRAS-FONADIN": serv_ban_i,
            "SHCP": serv_shcp_i,
            "Gob. Estado": serv_edo_i,
            "Privado": serv_pri_i,
        }).set_index("Año")
        show_det = detalle_df.copy()
        for c in show_det.columns:
            show_det[c] = show_det[c].map(with_commas)
        st.dataframe(show_det, use_container_width=True)

    st.caption("La utilidad es **operativa** (Ingresos − Costos). El servicio de deuda se calcula con pagos mensuales (r/12) y se **presenta anual**. Todo en **pesos constantes**.")
