# app.py — Modelo Cablebús (simple y claro, pesos constantes)
# Ejecuta: streamlit run app.py

import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# CONFIG & ESTILO
# ==============================
st.set_page_config(page_title="Modelo Cablebús — Tablero Financiero (Simple)", layout="wide")

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

# ==============================
# UTILIDADES
# ==============================
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

def _to_float(s, default: float) -> float:
    if s is None: return float(default)
    if isinstance(s, (int, float, np.floating)): return float(s)
    s = str(s).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float(default)

def float_input(label: str, key: str, default: float, help: str | None = None, decimals: int = 2) -> float:
    initial = st.session_state.get(key, default)
    txt = st.text_input(label, value=f"{float(initial):.{decimals}f}", key=f"{key}__txt", help=help)
    val = _to_float(txt, default)
    st.session_state[key] = val
    return val

def percent_input(label: str, key: str, default_percent: float, help: str | None = None, decimals: int = 2) -> float:
    """Devuelve fracción (0-1). El usuario teclea % con punto."""
    initial = st.session_state.get(key, default_percent)
    txt = st.text_input(f"{label} (%)", value=f"{float(initial):.{decimals}f}", key=f"{key}__pct", help=help)
    val_pct = _to_float(txt, default_percent)
    st.session_state[key] = val_pct
    return max(0.0, float(val_pct)) / 100.0

def df_wide(data, **kwargs):
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)

# ==============================
# PARÁMETROS ESTRUCTURALES (FIJOS)
# ==============================
COSTO_BASE_POR_KM = 450_000_000  # 450 millones por km (pesos constantes)
LONGITUD_KM = 10.1               # fija (no editable)

# ==============================
# LAYOUT GENERAL: MAIN + RAIL DERECHO
# ==============================
col_main, col_rail = st.columns([0.76, 0.24], gap="large")

with col_rail:
    st.markdown('<div class="rail-derecho">', unsafe_allow_html=True)
    st.subheader("Ajustes globales (multiplicadores)")
    st.caption("Afectan proporcionalmente a todo el modelo.")
    dem_x   = st.slider("Escala de Demanda (×)",      0.50, 1.50, st.session_state.get("dem_x",   1.00), 0.05, key="dem_x")
    capkm_x = st.slider("Escala de Costo por km (×)", 0.50, 1.50, st.session_state.get("capkm_x", 1.00), 0.05, key="capkm_x")
    opex_x  = st.slider("Escala de OPEX (×)",         0.50, 1.50, st.session_state.get("opex_x",  1.00), 0.05, key="opex_x")
    st.caption(
        "- **Demanda ×**: multiplica el aforo diario base.\n"
        "- **Costo por km ×**: multiplica $450M/km (impacta el CAPEX base).\n"
        "- **OPEX ×**: multiplica el OPEX del Año 1 (todo el trayecto)."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown("## 🚡 Modelo Cablebús — Tablero Financiero (Simple)")
    st.caption("Pesos constantes. Longitud fija **10.1 km**; costo base **$450,000,000 por km**. Sobrecosto (predios + otros) como **%** único.")

    # ==============================
    # TABS
    # ==============================
    T_INV, T_ING, T_OPEX, T_RES, T_GLOS = st.tabs(["Inversión", "Ingresos", "Costos Operativos", "Resultados", "Glosario"])

    # ==============================
    # INVERSIÓN (costo por km + sobrecosto)
    # ==============================
    with T_INV:
        st.markdown("### Inversión resumida")
        st.write("**Supuestos fijos:** Longitud del proyecto = **10.1 km**.")

        costo_km_aj = COSTO_BASE_POR_KM * float(st.session_state.get("capkm_x", 1.0))
        overrun_frac = percent_input(
            "Sobrecosto estimado por predios y otros (sobre CAPEX base)",
            key="overrun_pct", default_percent=10.0,
            help="Ej.: 10% sobre el CAPEX base (costo por km ajustado × 10.1 km)."
        )

        capex_base  = costo_km_aj * LONGITUD_KM
        sobrecosto  = capex_base * overrun_frac
        capex_total = capex_base + sobrecosto

        st.session_state["capex_total"]      = float(capex_total)
        st.session_state["capex_base"]       = float(capex_base)
        st.session_state["costo_km_ajustado"]= float(costo_km_aj)

        inv_tbl = pd.DataFrame({
            "Concepto": [
                "Costo por km (ajustado)", "Longitud (km)", "CAPEX base (sin sobrecosto)",
                "Sobrecosto (predios + otros)", "CAPEX total"
            ],
            "Valor": [
                pesos(costo_km_aj), f"{LONGITUD_KM:.1f}",
                pesos(capex_base), pesos(sobrecosto), pesos(capex_total)
            ]
        })
        df_wide(inv_tbl)
        st.caption(f"Costo por km base: {pesos(COSTO_BASE_POR_KM)}  •  Multiplicador aplicado: **{float(st.session_state.get('capkm_x',1.0)):.2f}×**.")

    # ==============================
    # INGRESOS (separado)
    # ==============================
    with T_ING:
        st.markdown("### Supuestos de ingresos")
        c1, c2, c3 = st.columns(3)
        with c1:
            aforo_diario = int(st.number_input("Aforo diario total (boletos)", min_value=0, value=24000, step=500, key="aforo_diario"))
        with c2:
            tarifa_prom  = float_input("Tarifa promedio (MXN/boleto)", key="tarifa_prom", default=20.0)
        with c3:
            ing_comp_anual = float_input("Ingresos complementarios anuales (rentas/otros)", key="ing_comp", default=0.0)
        g_demand = percent_input("Crecimiento anual de demanda", key="g_demanda", default_percent=3.0)

        # Guardamos base; el multiplicador de Demanda × se aplica en Resultados
        st.session_state["ingresos_params"] = {
            "aforo_diario": aforo_diario,
            "tarifa_prom": tarifa_prom,
            "ing_comp_anual": ing_comp_anual,
            "g_demand": g_demand
        }

    # ==============================
    # COSTOS OPERATIVOS (separado)
    # ==============================
    with T_OPEX:
        st.markdown("### Supuestos de costos operativos")
        opex_base_y1 = float_input("OPEX anual (Año 1)", key="opex_base_y1", default=38_000_000.0)
        g_opex       = percent_input("Crecimiento anual de OPEX", key="g_opex", default_percent=2.0)
        st.caption("El multiplicador **OPEX ×** del panel derecho se aplica además del crecimiento anual.")
        st.session_state["opex_params"] = {"opex_base_y1": opex_base_y1, "g_opex": g_opex}

    # ==============================
    # RESULTADOS (KPIs + 2 gráficas)
    # ==============================
    with T_RES:
        st.markdown("### Resultados")
        horiz = int(st.number_input("Horizonte (años)", min_value=1, value=20, step=1, key="horiz"))

        # Cargar insumos
        ingp = st.session_state.get("ingresos_params", {"aforo_diario":24000,"tarifa_prom":20.0,"ing_comp_anual":0.0,"g_demand":0.03})
        opxp = st.session_state.get("opex_params", {"opex_base_y1":38_000_000.0,"g_opex":0.02})
        dem_mult  = float(st.session_state.get("dem_x", 1.0))
        opex_mult = float(st.session_state.get("opex_x", 1.0))

        # Series (1..N)
        years = np.arange(1, horiz + 1)

        # Ingresos por boletaje Año 1 (aplica Demanda ×)
        ing_boletaje_y1 = (ingp["aforo_diario"] * dem_mult) * 365.0 * ingp["tarifa_prom"]
        ingresos = np.array([ing_boletaje_y1 * ((1 + ingp["g_demand"]) ** (t - 1)) + ingp["ing_comp_anual"] for t in years], dtype=float)

        # OPEX Año 1 (aplica OPEX ×)
        opex_y1 = opxp["opex_base_y1"] * opex_mult
        opex = np.array([opex_y1 * ((1 + opxp["g_opex"]) ** (t - 1)) for t in years], dtype=float)

        utilidad = ingresos - opex

        # Efectivo acumulado (recuperación simple; año 0 = -CAPEX)
        capex_total = float(st.session_state.get("capex_total", 0.0))
        efectivo = -capex_total + np.cumsum(utilidad)

        # KPIs Año 1
        y1_ing, y1_ox, y1_u = int(round(ingresos[0])), int(round(opex[0])), int(round(utilidad[0]))
        margen_y1 = (utilidad[0] / ingresos[0]) if ingresos[0] > 0 else np.nan

        # Payback simple (primer año con efectivo ≥ 0)
        br_idx = np.argmax(efectivo >= 0) if np.any(efectivo >= 0) else None
        year_break_even = int(years[br_idx]) if (br_idx is not None and efectivo[br_idx] >= 0) else None

        # KPIs arriba
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAPEX total", pesos(capex_total))
        k2.metric("Ingresos Año 1", pesos(y1_ing))
        k3.metric("OPEX Año 1", pesos(y1_ox))
        k4.metric("Utilidad operativa Año 1", pesos(y1_u))
        k5, k6 = st.columns(2)
        k5.metric("Margen operativo Año 1", f"{(margen_y1*100):.1f}%" if not np.isnan(margen_y1) else "—")
        k6.metric("Recuperación simple (años)", f"{year_break_even}" if year_break_even else "—")

        # Gráficas
        st.markdown("#### Ingresos, OPEX y Utilidad operativa")
        plot_df = pd.DataFrame({
            "Año": years,
            "Ingresos": np.rint(ingresos).astype("int64"),
            "OPEX": np.rint(opex).astype("int64"),
            "Utilidad": np.rint(utilidad).astype("int64"),
        }).set_index("Año")
        st.line_chart(plot_df)

        st.markdown("#### Efectivo acumulado (recuperación simple)")
        eff_df = pd.DataFrame({"Año": years, "Efectivo acumulado": np.rint(efectivo).astype("int64")}).set_index("Año")
        st.line_chart(eff_df)

        # Exportación simple (opcional)
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
                return buf.getvalue(), "modelo_cablebus_simple.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, df in sheets.items():
                    zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8"))
            return zbuf.getvalue(), "modelo_cablebus_simple.zip", "application/zip"

        sheets = {
            "Inversion": pd.DataFrame({
                "Variable": ["Costo por km (ajustado)", "Longitud (km)", "CAPEX base", "Sobrecosto (%)", "CAPEX total"],
                "Valor": [st.session_state.get("costo_km_ajustado", COSTO_BASE_POR_KM), LONGITUD_KM,
                          st.session_state.get("capex_base", 0.0), st.session_state.get("overrun_pct", 10.0),
                          st.session_state.get("capex_total", 0.0)]
            }),
            "Serie_Anual": plot_df.reset_index(),
            "Efectivo_Acumulado": eff_df.reset_index()
        }
        data_bytes, fname, mime = build_export(sheets)
        st.download_button("⬇️ Descargar resumen (XLSX o ZIP)", data=data_bytes, file_name=fname, mime=mime)

    # ==============================
    # GLOSARIO (mínimo)
    # ==============================
    with T_GLOS:
        st.markdown("""
### Términos (mínimo)
- **CAPEX**: Inversión inicial para construir (aquí: costo por km × 10.1 + sobrecosto).
- **OPEX**: Gasto operativo anual (Año 1) y su trayectoria con crecimiento.
- **Utilidad operativa**: Ingresos − OPEX.
- **Recuperación simple**: Años para que el **efectivo acumulado** (−CAPEX + utilidades anuales) sea ≥ 0.
        """)
