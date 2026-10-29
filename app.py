"""
BondAI · aplicación en Streamlit con seis pestañas: Curva en dólares, ONs, Tesoro en pesos, Mi portafolio, Flujos de fondos e
Informe semanal con IA.
Ejecutar:  pip install -r requirements.txt   →   streamlit run app.py
Publicada en internet: ver README.md (GitHub + Streamlit Community Cloud).
Usa las mismas funciones que el notebook: bondai.py, pesos.py, mercado.py, dolares.py, licitaciones.py, externos.py, portafolio.py,
herramientas.py e informe.py deben estar en la misma carpeta.
"""
import io
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import bondai as B
import dolares as DL
import externos as X
import herramientas as H
import informe as I
import licitaciones as L
import mercado as M
import pesos as P
import portafolio as PF

st.set_page_config(page_title="BondAI · Renta fija con IA", page_icon="📈", layout="wide")


def ver_grafico(fig, destino=None):
    """Muestra un gráfico de matplotlib como PNG a 110 dpi (st.pyplot usa 200: el doble de peso para el navegador) y libera la figura."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    (destino or st).image(buf, width="stretch")


# ── datos y cálculos (cacheados para que la app responda rápido) ──
@st.cache_data(ttl=900, show_spinner="Bajando datos de BYMA y BCRA…")
def cargar(en_vivo):
    d = B.cargar_datos(usar_api=en_vivo)
    d["bajado"] = datetime.now(timezone(timedelta(hours=-3)))  # hora en que se bajaron los datos
    return d


@st.cache_data(ttl=900, show_spinner="Calculando los bonos en dólares…")
def calcular(en_vivo, moneda="mep"):
    tabla_usd, liq_usd, _ = B.analisis_usd(cargar(en_vivo), moneda)
    return tabla_usd, liq_usd


@st.cache_data(ttl=900, show_spinner="Bajando bonos del Tesoro en pesos…")
def cargar_pesos(en_vivo):
    return P.cargar(usar_api=en_vivo)


@st.cache_data(ttl=900, show_spinner="Bajando los paneles de BYMA…")
def cargar_paneles(en_vivo):
    return M.cargar_paneles(usar_api=en_vivo)




@st.cache_data(ttl=900, show_spinner="Bajando el histórico de los dólares (BYMA y BCRA)…")
def cargar_dolares(en_vivo):
    return DL.cargar_dolares(usar_api=en_vivo)


# Bonos en pesos: el cálculo tarda unos 15 segundos, así que se guarda para cada combinación de supuestos.
# cache_resource (y no cache_data) porque el resultado incluye funciones (las curvas ajustadas), que no se pueden copiar.
@st.cache_resource(ttl=900, show_spinner="Calculando los 55 bonos en pesos…")
def calcular_pesos(en_vivo, plazo, precio, tamar, infl, dev, curva_be="auto"):
    d = cargar(en_vivo)
    return P.calcular(cargar_pesos(en_vivo), plazo, precio, tamar, infl, ipc_publicado=d["macro"]["ipc"].dropna().to_dict(),
                      rem_mensual=d["rem_mensual"]["mediana"].dropna().to_dict(), dev=dev, curva_be=curva_be)




@st.cache_data(ttl=900, show_spinner="Bajando los futuros de dólar de A3…")
def cargar_futuros(en_vivo):
    return X.cargar_futuros(en_vivo)


@st.cache_data(ttl=3600, show_spinner="Bajando el riesgo país…")
def cargar_riesgo(en_vivo):
    return X.cargar_riesgo_pais(en_vivo)


@st.cache_data(ttl=3600, show_spinner="Bajando la curva del Tesoro de EE.UU.…")
def cargar_ust(en_vivo, fecha):
    return X.cargar_ust(en_vivo, fecha)


@st.cache_data(ttl=300, show_spinner="Bajando las obligaciones negociables de data912…")
def cargar_ons(en_vivo):
    return X.cargar_ons(en_vivo)


@st.cache_data(ttl=3 * 3600, show_spinner="Buscando licitaciones nuevas en la Secretaría de Finanzas…")
def cargar_licitaciones(en_vivo, hoy):
    dp = cargar_pesos(en_vivo)
    return L.actualizar(en_vivo, dp["cer"], dp["tamar"], dp["a3500"], hoy)


@st.cache_data(ttl=3600, show_spinner=False)
def informe_semana(en_vivo, hoy, ons=()):
    dp = cargar_pesos(en_vivo)
    riesgo, _ = cargar_riesgo(en_vivo)
    sub, col, _, _ = cargar_licitaciones(en_vivo, hoy)
    return I.semana(en_vivo, dp["cer"], dp["tamar"], riesgo, DL.series(cargar_dolares(en_vivo)), (sub, col), ons=ons)




@st.cache_data(ttl=900, show_spinner="Bajando caución, plazo fijo, billeteras y fondos money market…")
def cargar_dinero(en_vivo):
    return H.cargar_dinero(en_vivo)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cargar_fichas_ons(en_vivo, raices):
    return X.cargar_fichas(list(raices), usar_api=en_vivo)


@st.cache_data(ttl=3600, show_spinner="Bajando el histórico de precios de cada bono (BYMA)…")
def historicos_pf(en_vivo, simbolos):
    return PF.historicos(list(simbolos), usar_api=en_vivo)


# colores de las variaciones: (factor para llevar a decimal, piso de la escala, invertir el signo)
CALOR = {"var": (1, 0.005, False), "var_periodo": (1, 0.01, False), "pb_rend": (1e-4, 0.001, True), "premio_pb": (1e-4, 0.01, False),
         "pct": (1, 0.003, False), "pp": (1, 0.03, False)}


def fondo(v, vmax):
    """Verde si subió, rojo si bajó; más intenso cuanto más grande el movimiento (relativo al mayor de la tabla)."""
    if v is None or pd.isna(v) or abs(v) < 5e-5:
        return ""
    a = 0.08 + 0.44 * np.sqrt(min(1.0, abs(v) / vmax))
    return f"background-color: rgba({'22,160,90' if v > 0 else '214,69,69'},{a:.2f})"


def mostrar(df, formatos, nombres=None, na="–", calor=None):
    """Tabla con formato propio (Streamlit muestra 'None' en celdas vacías aunque el Styler tenga na_rep).
    calor: {columna: tipo de CALOR} para pintar la celda de verde o rojo según la variación."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        f = formatos.get(col)
        out[col] = [na if pd.isna(v) else (f.format(v) if f else str(v)) for v in df[col]]
    est = pd.DataFrame("", index=out.index, columns=out.columns)
    for col, tipo in (calor or {}).items():
        if col not in df.columns:
            continue
        k, piso, inv = CALOR[tipo]
        vals = pd.to_numeric(df[col], errors="coerce") * k * (-1 if inv else 1)
        vmax = max(piso, float(vals.abs().max())) if vals.notna().any() else piso
        est[col] = [fondo(v, vmax) for v in vals]
    if nombres:
        out, est = out.rename(columns=nombres), est.rename(columns=nombres)
    if calor and not out.index.is_unique:  # el Styler necesita un índice sin repetidos: el índice pasa a ser la primera columna
        nom = out.index.name or ""
        out, est = out.reset_index(names=nom), est.reset_index(names=nom)
        est[nom] = ""
        st.dataframe(out.style.apply(lambda _: est, axis=None), width="stretch", hide_index=True)
        return
    st.dataframe(out.style.apply(lambda _: est, axis=None) if calor else out, width="stretch")


# ── barra lateral ──
st.sidebar.title("BondAI")
st.sidebar.caption("Terminal de renta fija argentina con IA · Programación para Economistas")
en_vivo = st.sidebar.toggle("Datos en vivo (APIs de BYMA y BCRA)", value=True,
                            help="Si una API no responde, se usa el respaldo del 09/10/2026 para esa parte.")
moneda = st.sidebar.radio("Valuar los bonos en dólares en", ["mep", "ccl"], horizontal=True,
                          format_func={"mep": "Dólar MEP (D)", "ccl": "Dólar cable (C)"}.get,
                          help="MEP: precios de los tickers terminados en D (AL30D…). Cable: tickers terminados en C (AL30C…).")
SUF = "C" if moneda == "ccl" else "D"
if st.sidebar.button("Actualizar datos ahora"):
    st.cache_data.clear()
    calcular_pesos.clear()


def borrar_precios():
    """Borra de la memoria solo los precios y tasas (no el volumen, que tarda en volver a bajarse)."""
    for f in (cargar, calcular, cargar_pesos, cargar_paneles, cargar_dolares, calcular_pesos, cargar_futuros, cargar_ons, cargar_riesgo):
        f.clear()


# Actualización automática: mientras la página esté abierta, cada 15 minutos vuelve a bajar precios y tasas.
# (Sin esto, los datos igual se renuevan solos a los 15 minutos, pero recién cuando alguien toca algo.)
if st.sidebar.toggle("Actualizar sola cada 15 minutos", value=False, disabled=not en_vivo):
    st.session_state.setdefault("ultima_auto", time.time())

    @st.fragment(run_every="60s")
    def reloj():
        if time.time() - st.session_state["ultima_auto"] >= 15 * 60:
            st.session_state["ultima_auto"] = time.time()
            borrar_precios()
            st.rerun()

    with st.sidebar:
        reloj()
D = cargar(en_vivo)
st.sidebar.markdown("**Fuentes usadas**")
for k, v in D["fuente"].items():
    st.sidebar.write(f"· {k}: {v}")
for k, v in cargar_pesos(en_vivo)["fuente"].items():
    st.sidebar.write(f"· pesos/{k}: {v}")
PAN = cargar_paneles(en_vivo)
st.sidebar.write(f"· paneles 24 hs: {PAN['fuente']}")


def px_pan(sym):
    """Precio de un símbolo en el panel de 24 hs: VWAP, último, cierre o, antes de la apertura, el cierre anterior."""
    x = PAN["t1"].get(sym) or []
    return next((float(v) for v in x[:4] if v and v > 0), np.nan)


def dolar_bono(simbolo_usd):
    """MEP (AL30D) o cable (AL30C) de hoy; si el panel todavía no tiene precios, el último de la serie histórica."""
    v = px_pan("AL30") / px_pan(simbolo_usd)
    if np.isfinite(v):
        return v
    s_ = DL.series(cargar_dolares(en_vivo))["mep" if simbolo_usd == "AL30D" else "cable"].dropna()
    return float(s_.iloc[-1]) if len(s_) else np.nan
st.title("BondAI · Renta fija argentina con IA")
tabla_usd, liq_usd = calcular(en_vivo, moneda)
FUT, F_FUT = cargar_futuros(en_vivo)
RIESGO, F_RIESGO = cargar_riesgo(en_vivo)
ONS, F_ONS = cargar_ons(en_vivo)
mep_on = dolar_bono("AL30D")
FICH = cargar_fichas_ons(en_vivo, tuple(sorted(set(ONS["raiz"])))) if len(ONS) else {}
TO = X.tabla_ons(ONS, mep_on, FICH) if len(ONS) else pd.DataFrame(columns=["emisor", "precio_mep", "precio_cable", "volumen_usd"])
ON_OK = X.ons_calculables(TO, FICH, liq_usd, moneda)   # ONs en dólares con tasa fija: van al portafolio, a los flujos y al informe
ONS_EV = tuple((t, r["emisor"], r["clase"], tuple(r["flujos"]), r["vence"], bool(r["frec_supuesta"])) for t, r in ON_OK.iterrows()) if len(ON_OK) else ()
for k, v in [("futuros", F_FUT), ("riesgo país", F_RIESGO), ("ONs", F_ONS)]:
    st.sidebar.write(f"· {k}: {v}")
st.sidebar.caption(f"Fecha de análisis: {D['fecha']:%d/%m/%Y} · datos bajados a las {D['bajado']:%H:%M} (hora argentina)")
st.sidebar.caption("Herramienta académica. No es recomendación de inversión.")

c1, c5, c2, c3, c4 = st.columns([1, 1, 1.25, 0.9, 0.9])
c1.metric("TIR mediana soberanos USD", f"{tabla_usd['tir'].median():.2%}")
# supuestos por defecto de «Tesoro en pesos» (24 hs, VWAP): el mismo cálculo queda en memoria y la pestaña lo reusa
DP = cargar_pesos(en_vivo)
TAMAR0 = round(float(DP["tamar"].sort_index().iloc[-5:].mean()), 2)
INFL0 = round(float(D["rem_mensual"]["mediana"].dropna().iloc[-1]), 2) / 100
RP0 = calcular_pesos(en_vivo, "24HS", "px", TAMAR0, INFL0, None)
BE0 = RP0["cer"][np.isfinite(RP0["cer"]["be_mensual"].astype(float))]
if len(BE0):
    ult = BE0.iloc[-1]
    c2.metric(f"Inflación breakeven {BE0.index[-1]}", f"{ult['be_mensual']:.2%} mensual", f"{(1 + ult['be_mensual']) ** 12 - 1:.1%} anual · REM {ult['rem_mensual']:.2%}",
              delta_color="off", help="Inflación mensual que el mercado descuenta para los meses todavía sin IPC publicado, despejada del bono CER "
                                      "cupón cero más largo que tiene LECAP o BONCAP con la cual compararse (pestaña «Tesoro en pesos», sección CER).")
rp_ult, rp_ant = float(RIESGO.iloc[-1]), float(RIESGO.iloc[-2])
c5.metric("Riesgo país", f"{rp_ult:,.0f} pb".replace(",", "."), f"{rp_ult - rp_ant:+.0f} pb al {RIESGO.index[-1]:%d/%m}", delta_color="inverse",
          help="EMBI de JP Morgan: cuántos puntos básicos (100 pb = 1 punto porcentual) más rinden los bonos argentinos en dólares que los "
               "del Tesoro de EE.UU. Sube cuando el mercado de afuera ve más riesgo.")
c3.metric("REM próximo mes", f"{D['rem_mensual']['mediana'].iloc[0]:.2f}%", f"{D['rem_mensual'].index[0]}", delta_color="off")
c4.metric("Último IPC", f"{D['macro']['ipc'].dropna().iloc[-1]:.1f}%", D["macro"]["ipc"].dropna().index[-1], delta_color="off")


(tab1, tabN, tabP, tabPF, tabF, tab4) = st.tabs(
    ["Curva en dólares", "ONs", "Tesoro en pesos", "Mi portafolio", "Flujos de fondos", "Informe semanal con IA"])

# ── 1. curva USD ──
with tab1:
    izq, der = st.columns([1.2, 1])
    with izq:
        st.subheader("Métricas por bono")
        st.caption(f"Precios en dólar {'cable (tickers C)' if moneda == 'ccl' else 'MEP (tickers D)'}, liquidación {liq_usd:%d/%m/%Y}. "
                   "Spread = TIR − curva Nelson-Siegel de su legislación. Rend. corriente = cupón anual vigente / precio (no incluye la ganancia "
                   "o pérdida de capital ni la reinversión). Paridad = precio / valor técnico (residual + intereses corridos 30/360). "
                   "Var.: verde si subió, rojo si bajó; más intenso, más movimiento.")
        tu = tabla_usd.copy()
        tu["var"] = [(PAN["t1"].get(t + SUF) or [0] * 8)[7] for t in tu.index]  # variación del día que informa BYMA
        mostrar(tu[["familia", "precio", "var", "tir", "current_yield", "md", "convexidad", "paridad", "vt", "spread_pb"]],
                {"precio": "{:.2f}", "var": "{:+.2%}", "tir": "{:.2%}", "current_yield": "{:.2%}", "md": "{:.2f}", "convexidad": "{:.1f}", "paridad": "{:.1%}",
                 "vt": "{:.2f}", "spread_pb": "{:+.0f}"},
                {"familia": "Familia", "precio": "Precio", "var": "Var.", "tir": "TIR", "current_yield": "Rend. corriente", "md": "MD",
                 "convexidad": "Convexidad", "paridad": "Paridad", "vt": "Valor técnico", "spread_pb": "Spread (pb)"}, calor={"var": "var"})
    with der:
        st.subheader("Curva TIR vs duration")
        fig, ax = plt.subplots(figsize=(6, 4))
        for fam, col, mk in [("Bonares", B.COLORES["bonar"], "o"), ("Tesoro", B.COLORES["tesoro"], "^"), ("Globales", B.COLORES["global"], "s")]:
            s = tabla_usd[tabla_usd["familia"] == fam]
            ax.scatter(s["md"], s["tir"] * 100, color=col, marker=mk, label=fam, zorder=3)
            for t, r in s.iterrows():
                ax.annotate(t, (r["md"], r["tir"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=7)
        g = np.linspace(0.9, 5.8, 80)
        ax.plot(g, B.nelson_siegel(g, *tabla_usd.attrs["Local"]) * 100, color=B.COLORES["bonar"], alpha=.6)
        ax.plot(g, B.nelson_siegel(g, *tabla_usd.attrs["Nueva York"]) * 100, color=B.COLORES["global"], alpha=.6)
        ax.set_xlabel("Duration modificada"); ax.set_ylabel("TIR %"); ax.legend(fontsize=8)
        ver_grafico(fig)

    st.subheader("¿Cuánto ganás o perdés según la TIR a la que vendas?")
    st.caption("Comprás hoy al precio de la tabla, cobrás los cupones y amortizaciones hasta la fecha de venta y vendés al precio que corresponde "
               "a la TIR de salida. Si la TIR baja, el precio sube y ganás más; si sube, perdés. «TIR igual» es lo que ganás si nada cambia "
               "(el devengamiento, o carry). Los pagos cobrados no se reinvierten; en dólares, antes de comisiones.")
    r1_, r2_, r3_ = st.columns([1, 1.3, 1.6])
    hz_r = r1_.date_input("Vendés el", liq_usd + timedelta(days=182), min_value=liq_usd + timedelta(days=1), format="DD/MM/YYYY", key="ret_hz")
    modo_r = r2_.radio("Escenarios", ["pb", "tir"], horizontal=True, key="ret_modo", format_func={"pb": "Cambio de TIR (pb)", "tir": "TIR de salida (%)"}.get)
    fam_r = r3_.multiselect("Bonos", ["Bonares", "Globales", "Tesoro"], default=["Bonares", "Globales", "Tesoro"], key="ret_fam")
    esc_r = [-300, -200, -100, -50, 0, 50, 100, 200, 300] if modo_r == "pb" else [6, 7, 8, 9, 10, 11, 12, 13, 14]
    if fam_r:
        TR_ = H.tabla_retornos(tabla_usd[tabla_usd["familia"].isin(fam_r)], liq_usd, hz_r, esc_r, modo_r)
        nom_r = {e: ((f"{e:+d} pb" if e else "TIR igual") if modo_r == "pb" else f"TIR {e}%") for e in esc_r}
        mostrar(TR_[["familia", "precio", "tir", "md"] + esc_r].rename(columns=nom_r),
                {"precio": "{:.2f}", "tir": "{:.2%}", "md": "{:.2f}", **{nom_r[e]: "{:+.1%}" for e in esc_r}},
                {"familia": "Familia", "precio": "Precio hoy", "tir": "TIR hoy", "md": "MD"}, calor={nom_r[e]: "var" for e in esc_r})
        st.caption(f"Retorno total en dólares del {liq_usd:%d/%m/%Y} al {hz_r:%d/%m/%Y} ({(hz_r - liq_usd).days} días). Regla rápida: retorno ≈ "
                   "carry − MD × cambio de TIR; los bonos con más duración (a la derecha de la curva) son los que más se mueven.")


# ── gráficos y formatos comunes ──
VERDE, ROJO, GRIS, AZUL, VIOLETA = "#16a05a", "#d64545", "#8a96aa", "#2a78d6", "#7a5cf0"


def num_es(v, d=2):
    """Número con punto de miles y coma decimal: 1.234,56."""
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


num_ar = lambda v, d=0: num_es(v, d)   # sin decimales por defecto


def ejes_log_dias(ax, xs):
    ticks = [t for t in [7, 15, 30, 60, 90, 180, 365, 540, 730, 1000, 1500, 2000, 3000, 5000, 7000] if min(xs) * 0.85 <= t <= max(xs) * 1.15]
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}".replace(",", ".") for t in ticks])
    ax.minorticks_off()


def pb(v):
    """Puntos básicos con signo, sin '-0'."""
    return "0 pb" if abs(v) < 0.5 else f"{v:+.0f} pb"


def etiquetar(ax, xs, ys, textos, extra=None, fs=7):
    for i, (x, y, t) in enumerate(zip(xs, ys, textos)):
        ax.annotate(t + (f"  {extra[i]}" if extra is not None else ""), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=fs)


def grafico_pesos(df, ycol, curva=None, color="#2a78d6", ylab="TIREA", xlab="Años al vencimiento"):
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    d = df[np.isfinite(df[ycol].astype(float))]
    ax.scatter(d["x"], d[ycol] * 100, color=color, zorder=3, s=28)
    etiquetar(ax, d["x"], d[ycol] * 100, d.index)
    if curva is not None and len(d):
        g = np.linspace(max(0.03, d["x"].min() * 0.8), d["x"].max() * 1.04, 80)
        ax.plot(g, [curva(x) * 100 for x in g], color="#2a78d6", alpha=.55, lw=1.5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab + " %")
    fig.tight_layout()
    return fig


def grafico_cer_tir(df):
    """TIR real contra el plazo (escala logarítmica): verde rinde más que la curva (barato), rojo rinde menos (caro)."""
    d = df[df["dias"] >= 7]
    base = d[d["dias"] >= 30]
    fit = M.ajustar_curva(base["dias"], base["tir_real"], "auto", 365) if len(base) >= 3 else None
    res = d["tir_real"] - (fit[0](d["dias"]) if fit else d["tir_real"])
    col = [GRIS if dd < 30 else (VERDE if r > 0 else ROJO) for dd, r in zip(d["dias"], res)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.scatter(d["dias"], d["tir_real"] * 100, c=col, s=[45 if c else 30 for c in d["canje"]], marker="o", zorder=3)
    etiquetar(ax, d["dias"], d["tir_real"] * 100, d.index)
    if fit:
        g = np.geomspace(base["dias"].min(), base["dias"].max(), 120)
        ax.plot(g, fit[0](g) * 100, color=AZUL, lw=1.6, label=f"Curva {fit[1].lower()}")
        ax.legend(fontsize=8, loc="lower right")
    ejes_log_dias(ax, d["dias"]); ax.set_xlabel("Plazo (días, escala logarítmica)"); ax.set_ylabel("TIR real %")
    ax.grid(alpha=.25); fig.tight_layout()
    return fig


def grafico_be(df, max_cero):
    """Inflación breakeven mensual por plazo con su curva y banda de ±1σ; arriba de la curva el CER está caro."""
    d = df[np.isfinite(df["be_mensual"].astype(float))]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if d.empty:
        ax.text(.5, .5, "Sin bonos con breakeven en ese plazo", ha="center", transform=ax.transAxes); return fig
    fit = M.ajustar_curva(d["plazo_be"], d["be_mensual"], "auto", 365)
    res = d["be_mensual"] - (fit[0](d["plazo_be"]) if fit else d["be_mensual"])
    ax.scatter(d["plazo_be"], d["be_mensual"] * 100, c=[ROJO if r > 0 else VERDE for r in res], s=40, zorder=3)
    etiquetar(ax, d["plazo_be"], d["be_mensual"] * 100, d.index)
    g = np.linspace(d["plazo_be"].min(), d["plazo_be"].max(), 80)
    if fit:
        ax.plot(g, fit[0](g) * 100, color=AZUL, lw=1.6, label=f"Curva {fit[1].lower()} · ±1σ = {fit[2] * 1e4:.0f} pb")
        ax.fill_between(g, (fit[0](g) - fit[2]) * 100, (fit[0](g) + fit[2]) * 100, color=AZUL, alpha=.12, lw=0)
    ax.plot(d["plazo_be"], d["rem_mensual"] * 100, ls="--", color=VIOLETA, lw=1.3, label="REM para los mismos meses")
    ax.set_xlabel("Plazo (días)"); ax.set_ylabel("Inflación breakeven mensual %"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def grafico_margen_tamar(tam, dual, tamar_sup):
    """Margen sobre la TAMAR supuesta contra el plazo, con los duales (*) y una curva de mejor ajuste."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    pts = []
    for df, col, mk, dual_ in ((tam, M.COLORES["tamar"], "o", False), (dual, M.COLORES["dual"], "^", True)):
        if df.empty:
            continue
        ax.scatter(df["dias"], df["margen_s_tamar"], color=col, marker=mk, s=40, zorder=3, label="Duales (*)" if dual_ else "TAMAR")
        etiquetar(ax, df["dias"], df["margen_s_tamar"], [t + (" *" if dual_ else "") for t in df.index])
        pts += list(zip(df["dias"], df["margen_s_tamar"]))
    if len(pts) >= 3:
        x, y = zip(*pts)
        fit = M.ajustar_curva(x, y, "auto", 365)
        if fit:
            g = np.linspace(min(x), max(x), 100)
            ax.plot(g, fit[0](g), color=AZUL, lw=1.6, label=f"Curva {fit[1].lower()}")
    ax.axhline(0, color=GRIS, lw=.8, ls="--")
    ax.set_xlabel("Plazo (días)"); ax.set_ylabel(f"Margen s/ TAMAR (TAMAR supuesta {tamar_sup:.2f}%)", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    return fig


# ── Obligaciones negociables (data912) ──
with tabN:
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("ONs con precio", f"{TO['precio_mep'].notna().sum()} en MEP", f"{len(TO)} en total", delta_color="off")
    n2.metric("Volumen en ONs", f"US$ {num_ar(TO['volumen_usd'].sum() / 1e6, 1)} M", "pesos, MEP y cable", delta_color="off")
    med_mep = TO["mep_implicito"].median()
    n3.metric("MEP implícito (mediana)", f"$ {num_ar(med_mep, 2)}" if np.isfinite(med_mep) else "–", f"AL30: $ {num_ar(mep_on, 2)}", delta_color="off",
              help="Precio en pesos / precio en dólares de cada ON. Si una ON da un MEP más bajo, conviene usarla para comprar dólares (o al revés).")
    n4.metric("Canje (mediana)", f"{TO['canje'].median():.1%}" if TO["canje"].notna().any() else "–", "cable sobre MEP", delta_color="off")
    o1, o2, o3 = st.columns([1.4, 1, 1])
    emis = sorted({e for e in TO["emisor"] if e})
    sel_e = o1.multiselect("Emisor", emis, placeholder="Todos")
    busca = o2.text_input("Buscar ticker", placeholder="Ej.: MGC, YMC, IRCP")
    orden = o3.selectbox("Ordenar por", ["volumen_usd", "precio_mep", "var", "spread_puntas"],
                         format_func={"volumen_usd": "Volumen", "precio_mep": "Precio", "var": "Variación", "spread_puntas": "Spread entre puntas"}.get)
    V_ = TO.copy()
    if sel_e:
        V_ = V_[V_["emisor"].isin(sel_e)]
    if busca:
        V_ = V_[V_.index.str.contains(busca.strip().upper(), regex=False)]
    V_ = V_.sort_values(orden, ascending=orden == "spread_puntas")
    mostrar(V_.head(150)[["emisor", "vence", "cupon", "precio_mep", "var", "precio_cable", "precio_pesos", "mep_implicito", "canje", "spread_puntas", "volumen_usd"]],
            {"vence": "{:%d/%m/%Y}", "cupon": "{:.2%}", "precio_mep": "{:.2f}", "var": "{:+.2%}", "precio_cable": "{:.2f}", "precio_pesos": "{:,.0f}",
             "mep_implicito": "{:,.2f}", "canje": "{:+.1%}", "spread_puntas": "{:.2%}", "volumen_usd": "{:,.0f}"},
            {"emisor": "Emisor", "vence": "Vence", "cupon": "Cupón", "precio_mep": "Precio MEP (D)", "var": "Var.", "precio_cable": "Precio cable (C)",
             "precio_pesos": "Precio $ (O)", "mep_implicito": "MEP implícito", "canje": "Canje", "spread_puntas": "Spread compra-venta",
             "volumen_usd": "Volumen US$"}, calor={"var": "var"})
    top = TO.sort_values("volumen_usd", ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([f"{t} · {e}" if e else t for t, e in zip(top.index, top["emisor"])], top["volumen_usd"] / 1e6, color=AZUL)
    ax.set_xlabel("Volumen del día, millones de US$"); ax.grid(alpha=.25, axis="x"); fig.tight_layout(); ver_grafico(fig)
    st.caption(f"Fuente: {F_ONS}. Cada ON cotiza en tres monedas: terminada en O en pesos, en D en dólar MEP y en C en dólar cable, por 100 de "
               "valor nominal. Emisor, vencimiento y cupón: ficha técnica de BYMA de cada ON (las nuevas se consultan solas). Cupón vacío: tasa variable "
               "o no informada en la ficha.")

    st.subheader("Las ONs de los grandes emisores")
    ONS_S = H.spread_ons(ON_OK, tabla_usd)
    if ONS_S is None or ONS_S.empty:
        st.info("Sin precios de ONs en dólares para calcular la TIR (antes de las 11 hs o si data912 no respondió).")
    else:
        em_sel = st.radio("Emisor", H.EMISORES_TOP, horizontal=True, key="on_emisor")
        E_ = ONS_S[ONS_S["emisor"] == em_sel].sort_values("vence")
        if len(E_):
            e1_, e2_, e3_, e4_ = st.columns(4)
            e1_.metric("ONs con precio", f"{len(E_)}", f"vencen entre {E_['vence'].min():%Y} y {E_['vence'].max():%Y}", delta_color="off")
            e2_.metric("TIR mediana", f"{E_['tir'].median():.2%}", f"MD promedio {E_['md'].mean():.2f}", delta_color="off")
            e3_.metric("Spread contra Globales", f"{E_['spread_pb'].median():+.0f} pb", "mediana, a igual duración", delta_color="off",
                       help="TIR de la ON menos la TIR de la curva de Globales (ley Nueva York) a la misma duración. Negativo: la empresa se "
                            "financia más barato que el Estado, algo habitual en las petroleras exportadoras.")
            e4_.metric("Volumen del día", f"US$ {num_ar(E_['volumen_usd'].sum() / 1e3, 0)} mil", delta_color="off")
            mostrar(E_[["clase", "vence", "cupon", "frecuencia", "cuotas", "precio", "tir", "md", "tir_soberano", "spread_pb", "volumen_usd"]],
                    {"vence": "{:%d/%m/%Y}", "cupon": "{:.3%}", "frecuencia": "{:.0f}", "cuotas": "{:.0f}", "precio": "{:.2f}", "tir": "{:.2%}", "md": "{:.2f}",
                     "tir_soberano": "{:.2%}", "spread_pb": "{:+.0f}", "volumen_usd": "{:,.0f}"},
                    {"clase": "Clase", "vence": "Vence", "cupon": "Cupón", "frecuencia": "Pagos/año", "cuotas": "Cuotas", "precio": "Precio MEP",
                     "tir": "TIR", "md": "MD", "tir_soberano": "TIR Globales a igual MD", "spread_pb": "Spread (pb)", "volumen_usd": "Volumen US$"})
        else:
            st.info("Ese emisor no tiene ONs en dólares a tasa fija con precio hoy.")
        fig, ax = plt.subplots(figsize=(11, 4.3))
        resto_ = ONS_S[~ONS_S["emisor"].isin(H.EMISORES_TOP)]
        ax.scatter(resto_["md"], resto_["tir"] * 100, s=12, color="#c9cdda", label="Otras ONs", zorder=2)
        for em_, c_ in zip(H.EMISORES_TOP, ["#2a78d6", "#e0569a", "#1baf7a", "#f0a020", "#7a5cf0", "#d64545"]):
            z_ = ONS_S[ONS_S["emisor"] == em_]
            ax.scatter(z_["md"], z_["tir"] * 100, s=42 if em_ == em_sel else 22, color=c_, label=em_, zorder=4 if em_ == em_sel else 3,
                       edgecolors="black" if em_ == em_sel else "none", linewidths=.6)
            if em_ == em_sel:
                for t_, r_ in z_.iterrows():
                    ax.annotate(t_, (r_["md"], r_["tir"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=7)
        sob_ = tabla_usd[tabla_usd["familia"].isin(["Bonares", "Globales"])]
        g_ = np.linspace(float(sob_["md"].min()), float(sob_["md"].max()), 80)
        ax.plot(g_, B.nelson_siegel(g_, *tabla_usd.attrs["Nueva York"]) * 100, color="black", lw=1.6, label="Curva de Globales (ley NY)", zorder=5)
        ax.plot(g_, B.nelson_siegel(g_, *tabla_usd.attrs["Local"]) * 100, color="black", lw=1.1, ls="--", label="Curva de Bonares (ley local)", zorder=5)
        ax.scatter(sob_["md"], sob_["tir"] * 100, marker="x", color="black", s=26, zorder=5)
        ax.set_ylim(max(0, float(ONS_S["tir"].quantile(.02)) * 100 - 1), max(float(ONS_S["tir"].quantile(.98)), float(sob_["tir"].max())) * 100 + 1)
        ax.set_xlabel("Duración modificada"); ax.set_ylabel("TIR % (en dólares)"); ax.grid(alpha=.25)
        ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False)
        fig.tight_layout(); ver_grafico(fig)
        st.caption("Cada punto es una ON en dólares a tasa fija con su TIR calculada sola (ficha de BYMA y precio MEP de data912); las cruces y líneas "
                   "negras son los soberanos. Las ONs por debajo de la curva de Globales rinden menos que el Estado: el mercado ve a esas empresas "
                   "con menos riesgo (exportan, cobran en dólares). Las ONs muy cortas tienen TIR ruidosa.")

    st.subheader("Calculadora de TIR de una ON")
    st.caption("La app completa cupón, vencimiento y cuotas con la ficha técnica de BYMA; revisalos contra el prospecto (la ficha no siempre "
               "informa la frecuencia de pago ni el cronograma de amortización) y la app calcula la TIR, la duración y la paridad al precio de hoy.")
    con_d = TO[TO["precio_mep"].notna()].sort_values("volumen_usd", ascending=False)
    if con_d.empty:
        st.info("Todavía no hay precios de ONs en dólares (antes de las 11 hs, o data912 no respondió). Probá de nuevo en un rato.")
    else:
        k1, k2, k3, k4, k5, k6 = st.columns([1.2, .8, .9, 1, .8, .8])
        on_sel = k1.selectbox("ON", con_d.index.tolist(), format_func=lambda t: f"{t}D · {TO.loc[t, 'emisor']}" if TO.loc[t, "emisor"] else f"{t}D")
        fi_on = FICH.get(on_sel) or X.FICHAS.get(on_sel) or [None] * 9
        liq_on = B.liquidacion_24hs(date.today() if en_vivo else D["fecha"])
        v_def = date.fromisoformat(fi_on[1]) if fi_on[1] and fi_on[1] > liq_on.isoformat() else date(2030, 6, 30)
        cup = k2.number_input("Cupón anual %", 0.0, 30.0, float(fi_on[4]) if fi_on[4] is not None else 8.0, 0.125, key=f"on_cup_{on_sel}",
                              help="De la ficha de BYMA." if fi_on[4] is not None else "La ficha no informa una tasa fija: cargala a mano.") / 100
        frec = k3.selectbox("Paga", [2, 4, 12, 1], index=[2, 4, 12, 1].index(fi_on[6]) if fi_on[6] in (2, 4, 12, 1) else 0, key=f"on_frec_{on_sel}",
                            format_func={1: "anual", 2: "semestral", 4: "trimestral", 12: "mensual"}.get)
        venc_on = k4.date_input("Vencimiento", v_def, format="DD/MM/YYYY", key=f"on_venc_{on_sel}")
        cuotas_on = k5.number_input("Cuotas de capital que faltan", 1, 60, int(fi_on[7]) if fi_on[7] else 1, key=f"on_cuo_{on_sel}",
                                    help="1 = paga todo al vencimiento (bullet). La ficha da las cuotas totales: si ya pagó alguna, bajalo.")
        resid_on = k6.number_input("Residual por 100 VN", 1.0, 100.0, 100.0, 1.0, key=f"on_res_{on_sel}")
        fl_on = X.flujos_on(venc_on, cup, frec, int(cuotas_on), liq_on, resid_on)
        px_on = float(TO.loc[on_sel, "precio_mep"])
        y_on, md_on = X.tir_on(px_on, fl_on, liq_on)
        ic_on = X.interes_corrido_on(venc_on, cup, frec, liq_on, resid_on)
        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("Precio MEP", f"{px_on:.2f}")
        j2.metric("TIR", f"{y_on:.2%}" if np.isfinite(y_on) else "–")
        j3.metric("Duración modificada", f"{md_on:.2f}" if np.isfinite(md_on) else "–")
        j4.metric("Valor técnico", f"{resid_on + ic_on:.2f}", f"residual {resid_on:.0f} + corrido {ic_on:.2f}", delta_color="off",
                  help="Valor residual más intereses corridos (base 30/360).")
        j5.metric("Paridad técnica", f"{px_on / (resid_on + ic_on):.1%}", help="Precio sucio / valor técnico, como en la cátedra.")
        if fi_on[0]:
            st.caption(f"Ficha de BYMA: {fi_on[0]} · {fi_on[8]} · emitida el {pd.Timestamp(fi_on[2]):%d/%m/%Y} · moneda {fi_on[3]}"
                       + (" · tasa variable" if fi_on[5] == "v" else "") if fi_on[2] else f"Ficha de BYMA: {fi_on[0]} · {fi_on[8]}")
        if fl_on:
            st.dataframe(pd.DataFrame(fl_on, columns=["Fecha", "Cupón", "Amortización"]).assign(Total=lambda t: t["Cupón"] + t["Amortización"])
                         .style.format({"Fecha": "{:%d/%m/%Y}", "Cupón": "{:.3f}", "Amortización": "{:.2f}", "Total": "{:.3f}"}), hide_index=True, height=220)
        st.caption(f"Liquidación {liq_on:%d/%m/%Y} (24 hs). Se toma el precio en dólar MEP como precio sucio (con intereses corridos), igual que los "
                   "soberanos. Las fechas de pago se cuentan hacia atrás desde el vencimiento.")

# ── Tesoro en pesos: una sección por tipo de bono ──
with tabP:
    q1, q2, q3, q4, q5, q6 = st.columns([.9, 1, 1.2, 1.3, 1.3, .8])
    plazo_p = q1.radio("Plazo", ["CI", "24HS"], index=1, horizontal=True, format_func=lambda x: "CI" if x == "CI" else "24 hs")
    precio_p = q2.selectbox("Precio", ["px", "last", "prev"], format_func=lambda x: {"px": "VWAP", "last": "Último", "prev": "Cierre anterior"}[x])
    tamar_hoy = float(DP["tamar"].sort_index().iloc[-5:].mean())
    rem_med = D["rem_mensual"]["mediana"].dropna()
    tamar_sup = q3.number_input("TAMAR futura supuesta (TNA %)", min_value=0.0, value=round(tamar_hoy, 2), step=0.25,
                                help="Arranca en el promedio de las últimas 5 ruedas. Se usa en TAMAR y Duales.")
    infl_sup = q4.number_input("Inflación mensual después del REM (%)", min_value=0.0, value=round(float(rem_med.iloc[-1]), 2), step=0.05,
                               help="Sendero del CER: IPC publicado, después medianas del REM y después este valor.") / 100
    dev_in = q5.number_input("Devaluación mensual esperada (%)", value=None, step=0.05, placeholder="implícita en la curva",
                             help="Para dólar linked y TMVE8. Vacío = la devaluación implícita en la curva dólar linked.")
    dev_sup = None if dev_in is None else dev_in / 100
    IPC_PUB, REM_MED = D["macro"]["ipc"].dropna().to_dict(), rem_med.to_dict()
    RP = calcular_pesos(en_vivo, plazo_p, precio_p, tamar_sup, infl_sup, dev_sup)
    n_inst = sum(len(RP[k]) for k, _, _ in P.TIPOS)
    q6.metric("Instrumentos", n_inst)
    st.caption(f"BYMA Open Data · liquidación {RP['liq']:%d/%m/%Y} · A3500 {RP['tc_fecha']:%d/%m/%Y} · CER publicado hasta {RP['cer_ultimo']:%d/%m/%Y} · fuentes: "
               + ", ".join(f"{k} {v}" for k, v in DP["fuente"].items()))
    cf, cc = RP["curva_fija"], RP["curva_cer"]
    mens = lambda y: (1 + y) ** (30 / 365) - 1
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tasa fija 6 meses · TEM", f"{mens(cf(0.5)):.2%}", f"TIREA {cf(0.5):.1%}", delta_color="off")
    m2.metric("Tasa fija 1 año · TEM", f"{mens(cf(1)):.2%}", f"TIREA {cf(1):.1%}", delta_color="off")
    m3.metric("CER 1 año", f"{cc(1):.2%}" if cc else "–", "TIR real", delta_color="off")
    m4.metric("TAMAR", f"{RP['tamar_hoy']:.2f}%", "TNA, 5 ruedas", delta_color="off")
    m5.metric("Dólar A3500", f"${RP['tc_hoy']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), f"{RP['tc_fecha']:%d/%m}", delta_color="off")
    dl_ = RP["dl"]
    dev6 = np.interp(180, dl_.loc[dl_["dias"] >= 15, "dias"], dl_.loc[dl_["dias"] >= 15, "deval_mensual"]) if len(dl_) else np.nan
    m6.metric("Devaluación implícita", f"{dev6:.2%}", "mensual a 6 meses", delta_color="off")

    alias = [f"{t}→{r['precio_de']}" for k, _, _ in P.TIPOS if len(RP[k]) for t, r in RP[k].iterrows() if r["precio_de"] != t]
    if alias:
        st.caption("El ticker base no operó; precio tomado del alias con el mismo ISIN: " + ", ".join(alias))

    COM = {"vence": "{:%d/%m/%Y}", "dias": "{:.0f}", "precio": "{:,.2f}", "var": "{:+.2%}", "vol": "{:,.0f}", "md": "{:.2f}", "spread_pb": "{:+.0f}"}
    NOM = {"vence": "Vence", "dias": "Días", "precio": "Precio", "var": "Var.", "vol": "Vol. $M", "md": "MD", "spread_pb": "Spread (pb)"}
    NOTAS = {
        "lecap": "Pago final = 100 × (1 + TEM de emisión)^(meses 30/360 desde la emisión). TIREA = (pago / precio)^(365/días) − 1. La curva de tasa fija es un Nelson-Siegel sobre LECAP y BONCAP con más de 15 días; el spread compara cada bono con esa curva.",
        "boncap": "BONCAP: cupón cero capitalizable, igual que las LECAP. BONTE (TO26, TY30P): tasa fija con cupones semestrales, valuados por flujos.",
        "cer": "Valor técnico = 100 × CER de 10 hábiles antes de la liquidación / CER base (10 hábiles antes de la emisión). Inflación BE: se toma tal cual el CER que el BCRA ya publicó y se busca la inflación mensual pareja de los meses todavía no publicados que hace que el bono pague lo mismo que la curva de tasa fija (ajustada a LECAP y BONCAP, mejor ajuste) a su plazo (n/a: el CER hasta el vencimiento ya está publicado; —: vence después de la letra a tasa fija más larga). REM: promedio mensual del sendero del REM para esos mismos meses. TNA 30/365 = ((1 + TIR)^(30/365) − 1) × 365/30; la nominal usa el sendero IPC publicado → REM → supuesto. DICP, DIP0, PARP, PAP0 y CUAP son los bonos del canje 2005 (base CER 31/12/2003).",
        "tamar": "Margen s/ TAMAR: TNA equivalente al rendimiento del bono menos la TAMAR supuesta. Pagan al vencimiento 100 × (1 + TAMAR TEM)^meses, con TAMAR TEM = [(1 + TAMAR × 32/365)^(365/32)]^(1/12) − 1 y la TAMAR promediada en toda la vida del título (desde 10 hábiles antes de la emisión hasta 10 antes del vencimiento), más el margen de la licitación: TMF27 6,50%, TML27 5,40%, TMG27 6,00%, TMF28 6,50% y TMG28 6,90% (comunicados de la Secretaría de Finanzas; TMG27, ficha técnica).",
        "dual": "TTD26: máximo entre TEM fija 2,14% y TAMAR. TXM*: máximo entre capital ajustado por CER (proyectado con IPC, REM y la inflación elegida) y TAMAR + 3%. TMVE8: máximo entre USD 100 al A3500 (con la devaluación elegida o la implícita) y el tipo de cambio inicial ($1.499,84) capitalizado a TAMAR. El piso es lo que asegura la otra pata.",
        "dl": "Pagan USD 100 por lámina en pesos, al A3500 de 3 hábiles antes del vencimiento. Dólar breakeven = precio / 100 × (1 + TIREA de la curva de tasa fija)^(días/365): si el A3500 termina por encima conviene el dólar linked; por debajo, la tasa fija.",
    }
    secciones = st.tabs([f"{nombre} ({len(RP[k])})" for k, nombre, _ in P.TIPOS])
    for (k, nombre, desc), sec in zip(P.TIPOS, secciones):
        with sec:
            df = RP[k]
            st.caption(desc)
            if df.empty:
                st.info("Sin datos para este tipo de bono."); continue
            if k in ("lecap", "boncap"):
                barato = df[df["dias"] >= 30].sort_values("spread_pb").tail(1)
                st.markdown(f"Mayor TEM: **{df['tem'].idxmax()}** {df['tem'].max():.2%}" + (f" · Más barato vs la curva: **{barato.index[0]}** {barato['spread_pb'].iloc[0]:+.0f} pb" if len(barato) else ""))
                t = df.copy()
                t["pago_t"] = ["cupón" if pd.isna(v) else f"{v:,.3f}" for v in t["pago_final"]]
                mostrar(t[["vence", "dias", "precio", "var", "pago_t", "tna", "tem", "tirea", "md", "spread_pb", "vol"]],
                        {**COM, "tna": "{:.2%}", "tem": "{:.2%}", "tirea": "{:.2%}"},
                        {**NOM, "pago_t": "Pago final", "tna": "TNA", "tem": "TEM", "tirea": "TIREA"}, calor={"var": "var"})
                ver_grafico(grafico_pesos(df, "tirea", cf, "#2a78d6" if k == "lecap" else "#1baf7a", "TIREA", "Duration (años)"))
            elif k == "cer":
                c1, c2 = st.columns([1.3, 1])
                vista_cer = c1.radio("Gráfico", ["TIR real por plazo", "Inflación breakeven mensual vs REM"], horizontal=True)
                plazo_cer = c2.selectbox("Plazo máximo", [0, 365, 730, 1095], format_func=lambda v: "Todos" if not v else f"{v} días", key="plazo_cer")
                d = df[df["dias"] <= plazo_cer] if plazo_cer else df
                be = d[np.isfinite(d["be_mensual"].astype(float))]
                l = d[d["dias"] >= 60]
                st.markdown(f"TIR real: **{l['tir_real'].min():.2%}** a **{l['tir_real'].max():.2%}**" +
                            (f" · Inflación BE mensual: **{be['be_mensual'].min():.2%}** a **{be['be_mensual'].max():.2%}**, mediana BE − REM **{be['be_rem_pb'].median():+.0f} pb**" if len(be) else ""))
                t = d.copy()
                t["be_t"] = [f"{v:.2%}" if np.isfinite(v) else ("n/a: paga fijo" if n == "CER ya publicado" else "—") for v, n in zip(t["be_mensual"], t["be_nota"])]
                mostrar(t[["vence", "dias", "precio", "var", "valor_tecnico", "paridad", "tir_real", "tna_real", "tna_nominal", "md", "spread_pb", "be_t", "rem_mensual", "be_rem_pb", "vol"]],
                        {**COM, "valor_tecnico": "{:,.2f}", "paridad": "{:.1%}", "tir_real": "{:.2%}", "tna_real": "{:.2%}", "tna_nominal": "{:.2%}",
                         "rem_mensual": "{:.2%}", "be_rem_pb": "{:+.0f}"},
                        {**NOM, "valor_tecnico": "VT", "paridad": "Paridad", "tir_real": "TIR real (TEA)", "tna_real": "TNA real 30/365",
                         "tna_nominal": "TNA nominal 30/365", "be_t": "Inflación BE mensual", "rem_mensual": "REM mensual", "be_rem_pb": "BE − REM (pb)"}, calor={"var": "var"})
                if vista_cer.startswith("TIR"):
                    ver_grafico(grafico_cer_tir(d))
                else:
                    ver_grafico(grafico_be(d, RP["max_cero"]))
                    fuera = d[~np.isfinite(d["be_mensual"].astype(float))]
                    if len(fuera):
                        largos = fuera[fuera["be_nota"].str.startswith("vence")].index.tolist(); pub = fuera[fuera["be_nota"] == "CER ya publicado"].index.tolist()
                        st.caption("Sin graficar: " + " · ".join(x for x in [f"{', '.join(largos)} (vencen después de la LECAP/BONCAP más larga, {RP['max_cero']} días)" if largos else "",
                                                                         f"{', '.join(pub)} (el CER hasta el vencimiento ya está publicado)" if pub else ""] if x))
            elif k == "tamar":
                st.markdown(f"TAMAR hoy **{RP['tamar_hoy']:.2f}%** TNA · supuesta a futuro **{tamar_sup:.2f}%** · si la TAMAR promedio futura supera la TAMAR BE, el bono rinde más que la tasa fija")
                t = df.copy()
                t["margen_t"] = [("s/d" if u else f"{m:g}%") for u, m in zip(t["margen_desconocido"], t["margen"])]
                mostrar(t[["vence", "dias", "precio", "var", "margen_t", "tamar_prom", "valor_tecnico", "paridad", "tirea", "tem", "tamar_be", "margen_s_tamar", "vol"]],
                        {**COM, "tamar_prom": "{:.2f}%", "valor_tecnico": "{:,.2f}", "paridad": "{:.1%}", "tirea": "{:.2%}", "tem": "{:.2%}", "tamar_be": "{:.2f}%", "margen_s_tamar": "{:.2f}%"},
                        {**NOM, "margen_t": "Margen licitación", "tamar_prom": "TAMAR prom.", "valor_tecnico": "VT est.", "paridad": "Paridad", "tirea": "TIREA est.",
                         "tem": "TEM est.", "tamar_be": "TAMAR BE", "margen_s_tamar": "Margen s/ TAMAR"}, calor={"var": "var"})
                ver_grafico(grafico_margen_tamar(df, RP["dual"], tamar_sup))
            elif k == "dual":
                st.markdown(f"Con TAMAR en {tamar_sup:.2f}%, pagan la pata TAMAR **{(df['paga'] == 'TAMAR').sum()} de {len(df)}** · el piso es lo que asegura la otra pata aunque la TAMAR caiga")
                t = df.copy()
                t["piso_t"] = [f"{p:.2%} {tp}" for p, tp in zip(t["piso"], t["piso_tipo"])]
                t["cruce_t"] = ["superada" if c < 0.01 else f"{c:.2f}%" for c in t["tamar_cruce"]]
                mostrar(t[["patas", "vence", "dias", "precio", "var", "pago_pata1", "pago_tamar", "paga", "tirea", "piso_t", "cruce_t", "margen_s_tamar", "vol"]],
                        {**COM, "pago_pata1": "{:,.2f}", "pago_tamar": "{:,.2f}", "tirea": "{:.2%}", "margen_s_tamar": "{:.2f}%"},
                        {**NOM, "patas": "Patas", "pago_pata1": "Pago pata 1", "pago_tamar": "Pago TAMAR", "paga": "Paga", "tirea": "TIREA est.", "piso_t": "Piso",
                         "cruce_t": "TAMAR de cruce", "margen_s_tamar": "Margen s/ TAMAR"}, calor={"var": "var"})
                ver_grafico(grafico_pesos(df, "tirea", cf, "#2a78d6", "TIREA estimada"))
            else:
                ult = df.iloc[-1]
                st.markdown(f"A3500 hoy **\\${RP['tc_hoy']:,.2f}** · Dólar breakeven {ult['vence']:%m/%Y}: **\\${ult['dolar_be']:,.0f}** ({ult['deval_total']:+.1%})")
                cols = ["vence", "dias", "precio", "precio_usd", "var", "tir_usd", "dolar_be", "deval_mensual", "tirea_esperada", "vs_fija_pb", "md", "vol"]
                mostrar(df[cols],
                        {**COM, "precio": "{:,.0f}", "precio_usd": "{:.2f}", "tir_usd": "{:.2%}", "dolar_be": "${:,.2f}", "deval_mensual": "{:.2%}",
                         "tirea_esperada": "{:.2%}", "vs_fija_pb": "{:+.0f}"},
                        {**NOM, "precio": "Precio $", "precio_usd": "Precio USD", "tir_usd": "TIR USD", "dolar_be": "Dólar BE", "deval_mensual": "Deval. mensual",
                         "tirea_esperada": "TIREA $ esperada", "vs_fija_pb": "vs tasa fija (pb)"},
                        na="= curva", calor={"var": "var"})
                ver_grafico(grafico_pesos(df[df["dias"] >= 15], "deval_mensual", None, "#1baf7a", "Devaluación mensual implícita"))
            st.caption(NOTAS[k])

# ── Mi portafolio: cartera propia con riesgo de tasa, VaR y tipo de cartera ──
GLOSARIO_PF = {
    "Valor de mercado": "Lo que vale hoy la cartera: nominales × precio / 100 de cada bono. Los bonos en dólares se pasan a pesos al dólar MEP (o cable, si lo elegiste).",
    "Duración modificada": "Cuánto cae el valor de la cartera, en %, si todas las tasas suben 1 punto porcentual. Una duración de 3 quiere decir: tasas +1 pp → cartera −3%. "
                           "Es el promedio de la duración de cada bono, ponderado por su peso.",
    "DV01": "La misma idea en plata: cuántos pesos (o dólares) perdés si todas las tasas suben 1 punto básico (0,01%).",
    "Convexidad": "La curvatura de la relación precio-tasa. Con más convexidad, cuando las tasas bajan ganás un poco más y cuando suben perdés un poco menos "
                  "de lo que dice la duración. Importa en movimientos grandes.",
    "Plazo promedio": "Cuántos días faltan, en promedio ponderado por valor, hasta el vencimiento de los bonos.",
    "Mayor exposición": "El grupo de bonos con más peso y cuántos bonos «efectivos» tenés: 1 / suma de los pesos al cuadrado. Con 10 bonos de igual peso son 10; "
                        "si uno pesa 90%, aunque tengas 10, es como tener poco más de 1.",
    "Key rate durations (KRD)": "Reparten la duración entre plazos de la curva (3 meses, 6 meses, 1, 2, 3, 5, 7, 10, 15 y 20 años). Dicen cuánto perdés si sube "
                                "solo la tasa de ese plazo: la curva casi nunca se mueve pareja.",
    "Escenarios de tasas": "Se elige cuánto sube o baja cada plazo de la curva y se revalúa cada bono descontando sus flujos a su TIR más el shock del plazo de "
                           "cada pago. La aproximación lineal (KRD × shock) no tiene en cuenta la convexidad: la diferencia entre las dos es la convexidad.",
    "VaR histórico": "Value at Risk: la pérdida que solo se habría superado en el (1 − confianza) de los días del período, si hubieras tenido esta misma "
                     "cartera en cada rueda. VaR 99% a 1 día de $100.000: en 99 de cada 100 días la pérdida fue menor.",
    "Expected shortfall": "La pérdida promedio en ese 1% (o 5%) de peores días: mira qué tan mala es la cola, no solo dónde empieza.",
    "Volatilidad anual": "Desvío estándar de los resultados diarios de la cartera, llevado a un año (× √252).",
    "Bullet, barbell y ladder": "Bullet: vencimientos concentrados alrededor de un plazo. Barbell: cargada en las dos puntas (corto y largo) con poco en el "
                                "medio. Ladder: repartida en escalones parejos a lo largo de la curva, que se van reinvirtiendo de a poco.",
}

H_ = {}   # historia de precios de tu cartera: la completa esta pestaña y la usa el informe semanal
with tabPF:
    st.caption("Cargá tus bonos y nominales (valor nominal, no pesos) y mirá cuánto vale la cartera, cuánto riesgo de tasa tiene, qué pasa en distintos "
               "escenarios, cuánto podría perder en un día malo (VaR) y qué tipo de cartera es. Cada número tiene su explicación al pasar el mouse por el ? "
               "y en el glosario de abajo.")
    tc_pf = dolar_bono("AL30C" if moneda == "ccl" else "AL30D")
    UPF = PF.agregar_ons(PF.universo(tabla_usd, liq_usd, RP0, tc_pf, moneda), ON_OK, liq_usd, tc_pf)
    orden_g = {g: i for i, g in enumerate(PF.GRUPOS)}
    opciones_pf = sorted(UPF, key=lambda t: (orden_g[UPF[t]["grupo"]], UPF[t]["vence"], t))
    e1, e2 = st.columns([1, 2.2])
    with e1:
        moneda_pf = st.radio("Valuar en", ["ARS", "USD"], horizontal=True, key="pf_moneda",
                             format_func={"ARS": "Pesos", "USD": "Dólares " + ("cable" if moneda == "ccl" else "MEP")}.get)
        st.caption(f"Dólar {'cable' if moneda == 'ccl' else 'MEP'} $ {num_ar(tc_pf, 2)} (AL30). Precios de hoy: hard dollar y ONs en dólar {'cable' if moneda == 'ccl' else 'MEP'}, "
                   f"bonos en pesos con los supuestos por defecto de «Tesoro en pesos» (24 hs). Hay {len(ON_OK)} ONs disponibles: en dólares, tasa fija, "
                   "condiciones de la ficha de BYMA (si no informa la frecuencia, pagos semestrales).")
        b_ej, b_vac = st.columns(2)
        if b_ej.button("Cargar ejemplo", key="pf_ej"):
            st.session_state["pf_base"] = pd.DataFrame({"Bono": list(PF.EJEMPLO), "Nominales": list(PF.EJEMPLO.values())})
            st.session_state.pop("pf_editor", None); st.rerun()
        if b_vac.button("Vaciar", key="pf_vac"):
            st.session_state["pf_base"] = pd.DataFrame({"Bono": pd.Series(dtype=str), "Nominales": pd.Series(dtype=float)})
            st.session_state.pop("pf_editor", None); st.rerun()
    st.session_state.setdefault("pf_base", pd.DataFrame({"Bono": list(PF.EJEMPLO), "Nominales": list(PF.EJEMPLO.values())}))
    with e2:
        ed = st.data_editor(st.session_state["pf_base"], num_rows="dynamic", key="pf_editor", width="stretch", height=260,
                            column_config={"Bono": st.column_config.SelectboxColumn("Bono", options=opciones_pf, required=True,
                                                                                    help="Hard dollar, obligaciones negociables (ONs en dólares a tasa fija), LECAP, "
                                                                                         "BONCAP, CER, TAMAR, duales y dólar linked. Escribí el ticker para buscar."),
                                           "Nominales": st.column_config.NumberColumn("Nominales (VN)", min_value=0, step=1000, format="%d")})
    g1_, g2_ = st.columns([1, 2.2])
    with g1_:
        st.download_button("Guardar mi portafolio (CSV)", ed.dropna().to_csv(index=False).encode("utf-8"), file_name="mi_portafolio_bondai.csv",
                           mime="text/csv", on_click="ignore", help="Baja un archivo con tus bonos y nominales para volver a subirlo otro día.")
        if st.session_state.get("_pf_aviso"):
            st.caption(st.session_state["_pf_aviso"])
    sub_pf = g2_.file_uploader("Subir un portafolio guardado (CSV con las columnas Bono y Nominales)", type=["csv"], key="pf_subir")
    if sub_pf is not None and st.session_state.get("_pf_subido") != sub_pf.file_id:
        try:
            nuevo_ = pd.read_csv(sub_pf, sep=None, engine="python")
            nuevo_.columns = [str(c).strip().capitalize() for c in nuevo_.columns]
            nuevo_ = nuevo_[["Bono", "Nominales"]].dropna()
            nuevo_["Bono"] = nuevo_["Bono"].astype(str).str.strip().str.upper()
            nuevo_["Nominales"] = pd.to_numeric(nuevo_["Nominales"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                                errors="coerce") if not pd.api.types.is_numeric_dtype(nuevo_["Nominales"]) else nuevo_["Nominales"]
            fuera_ = sorted(set(nuevo_["Bono"]) - set(UPF))
            st.session_state["pf_base"] = nuevo_[nuevo_["Bono"].isin(UPF)].reset_index(drop=True)
            st.session_state["_pf_subido"] = sub_pf.file_id
            st.session_state["_pf_aviso"] = ("Cargado. No están hoy en la lista (sin precio o vencidos): " + ", ".join(fuera_)) if fuera_ else "Portafolio cargado."
            st.session_state.pop("pf_editor", None)
            st.rerun()
        except Exception as e_:
            st.error(f"No pude leer el archivo ({type(e_).__name__}). Tiene que ser un CSV con las columnas Bono y Nominales.")
    pos = ed.dropna().groupby("Bono")["Nominales"].sum().to_dict() if len(ed) else {}
    CPF = PF.cartera(UPF, pos, moneda_pf, tc_pf, hoy=liq_usd)
    if CPF.empty:
        st.info("Agregá al menos un bono con nominales para ver los indicadores.")
    else:
        RS = PF.resumen(CPF)
        mon = "US$" if moneda_pf == "USD" else "$"
        plata = lambda v, d=0: f"{mon} {num_ar(v, d)}"
        CPF["grupo_n"] = CPF["grupo"].map(PF.GRUPOS)
        CPF["tasa_tipo"] = CPF["grupo"].map(PF.TASA_DE)
        mostrar(CPF[["grupo_n", "sub", "nominales", "precio", "valor", "peso", "tir", "tasa_tipo", "md", "vence", "dias"]],
                {"nominales": "{:,.0f}", "precio": "{:,.2f}", "valor": "{:,.0f}", "peso": "{:.1%}", "tir": "{:.2%}", "md": "{:.2f}", "vence": "{:%d/%m/%Y}", "dias": "{:.0f}"},
                {"grupo_n": "Grupo", "sub": "Tipo", "nominales": "Nominales", "precio": "Precio", "valor": f"Valor {mon}", "peso": "Peso", "tir": "Tasa",
                 "tasa_tipo": "Qué tasa es", "md": "MD", "vence": "Vence", "dias": "Días"})
        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)
        k1.metric("Valor de mercado", plata(RS["valor"]), f"{RS['n']} bonos", delta_color="off", help=GLOSARIO_PF["Valor de mercado"])
        k2.metric("Duración modificada", f"{RS['md']:.2f}", f"tasas +1 pp → {-RS['md'] / 100:.2%}", delta_color="off", help=GLOSARIO_PF["Duración modificada"])
        k3.metric("DV01", plata(RS["dv01"]), "pierde esto si todas las tasas suben 1 pb", delta_color="off", help=GLOSARIO_PF["DV01"])
        k4.metric("Convexidad", f"{RS['convexidad']:.2f}", "ponderada por valor", delta_color="off", help=GLOSARIO_PF["Convexidad"])
        k5.metric("Plazo promedio", f"{num_ar(RS['plazo_dias'])} días", f"{RS['plazo_dias'] / 365:.1f} años al vencimiento", delta_color="off", help=GLOSARIO_PF["Plazo promedio"])
        k6.metric("Mayor exposición", PF.GRUPOS[RS["mayor_grupo"]], f"{RS['mayor_peso']:.0%} · {RS['efectivos']:.1f} bonos efectivos", delta_color="off",
                  help=GLOSARIO_PF["Mayor exposición"])
        st.caption("Tasa de cada grupo, ponderada por valor: " + " · ".join(f"{PF.GRUPOS[g]} {v:.2%} ({PF.TASA_DE[g]})" for g, v in RS["tir_por_grupo"].items()) +
                   ". Las tasas no se suman entre grupos: cada una está en su propia moneda (dólares, pesos o pesos ajustados por inflación).")

        st.subheader("Qué tipo de cartera es")
        TC = PF.tipo_cartera(CPF)
        t1_, t2_, t3_, t4_ = st.columns([1, 1, 1, 1.4])
        t1_.metric("Tipo de cartera", TC["tipo"], delta_color="off", help=GLOSARIO_PF["Bullet, barbell y ladder"])
        t2_.metric("Plazo promedio", f"{TC['plazo_prom']:.2f} años", "al vencimiento, ponderado por valor", delta_color="off")
        t3_.metric("Dispersión", f"{TC['dispersion']:.2f} años", "desvío de los vencimientos alrededor del promedio", delta_color="off")
        t4_.metric("Corto · medio · largo", f"{TC['corto']:.0%} · {TC['medio']:.0%} · {TC['largo']:.0%}", "antes de ½ del plazo promedio, entre ½ y 1½, después de 1½", delta_color="off")
        fig, ax = plt.subplots(figsize=(11, 3.4))
        base = np.zeros(len(TC["tramos"]))
        for g in [g for g in PF.GRUPOS if g in TC["tramos"].columns]:
            ax.bar(TC["tramos"].index, TC["tramos"][g] * 100, bottom=base, color=PF.COLORES_G[g], label=PF.GRUPOS[g]); base += TC["tramos"][g].values * 100
        ax.set_ylabel("% del valor"); ax.set_xlabel("Vencimiento"); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(.5, 1.18), frameon=False)
        fig.tight_layout(); ver_grafico(fig)
        st.caption(f"**{TC['tipo']}**: {TC['explicacion']} Se clasifica por el vencimiento de cada bono, pesado por su valor. Los bonos que amortizan en "
                   "cuotas (AL30, GD35…) cobran antes del vencimiento: su riesgo real por plazo está en las key rate durations.")

        st.subheader("Cómo está distribuida")
        dd1, dd2, dd3 = st.columns(3)
        for col, campo, titulo, mapa in [(dd1, "grupo", "Por grupo", PF.GRUPOS), (dd2, "moneda", "Por moneda de pago", {"USD": "Dólares", "ARS": "Pesos"}),
                                         (dd3, "ley", "Por legislación", {"Local": "Ley argentina", "Nueva York": "Ley Nueva York"})]:
            z = CPF.groupby(campo)["peso"].sum().sort_values()
            fig, ax = plt.subplots(figsize=(4.2, 2.4))
            cols_ = [PF.COLORES_G.get(k, AZUL) for k in z.index] if campo == "grupo" else [AZUL, VIOLETA, VERDE][:len(z)]
            ax.barh([mapa.get(k, k) for k in z.index], z.values * 100, color=cols_)
            for i, v in enumerate(z.values):
                ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=8)
            ax.set_xlim(0, 115); ax.set_title(titulo, fontsize=10); ax.tick_params(labelsize=8); ax.set_xticks([])
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout(); ver_grafico(fig, col)

        st.subheader("Duración y key rate durations")
        KR = PF.krd_cartera(CPF)
        gk = [g for g in PF.GRUPOS if g in KR.columns and KR[g].abs().sum() > 0]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        base = np.zeros(len(KR))
        for g in gk:
            ax.bar(KR.index, KR[g], bottom=base, color=PF.COLORES_G[g], label=PF.GRUPOS[g]); base += KR[g].values
        ax.set_ylabel("Key rate duration (años)"); ax.set_xlabel("Plazo (nodo de la curva)"); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8)
        fig.tight_layout(); ver_grafico(fig)
        KRt = KR[gk + ["total", "dv01"]].copy(); KRt.loc["Total"] = KRt.sum()
        mostrar(KRt, {**{g: "{:.3f}" for g in gk}, "total": "{:.3f}", "dv01": f"{mon} {{:,.0f}}"}, {**{g: PF.GRUPOS[g] for g in gk}, "total": "Total", "dv01": "DV01"})
        st.caption(GLOSARIO_PF["Key rate durations (KRD)"] + " La suma de las barras es la duración de la cartera. TAMAR y duales llevan duración efectiva "
                   "casi nula: la TAMAR futura acompaña a la tasa, así que su riesgo es el margen.")

        st.subheader("Escenarios de tasas")
        es1, es2, es3 = st.columns([1.6, .8, 1.2])
        esc = es1.selectbox("Escenario", list(PF.ESCENARIOS), format_func=PF.ESCENARIOS.get, key="pf_esc")
        mag = es2.number_input("Magnitud (pb)", 0, 2000, 100, 25, key="pf_mag", disabled=esc == "propio")
        gr_pres = [g for g in ["hd", "fija", "cer", "dl"] if g in CPF["grupo"].values]
        aplicar = es3.selectbox("Aplicar a", ["todas"] + gr_pres, format_func=lambda g: "Todas las curvas" if g == "todas" else f"Solo {PF.GRUPOS[g]}", key="pf_aplicar")
        forma = PF.forma_escenario(esc if esc != "propio" else "suba", mag if esc != "propio" else 100)
        base_sh = pd.DataFrame({PF.GRUPOS[g]: (forma if aplicar in ("todas", g) else np.zeros(len(PF.NODOS))) for g in gr_pres}, index=PF.ETIQ_NODOS)
        shd = st.data_editor(base_sh, key=f"pf_sh_{esc}_{mag}_{aplicar}_{'-'.join(gr_pres)}", width="stretch", disabled=esc != "propio",
                             column_config={c: st.column_config.NumberColumn(c + " (pb)", step=5, format="%d") for c in base_sh.columns})
        shocks = {g: shd[PF.GRUPOS[g]].fillna(0).values.astype(float) for g in gr_pres}
        EPF = PF.escenario(UPF, CPF, shocks)
        imp, lin = EPF["impacto"].sum(), EPF["impacto_lineal"].sum()
        porg = EPF.groupby("grupo")["impacto"].sum()
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Impacto en la cartera", plata(imp), f"{imp / RS['valor']:+.2%} del valor", delta_color="normal",
                  help="Revaluación completa: cada flujo descontado a la TIR del bono más el shock de su plazo.")
        x2.metric("Con duración (lineal)", plata(lin), "suma de KRD × shock", delta_color="off", help="La diferencia con la revaluación completa es la convexidad.")
        x3.metric("Efecto convexidad", plata(imp - lin), "revaluación − lineal", delta_color="off")
        if len(porg):
            peor = porg.idxmin() if porg.min() < 0 else porg.idxmax()
            x4.metric("Grupo más afectado", PF.GRUPOS[peor], plata(porg[peor]), delta_color="off")
        ap = PF.aporte_por_nodo(CPF, shocks)
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.bar(ap.index, ap.values, color=[VERDE if v >= 0 else ROJO for v in ap.values])
        ax.axhline(0, color="black", lw=.8); ax.set_ylabel(mon); ax.set_xlabel("Qué plazo explica el resultado (KRD × shock)"); ax.grid(alpha=.25, axis="y")
        fig.tight_layout(); ver_grafico(fig)
        mostrar(EPF.assign(grupo=EPF["grupo"].map(PF.GRUPOS))[["grupo", "valor", "cambio", "impacto", "pct_impacto"]].sort_values("impacto"),
                {"valor": "{:,.0f}", "cambio": "{:+.2%}", "impacto": "{:+,.0f}", "pct_impacto": "{:.1%}"},
                {"grupo": "Grupo", "valor": f"Valor hoy {mon}", "cambio": "Cambio de precio", "impacto": f"Impacto {mon}", "pct_impacto": "% del impacto"},
                calor={"cambio": "var"})
        st.caption("Cada grupo se mueve con su propia curva: hard dollar en TIR en dólares, ONs en su TIR en dólares (riesgo soberano más el de la "
                   "empresa), tasa fija en TIREA nominal, CER en tasa real y dólar linked en TIR en dólares oficiales. Editá cualquier celda eligiendo "
                   "«Propio» para armar tu escenario.")

        st.markdown("**Movimiento paralelo de todas las tasas**")
        PA = PF.paralelos(UPF, CPF)
        PA.index = [f"{v:+,} pb".replace(",", ".") for v in PA.index]
        mostrar(PA, {"aprox": "{:+,.0f}", "completa": "{:+,.0f}", "pct": "{:+.2%}"},
                {"aprox": "Duración + convexidad", "completa": "Revaluación completa", "pct": "En % de la cartera"}, calor={"pct": "var"})
        st.caption("Todas las tasas se mueven igual. La primera columna aproxima con la serie de Taylor (−MD × Δy + ½ × C × Δy²); la segunda revalúa cada bono con "
                   "sus flujos. La diferencia son los términos de orden 3 en adelante: en movimientos grandes la aproximación falla.")

        st.subheader("Value at Risk")
        v1, v2, v3 = st.columns(3)
        conf = v1.selectbox("Confianza", [0.99, 0.95], format_func=lambda c: f"{c:.0%}", key="pf_conf")
        hz = v2.selectbox("Horizonte", [1, 5, 10], format_func=lambda h: f"{h} día" + ("s" if h > 1 else ""), key="pf_hz")
        vent = v3.selectbox("Ventana", [250, 500, 120], format_func=lambda w: f"{w} ruedas", key="pf_vent")
        sims = tuple(sorted({UPF[t]["simbolo"] for t in CPF.index} | {"AL30", "AL30D" if moneda != "ccl" else "AL30C"}))
        hist_vivo = en_vivo and st.toggle("Usar el histórico actualizado de BYMA", value=False, key="pf_hist_vivo",
                                          help="Apagado: precios diarios guardados hasta el 09/10/2026 (carga al instante). Prendido: baja de BYMA el "
                                               "histórico de cada bono de la cartera, de a uno (tarda unos segundos por bono).")
        H_, FH_ = historicos_pf(hist_vivo, sims)
        RPF, rell = PF.retornos(UPF, CPF, H_, moneda_pf, ("AL30", "AL30C" if moneda == "ccl" else "AL30D"))
        if RPF.empty or len(RPF) < 30:
            st.info("No hay suficiente historia de precios para estimar el VaR de esta cartera.")
        else:
            VR = PF.var_historico(RPF, CPF["valor"], conf, hz, vent)
            y1, y2, y3 = st.columns(3)
            y1.metric("VaR histórico", plata(VR["var"]), f"{VR['var_pct']:.2%} del valor · {conf:.0%} a {hz} día{'s' if hz > 1 else ''}", delta_color="off", help=GLOSARIO_PF["VaR histórico"])
            y2.metric("Expected shortfall", plata(VR["es"]), f"pérdida promedio en el {1 - conf:.0%} de los peores casos", delta_color="off", help=GLOSARIO_PF["Expected shortfall"])
            y3.metric("Volatilidad anual", f"{VR['vol_anual']:.1%}", f"con {VR['ruedas']} ruedas", delta_color="off", help=GLOSARIO_PF["Volatilidad anual"])
            cv1, cv2 = st.columns([1.5, 1])
            with cv1:
                fig, ax = plt.subplots(figsize=(7.5, 3.3))
                esc_m = 1e6 if VR["serie"].abs().max() > 2e6 else (1e3 if VR["serie"].abs().max() > 2e3 else 1)
                u_m = {1e6: " millones", 1e3: " miles", 1: ""}[esc_m]
                sv = VR["serie"] / esc_m
                ax.hist(sv, bins=50, color=[AZUL], alpha=.85)
                ax.axvline(-VR["var"] / esc_m, color=ROJO, lw=1.6, label=f"VaR {conf:.0%}")
                ax.axvline(-VR["es"] / esc_m, color=ROJO, lw=1, ls="--", label="Expected shortfall")
                ax.set_xlabel(f"Resultado de la cartera de hoy en {hz} día{'s' if hz > 1 else ''} ({mon}{u_m})"); ax.set_ylabel("Días"); ax.legend(fontsize=8); ax.grid(alpha=.25)
                fig.tight_layout(); ver_grafico(fig)
            with cv2:
                PBv = VR["por_bono"].copy()
                PBv["datos"] = ["propios" if t not in rell or not rell[t][1] else "propios + " + ", ".join(rell[t][1]) for t in PBv.index]
                PBv["fuente"] = [FH_.get(UPF[t]["simbolo"], "") for t in PBv.index]
                mostrar(PBv[["valor", "vol_diaria", "pct_riesgo", "datos", "fuente"]], {"valor": "{:,.0f}", "vol_diaria": "{:.2%}", "pct_riesgo": "{:.1%}"},
                        {"valor": f"Valor {mon}", "vol_diaria": "Vol. diaria", "pct_riesgo": "% del riesgo", "datos": "Historia", "fuente": "Fuente"})
            st.caption(f"Período: {VR['desde']:%d/%m/%Y} a {VR['hasta']:%d/%m/%Y} ({VR['ruedas']} ruedas). Histórico: se toma la cartera de hoy y se le aplican "
                       "los retornos diarios de cada bono en cada rueda del período (precio de BYMA en 24 hs, con cupones y amortizaciones sumados). Los días sin "
                       "precio (bonos nuevos o sin operaciones) usan el retorno del bono del mismo grupo con duración más parecida, escalado por la relación de "
                       "duraciones. En pesos, los bonos en dólares incluyen el movimiento del tipo de cambio; en dólares, al revés. El % del riesgo es la "
                       "contribución de cada bono a la varianza de la cartera (suma 100%).")

        st.subheader("Tu cartera contra un índice de bonos")
        st.caption("Qué hubiera pasado en los últimos meses con la cartera de hoy (mismos pesos de cada bono) contra un índice de igual peso. "
                   "No es lo que ganaste de verdad: es una prueba hacia atrás para ver si tu cartera fue más o menos volátil y si le ganó al mercado.")
        i1_, i2_ = st.columns([1.6, 1])
        idx_sel = i1_.selectbox("Índice", ["sob", "fija", "cer", "mixto"], key="pf_idx",
                                format_func={"sob": "Soberanos en dólares (Bonares y Globales)", "fija": "Tasa fija en pesos (LECAP y BONCAP)",
                                             "cer": "Bonos CER", "mixto": "Mitad soberanos en dólares, mitad tasa fija"}.get)
        per_idx = i2_.selectbox("Período", [60, 120, 250, 500], index=2, key="pf_idx_per", format_func=lambda n: f"últimas {n} ruedas")
        miembros = {"sob": [t for t in UPF if UPF[t]["grupo"] == "hd" and not UPF[t]["sub"].startswith("Tesoro")],
                    "fija": [t for t in UPF if UPF[t]["grupo"] == "fija"], "cer": [t for t in UPF if UPF[t]["grupo"] == "cer"]}
        miembros["mixto"] = miembros["sob"] + miembros["fija"]
        fx_pf = ("AL30", "AL30C" if moneda == "ccl" else "AL30D")
        C_idx = PF.cartera(UPF, {t: 1000.0 for t in miembros[idx_sel]}, moneda_pf, tc_pf, hoy=liq_usd)
        H_i, _ = historicos_pf(hist_vivo, tuple(sorted({UPF[t]["simbolo"] for t in miembros[idx_sel]} | set(fx_pf))))
        R_i, _ = PF.retornos(UPF, C_idx, H_i, moneda_pf, fx_pf) if len(C_idx) else (pd.DataFrame(), {})
        if R_i.empty or RPF.empty:
            st.info("No hay historia suficiente para comparar.")
        else:
            if idx_sel == "mixto":
                ri_ = 0.5 * R_i[[c for c in R_i if c in miembros["sob"]]].mean(axis=1) + 0.5 * R_i[[c for c in R_i if c in miembros["fija"]]].mean(axis=1)
            else:
                ri_ = R_i.mean(axis=1)
            rc_ = (RPF[CPF.index] * CPF["peso"]).sum(axis=1)
            comun_ = rc_.index.intersection(ri_.index)[-per_idx:]
            AC_, ST_ = H.comparar_indice(rc_.loc[comun_], ri_.loc[comun_])
            if not ST_:
                st.info("No hay historia suficiente para comparar.")
            else:
                j1_, j2_, j3_, j4_ = st.columns(4)
                j1_.metric("Retorno de tu cartera", f"{ST_['cartera']['retorno']:+.2%}", f"índice {ST_['indice']['retorno']:+.2%}",
                           delta_color="off", help="Acumulado en el período, en la moneda elegida arriba (pesos o dólares).")
                j2_.metric("Le ganó al índice por", f"{(ST_['cartera']['retorno'] - ST_['indice']['retorno']) * 100:+.2f} pp", delta_color="off")
                j3_.metric("Volatilidad anual", f"{ST_['cartera']['vol']:.1%}", f"índice {ST_['indice']['vol']:.1%}", delta_color="off")
                j4_.metric("Máxima caída", f"{ST_['cartera']['max_dd']:.1%}", f"índice {ST_['indice']['max_dd']:.1%} · beta {ST_['beta']:.2f}", delta_color="off",
                           help="Máxima caída: la peor baja desde un máximo. Beta: cuánto se mueve tu cartera cuando el índice se mueve 1%.")
                fig, ax = plt.subplots(figsize=(11, 3.4))
                ax.plot(AC_.index, (AC_["cartera"] - 1) * 100, color=AZUL, lw=1.8, label="Tu cartera")
                ax.plot(AC_.index, (AC_["indice"] - 1) * 100, color=GRIS, lw=1.4, label="Índice")
                ax.axhline(0, color=GRIS, lw=.6); ax.set_ylabel(f"Retorno acumulado % ({'dólares' if moneda_pf == 'USD' else 'pesos'})")
                ax.grid(alpha=.25); ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout(); ver_grafico(fig)
                st.caption(f"Del {ST_['desde']:%d/%m/%Y} al {ST_['hasta']:%d/%m/%Y} ({ST_['ruedas']} ruedas). Índice de igual peso con "
                           f"{len(miembros[idx_sel])} bonos (los que hoy están en la lista), rebalanceado cada día; los días sin precio de un bono se "
                           "completan como en el VaR. Correlación con el índice: " + f"{ST_['corr']:.2f}.")

        with st.expander("Glosario: qué significa cada número"):
            for k, v in GLOSARIO_PF.items():
                st.markdown(f"**{k}.** {v}")


# ── Flujos de fondos: todos los bonos ──
with tabF:
    USDU = B.universo_usd()
    DL_TK = {t for t, _, _ in P.DL} | {"TMVE8"}
    pesos_tk = [t for k, _, _ in P.TIPOS if len(RP[k]) for t in RP[k].index]
    tipo_de = {t: nombre for k, nombre, _ in P.TIPOS if len(RP[k]) for t in RP[k].index}
    ons_f = list(ON_OK.sort_values("volumen_usd", ascending=False).index) if len(ON_OK) else []
    f1, f2, f3 = st.columns([1.6, 1, 1])
    bono_f = f1.selectbox("Bono", list(tabla_usd.index) + pesos_tk + ons_f,
                          format_func=lambda t: (f"{t} · {tabla_usd.loc[t, 'familia']} (USD)" if t in tabla_usd.index else
                                                 f"{t} · ON {ON_OK.loc[t, 'emisor']} (USD)" if t in ons_f else f"{t} · {tipo_de[t]} ($)"),
                          help="Primero los soberanos en dólares, después los bonos en pesos y al final las obligaciones negociables (de la más operada a la menos).")
    es_on = bono_f in ons_f
    es_usd = bono_f in USDU or es_on
    vn_usd = es_usd or bono_f in DL_TK  # el valor nominal está en dólares
    k_tipo = None if es_usd else next(k for k, _, _ in P.TIPOS if len(RP[k]) and bono_f in RP[k].index)
    px_def = (float(ON_OK.loc[bono_f, "precio"]) if es_on else float(tabla_usd.loc[bono_f, "precio"]) if es_usd
              else float(RP[k_tipo].loc[bono_f, "precio"]))
    # claves fijas para los widgets; al cambiar de bono (o de moneda) se reponen los valores por defecto
    if st.session_state.get("_flujos_bono") != (bono_f, moneda):
        if st.session_state.get("_flujos_vn_usd") != vn_usd:
            st.session_state["vn_flujos"] = 1000 if vn_usd else 100000
        st.session_state["px_flujos"] = round(px_def, 2)
        st.session_state["_flujos_bono"], st.session_state["_flujos_vn_usd"] = (bono_f, moneda), vn_usd
    vn_f = f2.number_input("Nominales", min_value=1, step=100, key="vn_flujos")
    px_f = f3.number_input("Precio de compra", min_value=0.01, step=0.1, key="px_flujos")
    if es_usd:
        if es_on:
            o_ = ON_OK.loc[bono_f]
            g1, g2, g3 = st.columns(3)
            cup_f = g1.number_input("Cupón anual %", 0.0, 30.0, float(o_["cupon"] * 100), 0.125, key=f"fl_cup_{bono_f}", help="De la ficha técnica de BYMA.") / 100
            frec_f = g2.selectbox("Paga", [2, 4, 12, 1], index=[2, 4, 12, 1].index(int(o_["frecuencia"])), key=f"fl_frec_{bono_f}",
                                  format_func={1: "anual", 2: "semestral", 4: "trimestral", 12: "mensual"}.get,
                                  help="Supuesto: semestral (la ficha no lo informa)." if o_["frec_supuesta"] else "De la ficha técnica de BYMA.")
            cuo_f = g3.number_input("Cuotas de capital que faltan", 1, 60, int(o_["cuotas"]), key=f"fl_cuo_{bono_f}", help="1 = paga todo al vencimiento.")
            fl = X.flujos_on(o_["vence"], cup_f, frec_f, int(cuo_f), liq_usd)
            y_f, md_f = X.tir_on(px_f, fl, liq_usd)
            ic_f = X.interes_corrido_on(o_["vence"], cup_f, frec_f, liq_usd)
            m_ = {"tir": y_f, "md": md_f, "residual": 100.0, "paridad": px_f / (100 + ic_f)}
        else:
            b = USDU[bono_f]
            fl = [(f, c, a) for f, c, a in b["flujos"] if f > liq_usd]
            m_ = B.metricas_bono(b, liq_usd, px_f)
        res, filas = m_["residual"], []
        for f, c, a in fl:
            res -= a
            t_ = (f - liq_usd).days
            filas.append({"fecha": f, "paga": L.siguiente_habil(f), "dias": t_, "renta": c, "capital": a, "total": c + a, "residual": res,
                          "vp": (c + a) / (1 + m_["tir"]) ** (t_ / 365)})
        F = pd.DataFrame(filas).set_index("fecha")
        k = vn_f / 100
        if es_on:
            st.caption(f"{bono_f} · ON de {ON_OK.loc[bono_f, 'emisor']} ({ON_OK.loc[bono_f, 'clase']}) · precio de data912 en dólar "
                       f"{'cable' if ON_OK.loc[bono_f, 'simbolo'].endswith('C') else 'MEP'} ({ON_OK.loc[bono_f, 'simbolo']}) · liquida {liq_usd:%d/%m/%Y}. "
                       "Las fechas se cuentan hacia atrás desde el vencimiento y las cuotas de capital van en los últimos pagos: es una aproximación, "
                       "el cronograma exacto está en el prospecto.")
        else:
            st.caption(f"{bono_f} · precio en dólar {'cable (' + bono_f + 'C)' if moneda == 'ccl' else 'MEP (' + bono_f + 'D)'} · liquida {liq_usd:%d/%m/%Y} · "
                       "cambiá MEP / cable en la barra lateral")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Inversión", f"US$ {num_es(px_f * k)}", f"{num_es(vn_f, 0)} VN a {num_es(px_f)}", delta_color="off")
        c2.metric("Total a cobrar", f"US$ {num_es(F['total'].sum() * k)}", f"{F['total'].sum() / px_f - 1:+.1%} sobre lo invertido", delta_color="off")
        c3.metric("Renta / amortización", f"US$ {num_es(F['renta'].sum() * k)}", f"amortización US$ {num_es(F['capital'].sum() * k)}", delta_color="off")
        c4.metric("TIR", f"{m_['tir']:.2%}", f"MD {m_['md']:.2f} · paridad {m_['paridad']:.1%}", delta_color="off")
        c5.metric("Próximo pago", f"{F['paga'].iloc[0]:%d/%m/%Y}", f"US$ {num_es(F['total'].iloc[0] * k)}", delta_color="off")
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.bar(range(len(F)), F["renta"] * k, color=AZUL, label="Renta")
        ax.bar(range(len(F)), F["capital"] * k, bottom=F["renta"] * k, color="#1baf7a", label="Amortización")
        paso = max(1, len(F) // 12)
        ax.set_xticks(range(0, len(F), paso), [f"{d:%m/%y}" for d in F.index[::paso]], fontsize=8)
        ax.set_ylabel("US$"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); ver_grafico(fig)
        T_ = F.copy()
        T_[["renta", "capital", "total", "vp"]] *= k
        mostrar(T_[["paga", "dias", "renta", "capital", "total", "residual", "vp"]],
                {"paga": "{:%d/%m/%Y}", "dias": "{:.0f}", "renta": "{:,.2f}", "capital": "{:,.2f}", "total": "{:,.2f}", "residual": "{:.2f}%", "vp": "{:,.2f}"},
                {"paga": "Se paga el", "dias": "Días", "renta": "Renta (US$)", "capital": "Amortización (US$)", "total": "Total (US$)",
                 "residual": "Residual luego del pago", "vp": "Valor presente"})
    else:
        fila = RP[k_tipo].loc[bono_f]
        F = RP["flujos"][bono_f].copy()
        F["total"] = F["renta"] + F["capital"]
        s0 = RP["liq"]
        F["dias"] = [(f - s0).days for f in F["fecha"]]
        y_nom = P._tir(list(zip(F["fecha"], F["total"])), s0, px_f)
        F["vp"] = F["total"] / (1 + y_nom) ** (F["dias"] / 365)
        k = vn_f / 100
        st.caption(f"{bono_f} · {tipo_de[bono_f]} · {fila.get('condicion', '')} · liquida {s0:%d/%m/%Y}. Los pagos en pesos se proyectan con los supuestos de la "
                   "pestaña Tesoro en pesos (CER con IPC publicado → REM → inflación elegida; TAMAR supuesta; devaluación implícita o elegida).")
        cols = st.columns(5)
        cols[0].metric("Inversión", f"$ {num_es(px_f * k, 0)}", f"{num_es(vn_f, 0)} VN a {num_es(px_f)}", delta_color="off")
        cols[1].metric("Total a cobrar (proyectado)", f"$ {num_es(F['total'].sum() * k, 0)}", f"{F['total'].sum() / px_f - 1:+.1%} sobre lo invertido", delta_color="off")
        if "real" in F:
            y_real = P._tir(list(zip(F["fecha"], F["real"])), s0, px_f)
            cols[2].metric("En pesos de hoy", f"$ {num_es(F['real'].sum() * k, 0)}", f"TIR real {y_real:.2%}", delta_color="off")
        elif "usd" in F:
            cols[2].metric("Cobro en dólares", f"US$ {num_es(F['usd'].sum() * k)}", f"al A3500 estimado $ {num_es(F['tc'].iloc[0])}", delta_color="off")
        elif "pata1" in F:
            cols[2].metric("Paga la pata", fila["paga"], f"{fila['patas']}: {num_es(F['pata1'].iloc[0])} vs {num_es(F['pata_tamar'].iloc[0])}", delta_color="off")
        else:
            cols[2].metric("Renta / interés", f"$ {num_es(F['renta'].sum() * k, 0)}", f"capital $ {num_es(F['capital'].sum() * k, 0)}", delta_color="off")
        cols[3].metric("TIREA nominal estimada", f"{y_nom:.2%}", "en pesos", delta_color="off")
        cols[4].metric("Próximo pago", f"{F['fecha'].iloc[0]:%d/%m/%Y}", f"$ {num_es(F['total'].iloc[0] * k, 0)}", delta_color="off")
        fig, ax = plt.subplots(figsize=(11, 3.0))
        ax.bar(range(len(F)), F["renta"] * k, color=AZUL, label="Renta / interés")
        ax.bar(range(len(F)), F["capital"] * k, bottom=F["renta"] * k, color="#1baf7a", label="Capital")
        ax.set_xticks(range(len(F)), [f"{d:%m/%y}" for d in F["fecha"]], fontsize=8)
        ax.set_ylabel("$"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); ver_grafico(fig)
        T_ = F.set_index("fecha")
        for c in ["renta", "capital", "total", "vp"] + [c for c in ("real", "usd") if c in T_]:
            T_[c] = T_[c] * k
        cols_t = ["dias", "renta", "capital", "total"] + [c for c in ("real", "usd") if c in T_] + ["vp"]
        mostrar(T_[cols_t], {"dias": "{:.0f}", "renta": "{:,.2f}", "capital": "{:,.2f}", "total": "{:,.2f}", "real": "{:,.2f}", "usd": "{:,.2f}", "vp": "{:,.2f}"},
                {"dias": "Días", "renta": "Renta / interés", "capital": "Capital", "total": "Total proyectado", "real": "En pesos de hoy", "usd": "En dólares",
                 "vp": "Valor presente"})

    st.subheader("Próximo pago de cada bono")
    grupo_f = st.radio("Mostrar", ["Todos", "En dólares", "En pesos", "ONs"], horizontal=True, key="grupo_flujos",
                       format_func={"Todos": "Todos", "En dólares": "Soberanos en dólares", "En pesos": "En pesos", "ONs": "Obligaciones negociables"}.get)
    prox = []
    if grupo_f in ("Todos", "ONs"):
        for t, o_ in ON_OK.iterrows():
            f, c, a = o_["flujos"][0]
            prox.append({"ticker": t, "tipo": f"ON {o_['emisor']} (USD)", "moneda": "US$", "pago": L.siguiente_habil(f) if not L.habil(f) else f,
                         "que": "Vence" if f >= o_["vence"] else "Renta y amortización" if a else "Renta", "renta": c, "capital": a,
                         "quedan": len(o_["flujos"]), "vence": o_["vence"]})
    if grupo_f in ("Todos", "En dólares"):
        for t, b in {**USDU, **L._bopreales()}.items():
            fut = [(f, c, a) for f, c, a in b["flujos"] if f > liq_usd]
            if fut:
                f, c, a = fut[0]
                prox.append({"ticker": t, "tipo": "Bopreal (USD)" if t.startswith("BPO") else b.get("familia", "") + " (USD)", "moneda": "US$",
                             "pago": L.siguiente_habil(f), "que": "Vence" if f >= b["venc"] else "Renta y amortización" if a else "Renta",
                             "renta": c, "capital": a, "quedan": len(fut), "vence": b["venc"]})
    if grupo_f in ("Todos", "En pesos"):
        for t in pesos_tk:
            Fx = RP["flujos"].get(t)
            if Fx is None or Fx.empty:
                continue
            r0 = Fx.iloc[0]
            prox.append({"ticker": t, "tipo": tipo_de[t], "moneda": "$", "pago": r0["fecha"],
                         "que": "Vence" if len(Fx) == 1 else "Renta y amortización" if r0["capital"] else "Renta",
                         "renta": r0["renta"], "capital": r0["capital"], "quedan": len(Fx), "vence": Fx["fecha"].iloc[-1]})
    if prox:
        PX = pd.DataFrame(prox).sort_values(["pago", "ticker"]).set_index("ticker")
        PX["total"] = PX["renta"] + PX["capital"]
        PX["dias"] = [(d - liq_usd).days for d in PX["pago"]]
        mostrar(PX[["tipo", "pago", "dias", "que", "moneda", "renta", "capital", "total", "quedan", "vence"]],
                {"pago": "{:%d/%m/%Y}", "dias": "{:.0f}", "renta": "{:,.2f}", "capital": "{:,.2f}", "total": "{:,.2f}", "quedan": "{:.0f}", "vence": "{:%d/%m/%Y}"},
                {"tipo": "Tipo", "pago": "Próximo pago", "dias": "Días", "que": "Qué paga", "moneda": "Moneda", "renta": "Renta / interés",
                 "capital": "Amortización / capital", "total": "Total", "quedan": "Pagos que quedan", "vence": "Vence"})
    else:
        st.info("No hay bonos para mostrar con ese filtro (las ONs necesitan precio de data912).")
    st.caption("Por 100 VN originales. Bonos en dólares: pagan en dólares (fechas corridas al día hábil siguiente). Bonos en pesos: CER, TAMAR, duales y dólar linked "
               "son estimaciones con los supuestos de la pestaña Tesoro en pesos. ONs: condiciones de la ficha técnica de BYMA (pagos semestrales si la ficha no "
               "informa la frecuencia; fechas contadas hacia atrás desde el vencimiento), una aproximación al prospecto.")

# ── 4. LLM ──
with tab4:
    st.subheader("Informe semanal con IA")
    st.caption("Arma un PDF con el resumen ejecutivo de cómo se movieron en la última semana los precios y las tasas de los bonos, el dólar, el "
               "riesgo país, la TAMAR, las obligaciones negociables más operadas, la licitación del Tesoro, el carry trade con la banda, el mercado de dinero "
               "y la agenda de las próximas dos semanas (licitación, pagos del Tesoro y qué ONs pagan y qué día). Si tenés un portafolio cargado, "
               "también cuenta cómo le fue en la semana y explica sus indicadores. La app calcula todos los números (último cierre contra el de una semana antes) "
               "y un modelo de lenguaje (Claude) solo los redacta: recibe un JSON y tiene prohibido inventar datos. Sin clave, el texto sale de una "
               "plantilla con los mismos números.")
    try:  # publicada en Streamlit Community Cloud, la clave se guarda en Settings › Secrets y no hace falta escribirla
        clave_guardada = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        clave_guardada = ""
    if clave_guardada:
        os.environ["ANTHROPIC_API_KEY"] = clave_guardada
        st.caption("Clave de Anthropic cargada desde los Secrets de la app: el texto lo redacta Claude.")
        clave = ""
    else:
        clave = st.text_input("ANTHROPIC_API_KEY (opcional)", type="password", help="Sin clave se usa una plantilla determinística con los mismos números.")
    hay_pf = not CPF.empty
    incluir_pf = st.checkbox("Incluir mi portafolio (pestaña «Mi portafolio») en el informe", value=hay_pf, disabled=not hay_pf,
                             help="Suma dos párrafos y una tabla: cómo le fue a tu cartera en la semana (retorno, precios y aporte de cada bono) "
                                  "y qué dicen sus indicadores (duración, DV01, convexidad, tipo de cartera, exposición y VaR).")
    if hay_pf:
        st.caption(f"Tu portafolio hoy: {len(CPF)} bonos, valor {'US$' if moneda_pf == 'USD' else '$'} {num_es(float(CPF['valor'].sum()), 0)}.")
    if st.button("Generar el informe de la semana", type="primary"):
        if clave:
            os.environ["ANTHROPIC_API_KEY"] = clave
        with st.spinner("Bajando los cierres de la última semana de cada bono y armando el PDF… (en vivo tarda unos 20 segundos)"):
            top_on = ON_OK.sort_values("volumen_usd", ascending=False).head(15) if len(ON_OK) else ON_OK
            ons_inf = tuple((t, r["emisor"], r["vence"].isoformat(), float(r["cupon"]), int(r["frecuencia"]), int(r["cuotas"]),
                             t + "D", float(r["volumen_usd"])) for t, r in top_on.iterrows())
            S_ = informe_semana(en_vivo, date.today(), ons_inf)
            extra_ = {}
            ipc_pub0, rem0 = D["macro"]["ipc"].dropna().to_dict(), D["rem_mensual"]["mediana"].dropna().to_dict()
            ipc_f = lambda m: (ipc_pub0.get(m) or rem0.get(m) or list(rem0.values())[-1]) / 100   # IPC publicado y, después, el REM
            futuro_f = None
            try:   # futuros de dólar de A3: el contrato de diciembre y la curva que usa el carry trade
                tc_s = DP["a3500"].sort_index()
                f_fut = FUT["fecha"].iloc[0]
                spot_f = float(tc_s[tc_s.index <= pd.Timestamp(f_fut)].iloc[-1]) if (tc_s.index <= pd.Timestamp(f_fut)).any() else float(tc_s.iloc[-1])
                TF = X.devaluacion_futuros(FUT, spot_f, f_fut)
                futuro_f = lambda d: X.futuro_en(TF, spot_f, d)
                dic_ = TF[TF["vence"].map(lambda v: v.month == 12)]
                fila_f = dic_.iloc[0] if len(dic_) else TF.iloc[-1]
                extra_["dolar_futuro"] = {"a3500": round(spot_f, 2), "contrato": fila_f["contrato"], "precio": round(float(fila_f["ajuste"]), 1),
                                          "devaluacion_%": round(float(fila_f["dev_directa"]) * 100, 2), "mensual_%": round(float(fila_f["mensual"]) * 100, 2)}
            except Exception:
                pass
            try:
                T0 = P.trayectoria_be(RP0)
                fut_be = T0[T0["fuente"] != "INDEC (publicado)"]["inflacion"].head(4)
                extra_["breakeven_proximos_meses_%"] = {B.mes_corto(m): round(float(v) * 100, 2) for m, v in fut_be.items()}
                rem_ = T0.loc[fut_be.index, "rem"].dropna()
                extra_["rem_mismos_meses_%"] = {B.mes_corto(m): round(float(v) * 100, 2) for m, v in rem_.items()}
            except Exception:
                pass
            try:
                UST_c, _, _ = cargar_ust(en_vivo, D["fecha"])
                sp_g = X.spread_ust(tabla_usd, UST_c)
                extra_["spread_globales_vs_tesoro_eeuu_pb"] = int(round(sp_g.loc[sp_g["familia"] == "Globales", "spread_pb"].median()))
            except Exception:
                pass
            try:
                _, _, PR, _ = cargar_licitaciones(en_vivo, date.today())
                extra_["proxima_licitacion"] = f"{pd.Timestamp(PR['lic']):%d/%m/%Y}" + (" (" + ", ".join(m[0] for m in PR["menu"]) + ")" if PR["menu"] else "")
            except Exception:
                pass
            try:   # carry trade con la banda, medido con el dólar oficial
                a35_i = float(DP["a3500"].sort_index().iloc[-1])
                CT_i = H.tabla_carry(RP0, a35_i, ipc_f, futuro_f, 1.0)
                extra_["carry_trade"] = I.contexto_carry(CT_i, a35_i, *H.banda(date.today(), ipc_f))
            except Exception:
                pass
            try:   # mercado de dinero contra la LECAP más corta
                DN_i, _ = cargar_dinero(en_vivo)
                lec_i = RP0["lecap"]
                lec_i = lec_i[(lec_i["dias"] >= 14) & lec_i["pago_final"].notna()].sort_values("dias") if len(lec_i) else lec_i
                ref_i = lec_i.iloc[0] if len(lec_i) else None
                extra_["mercado_de_dinero"] = I.contexto_dinero(H.resumen_dinero(DN_i, ref_i), ref_i)
            except Exception:
                pass
            try:   # ONs de los grandes emisores: TIR y spread contra los Globales
                extra_["ons_grandes_emisores"] = I.contexto_emisores(H.spread_ons(ON_OK, tabla_usd), H.EMISORES_TOP)
            except Exception:
                pass
            try:   # lo que viene: licitación, pagos del Tesoro y de los soberanos, y pagos de ONs
                hoy_i = max(date.today(), D["fecha"])
                extra_["agenda_proximos_dias"] = I.agenda(hoy_i, hoy_i + timedelta(days=14), ONS_EV)
            except Exception:
                pass
            if incluir_pf:   # tu cartera: cómo le fue en la semana y sus indicadores
                try:
                    fx_pf = ("AL30", "AL30C" if moneda == "ccl" else "AL30D")
                    H_pf = I.historia_cartera(H_, sorted({UPF[t]["simbolo"] for t in CPF.index} | set(fx_pf)), en_vivo)
                    R_pf, _ = PF.retornos(UPF, CPF, H_pf, moneda_pf, fx_pf)
                    extra_["portafolio"] = I.contexto_portafolio(CPF, R_pf, H_pf, UPF, S_["ini"], S_["fin"], moneda_pf, S_["tabla"], S_.get("ons"))
                except Exception:
                    pass
            extra_ = {k: v for k, v in extra_.items() if v}
            ctx_ = {**S_["contexto"], **extra_}
            texto_, origen_ = I.redactar(ctx_)
            pdf_ = I.pdf(S_, texto_, origen_, ctx_)
            st.session_state["informe"] = {"texto": texto_, "origen": origen_, "pdf": pdf_, "ctx": ctx_, "ini": S_["ini"], "fin": S_["fin"],
                                           "ind": S_["indicadores"], "tabla": S_["tabla"], "ons": S_.get("ons", pd.DataFrame())}
    INF = st.session_state.get("informe")
    if INF:
        st.markdown(f"**Semana del {INF['ini']:%d/%m/%Y} al {INF['fin']:%d/%m/%Y}** · {INF['ctx']['semana']['precios']} · texto: {INF['origen']}")
        cols_ = st.columns(4)
        for c_, (k, fmt, unidad) in zip(cols_, [("riesgo_pais", lambda v: f"{v:.0f} pb", "pb"), ("mep", lambda v: f"$ {num_es(v)}", "$"),
                                                ("tir_soberanos", lambda v: f"{num_es(v)}%", "pp"), ("tirea_fija", lambda v: f"{num_es(v)}%", "pp")]):
            v = INF["ind"].get(k)
            if v and np.isfinite(v["fin"]) and np.isfinite(v["inicio"]):
                d_ = (f"{(v['fin'] / v['inicio'] - 1):+.2%}" if unidad == "$" else f"{v['fin'] - v['inicio']:+.0f} pb" if unidad == "pb"
                      else f"{(v['fin'] - v['inicio']) * 100:+.0f} pb")
                c_.metric(v["nombre"], fmt(v["fin"]), d_, delta_color="inverse" if k in ("riesgo_pais", "tir_soberanos", "tirea_fija") else "normal")
        st.download_button("Descargar el informe en PDF", INF["pdf"], file_name=f"BondAI_informe_semanal_{INF['fin']:%Y%m%d}.pdf", mime="application/pdf",
                           type="primary", on_click="ignore")
        for par in [x.strip() for x in INF["texto"].split("\n\n") if x.strip()]:
            m_ = re.match(r"^([^:]{3,40}):\s*(.*)$", par, re.S)
            st.markdown((f"**{m_.group(1)}:** {m_.group(2)}" if m_ else par).replace("$", "\\$"))
        with st.expander("Bonos de la semana (precio y tasa al inicio y al final)"):
            Tw = INF["tabla"]
            if len(Tw):
                mostrar(Tw.sort_values(["grupo", "var"]), {"precio_ini": "{:,.2f}", "precio_fin": "{:,.2f}", "var": "{:+.2%}", "tasa_ini": "{:.2%}",
                                                           "tasa_fin": "{:.2%}", "var_pb": "{:+.0f}", "md": "{:.2f}"},
                        {"grupo": "Tipo", "precio_ini": f"Precio {INF['ini']:%d/%m}", "precio_fin": f"Precio {INF['fin']:%d/%m}", "var": "Var. precio",
                         "tasa_ini": f"Tasa {INF['ini']:%d/%m}", "tasa_fin": f"Tasa {INF['fin']:%d/%m}", "var_pb": "Cambio tasa (pb)", "md": "MD"},
                        calor={"var": "var"})
        if len(INF.get("ons", [])):
            with st.expander("Obligaciones negociables más operadas (precio y TIR al inicio y al final)"):
                mostrar(INF["ons"].sort_values("volumen_usd", ascending=False),
                        {"precio_ini": "{:,.2f}", "precio_fin": "{:,.2f}", "var": "{:+.2%}", "tir_ini": "{:.2%}", "tir_fin": "{:.2%}", "var_pb": "{:+.0f}",
                         "volumen_usd": "{:,.0f}"},
                        {"emisor": "Emisor", "precio_ini": f"Precio {INF['ini']:%d/%m}", "precio_fin": f"Precio {INF['fin']:%d/%m}", "var": "Var. precio",
                         "tir_ini": f"TIR {INF['ini']:%d/%m}", "tir_fin": f"TIR {INF['fin']:%d/%m}", "var_pb": "Cambio TIR (pb)", "volumen_usd": "Volumen hoy US$"},
                        calor={"var": "var"})
        PFi = INF["ctx"].get("portafolio")
        if PFi:
            with st.expander(f"Tu portafolio en la semana: {'' if PFi['retorno_semana_%'] is None else num_es(PFi['retorno_semana_%']) + '%'}", expanded=True):
                BD_ = pd.DataFrame(PFi["bonos_detalle"]).set_index("ticker")
                mostrar(BD_[["grupo", "peso_%", "precio_ini", "precio_fin", "var_precio_%", "retorno_semana_%", "aporte_pp", "cambio_tasa_pb"]],
                        {"peso_%": "{:.1f}%", "precio_ini": "{:,.2f}", "precio_fin": "{:,.2f}", "var_precio_%": "{:+.2f}%", "retorno_semana_%": "{:+.2f}%",
                         "aporte_pp": "{:+.2f}", "cambio_tasa_pb": "{:+.0f}"},
                        {"grupo": "Grupo", "peso_%": "Peso", "precio_ini": f"Precio {INF['ini']:%d/%m}", "precio_fin": f"Precio {INF['fin']:%d/%m}",
                         "var_precio_%": "Var. precio", "retorno_semana_%": "Retorno", "aporte_pp": "Aporte (pp)", "cambio_tasa_pb": "Cambio de tasa (pb)"})
                st.caption("Retorno en la moneda de la cartera, con cupones y tipo de cambio; aporte = peso × retorno (la suma da el retorno de la "
                           "cartera). Los indicadores y su explicación están en el texto y en el PDF.")
        AGi = INF["ctx"].get("agenda_proximos_dias")
        if AGi:
            with st.expander(f"Agenda del {AGi['desde'][:5]} al {AGi['hasta'][:5]}: licitación, pagos del Tesoro y {len(AGi['pagos_de_ons'])} pagos de ONs"):
                if AGi["tesoro_y_soberanos"]:
                    mostrar(pd.DataFrame(AGi["tesoro_y_soberanos"]).set_index("fecha"), {},
                            {"dia": "Día", "que": "Qué pasa", "instrumentos": "Bonos", "detalle": "Detalle"})
                if AGi["pagos_de_ons"]:
                    PO_ = pd.DataFrame(AGi["pagos_de_ons"]).set_index("fecha")
                    mostrar(PO_[["dia", "ticker", "emisor", "clase", "que", "renta_usd_por_100", "capital_usd_por_100"]],
                            {"renta_usd_por_100": "{:,.3f}", "capital_usd_por_100": "{:,.2f}"},
                            {"dia": "Día", "ticker": "ON", "emisor": "Emisor", "clase": "Clase", "que": "Qué paga", "renta_usd_por_100": "Renta US$ por 100 VN",
                             "capital_usd_por_100": "Capital US$ por 100 VN"})
                    st.caption("Fechas de las ONs estimadas (desde el vencimiento con la frecuencia de la ficha de BYMA, al día hábil siguiente).")
        with st.expander("Contexto enviado al modelo (JSON)"):
            st.json(I._para_llm(INF["ctx"]))
    else:
        st.info("Tocá «Generar el informe de la semana». Después aparece el botón para descargar el PDF.")
