"""
BondAI · prototipo en Streamlit
Ejecutar:  pip install -r requirements.txt   →   streamlit run app.py
Publicada en internet: ver README.md (GitHub + Streamlit Community Cloud).
Usa las mismas funciones que el notebook (bondai.py, pesos.py, mercado.py, dolares.py, licitaciones.py, externos.py y portafolio.py
deben estar en la misma carpeta).
"""
import os
import time
from datetime import date, datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import bondai as B
import dolares as DL
import externos as X
import licitaciones as L
import mercado as M
import pesos as P
import portafolio as PF

st.set_page_config(page_title="BondAI · Renta fija con IA", page_icon="📈", layout="wide")


# ── datos y cálculos (cacheados para que la app responda rápido) ──
@st.cache_data(ttl=900, show_spinner="Bajando datos de BYMA y BCRA…")
def cargar(en_vivo):
    d = B.cargar_datos(usar_api=en_vivo)
    d["bajado"] = datetime.now(timezone(timedelta(hours=-3)))  # hora en que se bajaron los datos
    return d


@st.cache_data(ttl=900, show_spinner="Calculando curvas y breakeven…")
def calcular(en_vivo, moneda="mep"):
    d = cargar(en_vivo)
    tabla_usd, liq_usd, _ = B.analisis_usd(d, moneda)
    curva, tabla_be, ref0, cer_ref, liq_ars = B.analisis_breakeven(d)
    mensual = B.inflacion_implicita(d, curva, cer_ref, liq_ars)
    return tabla_usd, liq_usd, curva, tabla_be, cer_ref, liq_ars, mensual


@st.cache_data(ttl=900, show_spinner="Bajando bonos del Tesoro en pesos…")
def cargar_pesos(en_vivo):
    return P.cargar(usar_api=en_vivo)


@st.cache_data(ttl=900, show_spinner="Bajando los paneles de BYMA…")
def cargar_paneles(en_vivo):
    return M.cargar_paneles(usar_api=en_vivo)


@st.cache_data(ttl=3600, show_spinner="Bajando el histórico de volumen de BYMA, de a un bono por vez (tarda alrededor de un minuto)…")
def cargar_volumen(en_vivo):
    return M.cargar_volumen(usar_api=en_vivo)


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


@st.cache_resource(ttl=900, show_spinner="Comparando contra el cierre anterior…")
def variacion_mercado(en_vivo, grupo, tamar, infl, duales):
    d = cargar(en_vivo)
    return M.variacion(grupo, cargar_paneles(en_vivo), cargar_pesos(en_vivo), tamar=tamar, infl=infl, duales=duales,
                       ipc_publicado=d["macro"]["ipc"].dropna().to_dict(), rem_mensual=d["rem_mensual"]["mediana"].dropna().to_dict())


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
    calcular_pesos.clear(); variacion_mercado.clear()


def borrar_precios():
    """Borra de la memoria solo los precios y tasas (no el volumen, que tarda en volver a bajarse)."""
    for f in (cargar, calcular, cargar_pesos, cargar_paneles, cargar_dolares, calcular_pesos, variacion_mercado, cargar_futuros, cargar_ons, cargar_riesgo):
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
FUT, F_FUT = cargar_futuros(en_vivo)
RIESGO, F_RIESGO = cargar_riesgo(en_vivo)
ONS, F_ONS = cargar_ons(en_vivo)
for k, v in [("futuros", F_FUT), ("riesgo país", F_RIESGO), ("ONs", F_ONS)]:
    st.sidebar.write(f"· {k}: {v}")
st.sidebar.caption(f"Fecha de análisis: {D['fecha']:%d/%m/%Y} · datos bajados a las {D['bajado']:%H:%M} (hora argentina)")
st.sidebar.caption("Herramienta académica. No es recomendación de inversión.")

tabla_usd, liq_usd, curva, tabla_be, cer_ref, liq_ars, mensual = calcular(en_vivo, moneda)

st.title("BondAI · Renta fija argentina con IA")
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
                                      "cupón cero más largo que tiene LECAP o BONCAP con la cual compararse (pestaña «Inflación breakeven»).")
rp_ult, rp_ant = float(RIESGO.iloc[-1]), float(RIESGO.iloc[-2])
c5.metric("Riesgo país", f"{rp_ult:,.0f} pb".replace(",", "."), f"{rp_ult - rp_ant:+.0f} pb al {RIESGO.index[-1]:%d/%m}", delta_color="inverse",
          help="EMBI de JP Morgan: cuántos puntos básicos (100 pb = 1 punto porcentual) más rinden los bonos argentinos en dólares que los "
               "del Tesoro de EE.UU. Sube cuando el mercado de afuera ve más riesgo (pestaña «Riesgo país»).")
c3.metric("REM próximo mes", f"{D['rem_mensual']['mediana'].iloc[0]:.2f}%", f"{D['rem_mensual'].index[0]}", delta_color="off")
c4.metric("Último IPC", f"{D['macro']['ipc'].dropna().iloc[-1]:.1f}%", D["macro"]["ipc"].dropna().index[-1], delta_color="off")

(tab1, tabR, tabN, tabP, tab2, tabFU, tabW, tabPF, tabV, tabO, tabF, tabD, tabL, tab4) = st.tabs(
    ["Curva en dólares", "Riesgo país", "ONs", "Tesoro en pesos", "Inflación breakeven", "Devaluación esperada", "Tasas forward", "Mi portafolio",
     "Variación diaria", "Volumen", "Flujos de fondos", "Dólares", "Licitaciones", "Resumen con IA"])

# ── 1. curva USD ──
with tab1:
    izq, der = st.columns([1.2, 1])
    with izq:
        st.subheader("Métricas por bono")
        st.caption(f"Precios en dólar {'cable (tickers C)' if moneda == 'ccl' else 'MEP (tickers D)'}, liquidación {liq_usd:%d/%m/%Y}. "
                   "Spread = TIR − curva Nelson-Siegel de su legislación. Var.: verde si subió, rojo si bajó; más intenso, más movimiento.")
        tu = tabla_usd.copy()
        tu["var"] = [(PAN["t1"].get(t + SUF) or [0] * 8)[7] for t in tu.index]  # variación del día que informa BYMA
        mostrar(tu[["familia", "precio", "var", "tir", "md", "convexidad", "paridad", "vt", "spread_pb"]],
                {"precio": "{:.2f}", "var": "{:+.2%}", "tir": "{:.2%}", "md": "{:.2f}", "convexidad": "{:.1f}", "paridad": "{:.1%}", "vt": "{:.2f}", "spread_pb": "{:+.0f}"},
                {"familia": "Familia", "precio": "Precio", "var": "Var.", "tir": "TIR", "md": "MD", "convexidad": "Convexidad", "paridad": "Paridad",
                 "vt": "Valor técnico", "spread_pb": "Spread (pb)"}, calor={"var": "var"})
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
        st.pyplot(fig)


# ── gráficos comunes ──
VERDE, ROJO, GRIS, AZUL, VIOLETA = "#16a05a", "#d64545", "#8a96aa", "#2a78d6", "#7a5cf0"


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
                st.pyplot(grafico_pesos(df, "tirea", cf, "#2a78d6" if k == "lecap" else "#1baf7a", "TIREA", "Duration (años)"))
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
                t["be_t"] = [f"{v:.2%}" if np.isfinite(v) else ("n/a" if n == "CER ya publicado" else "—") for v, n in zip(t["be_mensual"], t["be_nota"])]
                mostrar(t[["vence", "dias", "precio", "var", "valor_tecnico", "paridad", "tir_real", "tna_real", "tna_nominal", "md", "spread_pb", "be_t", "rem_mensual", "be_rem_pb", "vol"]],
                        {**COM, "valor_tecnico": "{:,.2f}", "paridad": "{:.1%}", "tir_real": "{:.2%}", "tna_real": "{:.2%}", "tna_nominal": "{:.2%}",
                         "rem_mensual": "{:.2%}", "be_rem_pb": "{:+.0f}"},
                        {**NOM, "valor_tecnico": "VT", "paridad": "Paridad", "tir_real": "TIR real (TEA)", "tna_real": "TNA real 30/365",
                         "tna_nominal": "TNA nominal 30/365", "be_t": "Inflación BE mensual", "rem_mensual": "REM mensual", "be_rem_pb": "BE − REM (pb)"}, calor={"var": "var"})
                if vista_cer.startswith("TIR"):
                    st.pyplot(grafico_cer_tir(d))
                else:
                    st.pyplot(grafico_be(d, RP["max_cero"]))
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
                st.pyplot(grafico_margen_tamar(df, RP["dual"], tamar_sup))
            elif k == "dual":
                st.markdown(f"Con TAMAR en {tamar_sup:.2f}%, pagan la pata TAMAR **{(df['paga'] == 'TAMAR').sum()} de {len(df)}** · el piso es lo que asegura la otra pata aunque la TAMAR caiga")
                t = df.copy()
                t["piso_t"] = [f"{p:.2%} {tp}" for p, tp in zip(t["piso"], t["piso_tipo"])]
                t["cruce_t"] = ["superada" if c < 0.01 else f"{c:.2f}%" for c in t["tamar_cruce"]]
                mostrar(t[["patas", "vence", "dias", "precio", "var", "pago_pata1", "pago_tamar", "paga", "tirea", "piso_t", "cruce_t", "margen_s_tamar", "vol"]],
                        {**COM, "pago_pata1": "{:,.2f}", "pago_tamar": "{:,.2f}", "tirea": "{:.2%}", "margen_s_tamar": "{:.2f}%"},
                        {**NOM, "patas": "Patas", "pago_pata1": "Pago pata 1", "pago_tamar": "Pago TAMAR", "paga": "Paga", "tirea": "TIREA est.", "piso_t": "Piso",
                         "cruce_t": "TAMAR de cruce", "margen_s_tamar": "Margen s/ TAMAR"}, calor={"var": "var"})
                st.pyplot(grafico_pesos(df, "tirea", cf, "#2a78d6", "TIREA estimada"))
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
                st.pyplot(grafico_pesos(df[df["dias"] >= 15], "deval_mensual", None, "#1baf7a", "Devaluación mensual implícita"))
            st.caption(NOTAS[k])

# ── Tasas forward implícitas en las curvas en pesos y en dólares ──
with tabW:
    st.caption("La tasa forward es la tasa que el mercado ya fija hoy para un período futuro. Si una LECAP vence en 3 meses y otra en 6, "
               "comprar la de 6 tiene que rendir lo mismo que comprar la de 3 y renovar: la tasa de esa renovación es la forward entre el mes 3 y el 6. "
               "En dólares es igual: comprar el AO28 tiene que rendir lo mismo que comprar el AO27 y reinvertir hasta el vencimiento del AO28.")
    w1, w2 = st.columns([2.4, 1])
    tipo_w = w1.radio("Curva", list(M.FWD_TIPOS), horizontal=True, format_func=M.FWD_TIPOS.get)
    en_usd = tipo_w in M.FWD_USD
    if en_usd:
        minimo_w = w2.selectbox("Tramo mínimo", [90, 180, 365], format_func={90: "3 meses", 180: "6 meses", 365: "1 año"}.get, key="min_w_usd",
                                help="Medido en duration: se saltean bonos con una duration muy parecida a la del anterior.")
    else:
        minimo_w = w2.selectbox("Tramo mínimo", [14, 30, 60], format_func=lambda d: f"{d} días", key="min_w_pesos",
                                help="Se saltean vencimientos más cercanos que esto al anterior: en tramos muy cortos cualquier ruido en el precio da una forward absurda.")
    FW = M.forwards(RP, tipo_w, minimo_w, tabla_usd)
    if len(FW) < 2:
        st.info("No hay suficientes vencimientos para calcular forwards con ese tramo mínimo.")
    elif en_usd:
        alta, baja = FW["fwd_anual"].idxmax(), FW["fwd_anual"].idxmin()
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Primer tramo (= TIR)", f"{FW['fwd_anual'].iloc[0]:.2%}", f"anual en USD · {FW.index[0]}", delta_color="off")
        u2.metric("Forward más alta", f"{FW.loc[alta, 'fwd_anual']:.2%}", alta, delta_color="off")
        u3.metric("Forward más baja", f"{FW.loc[baja, 'fwd_anual']:.2%}", baja, delta_color="off")
        u4.metric("Último tramo", f"{FW['fwd_anual'].iloc[-1]:.2%}", f"{(FW['fwd_anual'].iloc[-1] - FW['fwd_anual'].iloc[0]) * 1e4:+.0f} pb vs el primero", delta_color="off")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, (tr, r) in enumerate(FW.iterrows()):
            ax.hlines(r["fwd_anual"] * 100, r["desde"] / 365, r["hasta"] / 365, color=AZUL, lw=3, label="Forward de cada tramo" if i == 0 else None)
        ax.scatter(FW["hasta"] / 365, FW["spot_anual"] * 100, color=GRIS, s=26, zorder=3, label="TIR de cada bono (de hoy a su vencimiento)")
        etiquetar(ax, FW["hasta"] / 365, FW["spot_anual"] * 100, [t.split(" → ")[1] for t in FW.index])
        ax.set_xlabel("Duration (años)"); ax.set_ylabel("Tasa anual en dólares (%)")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
        fig.tight_layout(); st.pyplot(fig)
        st.caption(f"Precios en dólar {'cable' if moneda == 'ccl' else 'MEP'} (se cambia en la barra lateral). Cada barra azul es la tasa anual en dólares "
                   "que el mercado ya fija para ese tramo; los puntos grises, la TIR de cada bono. Si las barras quedan arriba de los puntos, la curva tiene pendiente positiva.")
        T = FW[["vence", "desde", "hasta", "dias", "fwd_anual", "spot_anual", "dif_anual_pb"]].copy()
        T[["desde", "hasta", "dias"]] = T[["desde", "hasta", "dias"]] / 365
        fmt = {"vence": "{:%d/%m/%Y}", "desde": "{:.2f}", "hasta": "{:.2f}", "dias": "{:.2f}", "fwd_anual": "{:.2%}", "spot_anual": "{:.2%}", "dif_anual_pb": "{:+.0f}"}
        nom = {"vence": "Vence", "desde": "Desde (años)", "hasta": "Hasta (años)", "dias": "Años del tramo", "fwd_anual": "Forward anual",
               "spot_anual": "TIR del bono", "dif_anual_pb": "Forward − TIR (pb)"}
        if tipo_w == "tesoro":
            BX = M.forwards_bootstrap(tabla_usd, liq_usd, [t.split(" → ")[1] for t in FW.index])
            T["exacta"] = BX["fwd_exacta"].reindex(T.index).values if len(BX) else np.nan
            fmt["exacta"], nom["exacta"] = "{:.2%}", "Forward exacta (bootstrapping)"
        mostrar(T, fmt, nom)
        notas_u = {"tesoro": "Los AO27, AO28 y AO29 pagan un cupón mensual del 6% y devuelven todo el capital al vencimiento. "
                             "Fórmula: (1 + forward) ^ (D2 − D1) = (1 + TIR2) ^ D2 / (1 + TIR1) ^ D1, con D la duration de cada bono. "
                             "Como no amortizan, se puede verificar con el bootstrapping exacto (curva cupón cero, última columna): da prácticamente lo mismo. "
                             "El AN29 queda afuera porque su duration es casi igual a la del AO29.",
                   "bonares": "Los Bonares (ley local) amortizan en cuotas, así que el plazo de cada uno es su duration y no su vencimiento: es la forma habitual en el mercado. "
                              "El bootstrapping exacto no sirve acá: con pocos bonos que devuelven el capital en cuotas da forwards absurdas, incluso negativas. "
                              "El AE38 tiene menos duration que el AL35 aunque vence después, porque empieza a devolver el capital antes (2027 contra 2031).",
                   "globales": "Los Globales (ley Nueva York) amortizan en cuotas, así que el plazo de cada uno es su duration y no su vencimiento: es la forma habitual en el mercado. "
                               "El GD38 tiene menos duration que el GD35 aunque vence después, porque empieza a devolver el capital antes (2027 contra 2031). "
                               "Las forwards de los tramos largos son poco confiables: las durations están muy juntas y una diferencia chica de precio mueve mucho la forward."}
        st.caption(notas_u[tipo_w])
    else:
        unidad = "devaluación mensual" if tipo_w == "dl" else ("tasa real mensual" if tipo_w == "cer" else "TEM")
        alta, baja = FW["fwd_mensual"].idxmax(), FW["fwd_mensual"].idxmin()
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Primer tramo (spot)", f"{FW['fwd_mensual'].iloc[0]:.2%}", f"{unidad} · {FW.index[0]}", delta_color="off")
        u2.metric("Forward más alta", f"{FW.loc[alta, 'fwd_mensual']:.2%}", alta, delta_color="off")
        u3.metric("Forward más baja", f"{FW.loc[baja, 'fwd_mensual']:.2%}", baja, delta_color="off")
        u4.metric("Último tramo", f"{FW['fwd_mensual'].iloc[-1]:.2%}", f"{(FW['fwd_mensual'].iloc[-1] - FW['fwd_mensual'].iloc[0]) * 1e4:+.0f} pb vs el primero", delta_color="off")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, (tr, r) in enumerate(FW.iterrows()):
            ax.hlines(r["fwd_mensual"] * 100, r["desde"], r["hasta"], color=AZUL, lw=3, label="Forward de cada tramo" if i == 0 else None)
        ax.scatter(FW["hasta"], FW["spot_mensual"] * 100, color=GRIS, s=26, zorder=3, label="Tasa spot de cada bono (de hoy a su vencimiento)")
        etiquetar(ax, FW["hasta"], FW["spot_mensual"] * 100, [t.split(" → ")[1] for t in FW.index])
        curva_w = None
        if tipo_w == "fija":
            curva_w = RP["curva_fija"]
        elif tipo_w == "cer":
            ceros = RP["cer"][RP["cer"].index.isin({c[0] for c in P.CERZ}) & (RP["cer"]["dias"] >= 30)]
            curva_w = P.ajustar_ns([(d / 365, y) for d, y in zip(ceros["dias"], ceros["tir_real"])]) if len(ceros) >= 4 else None
        if curva_w:
            FC = M.forward_curva(curva_w, FW["hasta"].max())
            if len(FC):
                ax.plot(FC["dias"], FC["fwd_mensual"] * 100, color=VIOLETA, ls="--", lw=1.3, label="Forward a 30 días según la curva ajustada")
        ax.set_xlabel("Días desde hoy"); ax.set_ylabel(("Devaluación" if tipo_w == "dl" else "Tasa") + " mensual (%)")
        ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig)
        st.caption("Cada barra azul es la tasa que el mercado ya fija para ese período futuro; los puntos grises, la tasa de hoy hasta cada vencimiento. "
                   "Si las barras quedan arriba de los puntos, el mercado espera tasas más altas más adelante; si quedan abajo, espera que bajen.")
        mostrar(FW[["desde", "hasta", "dias", "fwd_mensual", "fwd_anual", "spot_mensual", "dif_pb"]],
                {"desde": "{:.0f}", "hasta": "{:.0f}", "dias": "{:.0f}", "fwd_mensual": "{:.2%}", "fwd_anual": "{:.1%}", "spot_mensual": "{:.2%}", "dif_pb": "{:+.0f}"},
                {"desde": "Desde (días)", "hasta": "Hasta (días)", "dias": "Días del tramo", "fwd_mensual": "Forward mensual", "fwd_anual": "Forward anual (TEA)",
                 "spot_mensual": "Spot al final del tramo", "dif_pb": "Forward − spot (pb)"})
        notas_w = {"fija": "Con LECAP y BONCAP: (1 + forward) = (pago final / precio del bono largo ÷ pago final / precio del corto) ^ (30 / días del tramo). Los BONTE con cupones no entran.",
                   "cer": "Con los bonos CER cupón cero: es la tasa real (por encima de la inflación) que el mercado fija para cada período futuro.",
                   "dl": "Con el dólar de equilibrio de cada dólar linked: es la devaluación mensual del A3500 que el mercado descuenta para cada período futuro, comparando contra la tasa fija."}
        st.caption(notas_w[tipo_w])

# ── Variación diaria contra el cierre anterior ──
with tabV:
    v1, v2, v3, v4, v5 = st.columns([1, 1.3, 1.2, 1, 1])
    grupo = v1.selectbox("Grupo", list(M.GRUPOS), format_func=M.GRUPOS.get)
    vista = v2.radio("Vista", ["Curva hoy vs ayer", "Comparativa"], horizontal=True)
    tipo_curva = v3.selectbox("Curva", ["auto", "log", "sqrt", "lin", "ns"], disabled=vista != "Curva hoy vs ayer",
                              format_func={"auto": "Automática (mejor ajuste)", "log": "Logarítmica", "sqrt": "Raíz cuadrada", "lin": "Lineal", "ns": "Nelson-Siegel"}.get)
    plazo_v = v4.selectbox("Plazo máximo", [0, 365, 730, 1095], format_func=lambda v: "Todos" if not v else f"{v // 365} año{'s' if v > 365 else ''}", key="plazo_var")
    if st.session_state.get("_grupo_var") != grupo:  # por defecto los duales entran en la curva TAMAR y no en la CER, como en Bonerío
        st.session_state["duales_var"] = grupo == "tamar"; st.session_state["_grupo_var"] = grupo
    duales_v = v5.checkbox("Incluir duales en la curva", key="duales_var", disabled=grupo not in ("tamar", "cer")) and grupo in ("tamar", "cer")
    VD, info = variacion_mercado(en_vivo, grupo, tamar_sup, infl_sup, duales_v)
    if plazo_v:
        VD = VD[VD["dias"] <= plazo_v]
    if VD.empty:
        st.info("Sin datos para ese grupo.")
    else:
        rv = M.resumen_variacion(VD)
        st.caption(f"Hoy contra el cierre anterior · {len(VD)} bonos · liquidación {info['liq']:%d/%m/%Y} · en {info['unidad']} · paneles: {PAN['fuente']}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Suben / bajan", f"{rv['suben']} / {rv['bajan']}", f"{rv['igual']} sin cambio", delta_color="off")
        k2.metric("Variación media", f"{rv['media']:+.2%}", "promedio simple de precios", delta_color="off")
        k3.metric(f"{info['metrica']} (mediana)", f"{rv['mediana_pb']:+.0f} pb", "cambio de rendimiento", delta_color="off")
        k4.metric("Mayor suba", rv["max"], f"{rv['max_v']:+.2%}")
        k5.metric("Mayor baja", rv["min"], f"{rv['min_v']:+.2%}")
        if vista == "Curva hoy vs ayer":
            fig, ax = plt.subplots(figsize=(11, 4.8))
            col = [VERDE if c < -0.5 else ROJO if c > 0.5 else GRIS for c in VD["cambio_pb"]]
            grupos = [("local", "^", VIOLETA, "ley local"), ("ny", "o", AZUL, "ley NY")] if grupo == "hd" else [("a", "o", AZUL, "")]
            for g, mk, cc, lab in grupos:
                sub = VD[VD["grupo"] == g]
                if sub.empty:
                    continue
                cols_g = [c for c, gg in zip(col, VD["grupo"]) if gg == g]
                mks = ["^" if du else mk for du in sub["dual"]]
                for (x, y, y0, c, m_) in zip(sub["x"], sub["y"], sub["y0"], cols_g, mks):
                    ax.plot([x, x], [y0 * 100, y * 100], ls=":", color=GRIS, lw=1)
                    ax.scatter([x], [y0 * 100], facecolors="none", edgecolors=GRIS, s=26, marker=m_, zorder=2)
                    ax.scatter([x], [y * 100], color=c, s=40, marker=m_, zorder=3)
                if len(sub) >= 2:
                    f1 = M.ajustar_curva(sub["x"], sub["y"], tipo_curva, info["escala"]); f0 = M.ajustar_curva(sub["x"], sub["y0"], tipo_curva, info["escala"])
                    g_ = np.geomspace(sub["x"].min(), sub["x"].max(), 120) if info["xlog"] else np.linspace(sub["x"].min(), sub["x"].max(), 120)
                    if f1:
                        ax.plot(g_, f1[0](g_) * 100, color=cc, lw=1.8, label=f"Curva hoy {lab}".strip() + f" ({f1[1].lower()})")
                    if f0:
                        ax.plot(g_, f0[0](g_) * 100, color=cc, lw=1.2, ls="--", alpha=.8, label=f"Curva ayer {lab}".strip())
            etiquetar(ax, VD["x"], VD["y"] * 100, [t + (" *" if du else "") for t, du in zip(VD.index, VD["dual"])], [pb(c) for c in VD["cambio_pb"]])
            if info["xlog"]:
                ejes_log_dias(ax, VD["x"])
            if grupo == "tamar":
                ax.axhline(0, color=GRIS, lw=.8, ls="--")
            ax.set_xlabel(info["xlab"]); ax.set_ylabel(info["ylab"]); ax.grid(alpha=.25); ax.legend(fontsize=8)
            fig.tight_layout(); st.pyplot(fig)
            st.caption("Verde: la tasa bajó (el precio subió); rojo: la tasa subió; círculo hueco: rendimiento al cierre anterior. "
                       "El rendimiento de ayer se calcula con el cierre anterior y la liquidación del día hábil previo, así que también refleja el paso del tiempo."
                       + (f" Margen s/ TAMAR con la TAMAR supuesta de {info['tamar_sup']:.2f}%." if grupo == "tamar" else "") + (" * = dual." if VD["dual"].any() else ""))
        else:
            Rv = VD.sort_values("x")
            fig, ax = plt.subplots(figsize=(10, 0.32 * len(Rv) + 1))
            ax.barh(range(len(Rv)), Rv["var"] * 100, color=[VERDE if v > 5e-5 else ROJO if v < -5e-5 else GRIS for v in Rv["var"]])
            ax.set_yticks(range(len(Rv)), Rv.index, fontsize=8); ax.invert_yaxis(); ax.axvline(0, color="black", lw=.8)
            for i, (v, c) in enumerate(zip(Rv["var"], Rv["cambio_pb"])):
                ax.annotate(f"{v:+.2%} · {pb(c)}", (v * 100, i), xytext=(4 if v >= 0 else -4, -3), textcoords="offset points", fontsize=7, ha="left" if v >= 0 else "right")
            ax.set_xlabel("Variación de precio contra el cierre anterior (%)"); fig.tight_layout(); st.pyplot(fig)
            st.caption("Ordenados por plazo. Al lado de cada barra, el cambio de rendimiento en puntos básicos.")
        es_md = grupo == "hd"
        mostrar(VD[["vence", "x", "precio", "p0", "var", "y", "y0", "cambio_pb"]],
                {"vence": "{:%d/%m/%Y}", "x": "{:.2f}" if es_md else "{:.0f}", "precio": "{:,.2f}", "p0": "{:,.2f}", "var": "{:+.2%}", "y": "{:.2%}", "y0": "{:.2%}", "cambio_pb": "{:+.0f}"},
                {"vence": "Vence", "x": "MD" if es_md else "Días", "precio": "Precio", "p0": "Cierre ant.", "var": "Var.", "y": f"{info['metrica']} hoy",
                 "y0": f"{info['metrica']} ayer", "cambio_pb": "Cambio (pb)"}, calor={"var": "var", "cambio_pb": "pb_rend"})

# ── Volumen operado ──
with tabO:
    o1, o2, o3, o4 = st.columns([1, 1.2, 1, 1.4])
    per = o1.selectbox("Período", [5, 20, 60], index=1, format_func={5: "5 ruedas (≈ 1 semana)", 20: "20 ruedas (≈ 1 mes)", 60: "60 ruedas (≈ 3 meses)"}.get)
    fam_sel = o2.selectbox("Familia", ["all"] + list(M.FAMILIAS), format_func=lambda f: "Todas" if f == "all" else M.FAMILIAS[f])
    agrupar = o3.radio("Agrupar", ["Por especie", "Por familia"], horizontal=True)
    if o4.button("Bajar el histórico actualizado de BYMA (≈ 1 minuto)", disabled=not en_vivo):
        st.session_state["vol_vivo"] = True
    V = cargar_volumen(bool(st.session_state.get("vol_vivo")) and en_vivo)
    ult_vol = M.fecha_rueda(V["d"][-1])
    hay_hoy = PAN["fecha"] > ult_vol
    incluir_hoy = st.checkbox("Incluir la rueda de hoy (parcial)", value=False, disabled=not hay_hoy,
                              help="Suma el monto del panel de hoy. Sólo se habilita si el panel es de una rueda posterior a la última del histórico.")
    esp, fechas_v, con_hoy = M.volumen_periodo(V, per, PAN if (incluir_hoy and hay_hoy) else None)
    st.caption("1 billón = un millón de millones de pesos. "
               f"BYMA, segmento 24 hs · {per} ruedas: {fechas_v[0]:%d/%m/%Y} al {fechas_v[-1]:%d/%m/%Y}{' (hoy parcial)' if con_hoy else ''} · histórico: {V['fuente']} · "
               "montos en pesos; dólar MEP y cable convertidos al tipo de cambio implícito de cada día; sin los alias X (mismo ISIN, otro segmento)")
    sel = esp if fam_sel == "all" else esp[esp["familia"] == fam_sel]

    st.subheader("Volumen contra precio")
    st.caption("Cada punto es un bono: hacia la derecha subió, hacia la izquierda bajó; más arriba tuvo más volumen que su rutina "
               "(1 = el promedio diario de las 60 ruedas anteriores al período). El tamaño es el monto operado.")
    if agrupar == "Por familia":
        filas = []
        for f in [f for f in M.FAMILIAS if fam_sel in ("all", f)]:
            g = esp[(esp["familia"] == f) & (esp["monto"] > 0)]
            if g.empty:
                continue
            w = g[np.isfinite(g["var"])]; r_ = g[np.isfinite(g["relativo"])]
            filas.append({"ticker": M.FAMILIAS[f], "familia": f, "monto": g["monto"].sum(),
                          "var": (w["var"] * w["monto"]).sum() / w["monto"].sum() if w["monto"].sum() else np.nan,
                          "relativo": r_["monto"].sum() / (r_["monto"] / r_["relativo"]).sum() if len(r_) else np.nan})
        pts = pd.DataFrame(filas).set_index("ticker")
    else:
        pts = sel[sel["monto"] > 0]
    pts = pts[np.isfinite(pts["var"]) & np.isfinite(pts["relativo"])]
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(pts):
        tam = 20 + 900 * np.sqrt(pts["monto"] / pts["monto"].max())
        ax.scatter(pts["var"] * 100, pts["relativo"].clip(lower=0.02), s=tam, c=[M.COLORES[f] for f in pts["familia"]], alpha=.7, edgecolors="white", lw=.6)
        top = set(pts.sort_values("monto", ascending=False).index[:18]) | set(pts.sort_values("relativo", ascending=False).index[:6])
        for t, r in pts.iterrows():
            if t in top:
                ax.annotate(t, (r["var"] * 100, max(r["relativo"], 0.02)), xytext=(5, 2), textcoords="offset points", fontsize=7)
    ax.set_yscale("log"); ax.axhline(1, color=GRIS, ls="--", lw=.8); ax.axvline(0, color=GRIS, ls="--", lw=.8)
    for (xx, yy, ha, va, txt) in [(.01, .98, "left", "top", "Bajó con volumen"), (.99, .98, "right", "top", "Subió con volumen"),
                                  (.01, .02, "left", "bottom", "Bajó sin volumen"), (.99, .02, "right", "bottom", "Subió sin volumen")]:
        ax.text(xx, yy, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=8, color=GRIS)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("Variación del precio en el período (%)"); ax.set_ylabel("Volumen relativo (× lo habitual)"); ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=M.COLORES[f], label=M.FAMILIAS[f]) for f in M.FAMILIAS if fam_sel in ("all", f)],
              fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.13), frameon=False)
    fig.tight_layout(); st.pyplot(fig)

    st.subheader("Dónde estuvo el volumen · por rueda")
    fam_tot = esp.groupby("familia")[["monto", "monto_previo"]].sum().reindex(list(M.FAMILIAS)).fillna(0)
    tot, tot_prev = sel["monto"].sum(), sel["monto_previo"].sum()
    rank = sel.sort_values("monto", ascending=False)
    lider = fam_tot["monto"].idxmax()
    corto = lambda v: ("$ " + f"{v / 1e6:.2f}".replace(".", ",") + " bill.") if v >= 1e6 else ("$ " + f"{v / 1e3:,.0f}".replace(",", ".") + " mil M")
    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Total del período", corto(tot), f"{tot / tot_prev - 1:+.0%} vs. las {per} ruedas anteriores" if tot_prev else None, delta_color="off")
    w2.metric("Promedio por rueda", corto(tot / per), f"{fechas_v[0]:%d/%m} al {fechas_v[-1]:%d/%m}", delta_color="off")
    w3.metric("Familia líder", M.CORTO[lider], f"{fam_tot.loc[lider, 'monto'] / fam_tot['monto'].sum():.0%} del total", delta_color="off")
    w4.metric("Más operado", rank.index[0], f"{rank['monto'].iloc[0] / tot:.1%} del total", delta_color="off")
    w5.metric("Concentración", f"{rank['monto'].head(5).sum() / tot:.0%}", "de los 5 primeros bonos", delta_color="off")
    g1, g2 = st.columns([1.5, 1])
    with g1:
        fams = list(M.FAMILIAS) if fam_sel == "all" else [fam_sel]
        fig, ax = plt.subplots(figsize=(8, 4))
        abajo = np.zeros(len(fechas_v))
        for f in fams:
            sub = esp[esp["familia"] == f]
            vals = np.array([sum(r[i] for r in sub["por_rueda"]) for i in range(len(fechas_v))], float) / 1e6
            ax.bar(range(len(fechas_v)), vals, bottom=abajo, color=M.COLORES[f], label=M.FAMILIAS[f], width=.8); abajo += vals
        paso = max(1, len(fechas_v) // 10)
        ax.set_xticks(range(0, len(fechas_v), paso), [f"{d:%d/%m}" for d in fechas_v[::paso]], fontsize=8)
        ax.set_ylabel("Billones de pesos"); ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.18), frameon=False); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); st.pyplot(fig)
    with g2:
        ft = fam_tot.copy()
        ft["pct"] = ft["monto"] / ft["monto"].sum(); ft["pct_prev"] = ft["monto_previo"] / ft["monto_previo"].sum(); ft["dpp"] = (ft["pct"] - ft["pct_prev"]) * 100
        ft.index = [M.CORTO[f] for f in ft.index]
        ft["monto_t"] = [corto(v) for v in ft["monto"]]
        mostrar(ft[["monto_t", "pct", "pct_prev", "dpp"]], {"pct": "{:.1%}", "pct_prev": "{:.1%}", "dpp": "{:+.1f}"},
                {"monto_t": "Monto", "pct": "% total", "pct_prev": "% previo", "dpp": "Δ pp"})
    st.subheader("Ranking por especie")
    rk = rank.head(40).copy()
    rk["familia"] = [M.FAMILIAS[f] for f in rk["familia"]]
    rk["pct"] = rk["monto"] / tot; rk["prom"] = rk["monto"] / per
    mostrar(rk[["familia", "monto", "pct", "prom", "relativo", "var"]],
            {"monto": "{:,.0f}", "pct": "{:.1%}", "prom": "{:,.0f}", "relativo": "{:.2f}×", "var": "{:+.2%}"},
            {"familia": "Familia", "monto": "Monto ($M)", "pct": "% total", "prom": "Promedio por rueda ($M)", "relativo": "Vol. relativo", "var": "Var. precio"}, na="nuevo",
            calor={"var": "var_periodo"})

# ── Flujos de fondos: todos los bonos ──
def num_es(v, d=2):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


with tabF:
    USDU = B.universo_usd()
    DL_TK = {t for t, _, _ in P.DL} | {"TMVE8"}
    pesos_tk = [t for k, _, _ in P.TIPOS if len(RP[k]) for t in RP[k].index]
    tipo_de = {t: nombre for k, nombre, _ in P.TIPOS if len(RP[k]) for t in RP[k].index}
    f1, f2, f3 = st.columns([1.6, 1, 1])
    bono_f = f1.selectbox("Bono", list(tabla_usd.index) + pesos_tk,
                          format_func=lambda t: f"{t} · {tabla_usd.loc[t, 'familia']} (USD)" if t in tabla_usd.index else f"{t} · {tipo_de[t]} ($)")
    es_usd = bono_f in USDU
    vn_usd = es_usd or bono_f in DL_TK  # el valor nominal está en dólares
    k_tipo = None if es_usd else next(k for k, _, _ in P.TIPOS if len(RP[k]) and bono_f in RP[k].index)
    px_def = float(tabla_usd.loc[bono_f, "precio"]) if es_usd else float(RP[k_tipo].loc[bono_f, "precio"])
    # claves fijas para los widgets; al cambiar de bono (o de moneda) se reponen los valores por defecto
    if st.session_state.get("_flujos_bono") != (bono_f, moneda):
        if st.session_state.get("_flujos_vn_usd") != vn_usd:
            st.session_state["vn_flujos"] = 1000 if vn_usd else 100000
        st.session_state["px_flujos"] = round(px_def, 2)
        st.session_state["_flujos_bono"], st.session_state["_flujos_vn_usd"] = (bono_f, moneda), vn_usd
    vn_f = f2.number_input("Nominales", min_value=1, step=100, key="vn_flujos")
    px_f = f3.number_input("Precio de compra", min_value=0.01, step=0.1, key="px_flujos")
    if es_usd:
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
        fig.tight_layout(); st.pyplot(fig)
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
        fig.tight_layout(); st.pyplot(fig)
        T_ = F.set_index("fecha")
        for c in ["renta", "capital", "total", "vp"] + [c for c in ("real", "usd") if c in T_]:
            T_[c] = T_[c] * k
        cols_t = ["dias", "renta", "capital", "total"] + [c for c in ("real", "usd") if c in T_] + ["vp"]
        mostrar(T_[cols_t], {"dias": "{:.0f}", "renta": "{:,.2f}", "capital": "{:,.2f}", "total": "{:,.2f}", "real": "{:,.2f}", "usd": "{:,.2f}", "vp": "{:,.2f}"},
                {"dias": "Días", "renta": "Renta / interés", "capital": "Capital", "total": "Total proyectado", "real": "En pesos de hoy", "usd": "En dólares",
                 "vp": "Valor presente"})

    st.subheader("Próximo pago de cada bono")
    grupo_f = st.radio("Mostrar", ["Todos", "En dólares", "En pesos"], horizontal=True, key="grupo_flujos")
    prox = []
    if grupo_f != "En pesos":
        for t, b in {**USDU, **L._bopreales()}.items():
            fut = [(f, c, a) for f, c, a in b["flujos"] if f > liq_usd]
            if fut:
                f, c, a = fut[0]
                prox.append({"ticker": t, "tipo": "Bopreal (USD)" if t.startswith("BPO") else b.get("familia", "") + " (USD)", "moneda": "US$",
                             "pago": L.siguiente_habil(f), "que": "Vence" if f >= b["venc"] else "Renta y amortización" if a else "Renta",
                             "renta": c, "capital": a, "quedan": len(fut), "vence": b["venc"]})
    if grupo_f != "En dólares":
        for t in pesos_tk:
            Fx = RP["flujos"].get(t)
            if Fx is None or Fx.empty:
                continue
            r0 = Fx.iloc[0]
            prox.append({"ticker": t, "tipo": tipo_de[t], "moneda": "$", "pago": r0["fecha"],
                         "que": "Vence" if len(Fx) == 1 else "Renta y amortización" if r0["capital"] else "Renta",
                         "renta": r0["renta"], "capital": r0["capital"], "quedan": len(Fx), "vence": Fx["fecha"].iloc[-1]})
    PX = pd.DataFrame(prox).sort_values(["pago", "ticker"]).set_index("ticker")
    PX["total"] = PX["renta"] + PX["capital"]
    PX["dias"] = [(d - liq_usd).days for d in PX["pago"]]
    mostrar(PX[["tipo", "pago", "dias", "que", "moneda", "renta", "capital", "total", "quedan", "vence"]],
            {"pago": "{:%d/%m/%Y}", "dias": "{:.0f}", "renta": "{:,.2f}", "capital": "{:,.2f}", "total": "{:,.2f}", "quedan": "{:.0f}", "vence": "{:%d/%m/%Y}"},
            {"tipo": "Tipo", "pago": "Próximo pago", "dias": "Días", "que": "Qué paga", "moneda": "Moneda", "renta": "Renta / interés",
             "capital": "Amortización / capital", "total": "Total", "quedan": "Pagos que quedan", "vence": "Vence"})
    st.caption("Por 100 VN originales. Bonos en dólares: pagan en dólares (fechas corridas al día hábil siguiente). Bonos en pesos: CER, TAMAR, duales y dólar linked "
               "son estimaciones con los supuestos de la pestaña Tesoro en pesos.")

# ── Dólares ──
with tabD:
    DFD = cargar_dolares(en_vivo)
    d1, d2 = st.columns([1, 2])
    ref_d = d1.radio("Bono de referencia", ["al30", "gd30"], horizontal=True, format_func=str.upper)
    per_d = d2.radio("Período", [21, 63, 126, 252, 0], index=3, horizontal=True, format_func={21: "1M", 63: "3M", 126: "6M", 252: "1A", 0: "2A"}.get)
    SD = DL.series(DFD, ref_d)
    REF = ref_d.upper()
    pan_px = lambda t: (lambda x: (x[0] or x[1] or x[2]) if x else 0)(PAN["t1"].get(t))
    mep_h = pan_px(REF) / pan_px(REF + "D") if pan_px(REF + "D") else float(SD["mep"].dropna().iloc[-1])
    ccl_h = pan_px(REF) / pan_px(REF + "C") if pan_px(REF + "C") else float(SD["cable"].dropna().iloc[-1])
    ayer = PAN["fecha"] - timedelta(days=1)
    (f_m0, mep0), (_, ccl0) = DL.valor_al(SD["mep"], ayer), DL.valor_al(SD["cable"], ayer)
    (f_a, a35), (f_mi, mino) = DL.valor_al(SD["a3500"], PAN["fecha"]), DL.valor_al(SD["minorista"], PAN["fecha"])
    a350, mino0 = DL.valor_al(SD["a3500"], f_a - timedelta(days=1))[1], DL.valor_al(SD["minorista"], f_mi - timedelta(days=1))[1]
    techo = DL.techo_banda(PAN["fecha"])
    st.caption(f"MEP y cable con {REF} (VWAP del panel de 24 hs del {PAN['fecha']:%d/%m/%Y}) · A3500 y minorista del {f_a:%d/%m/%Y} · fuentes: "
               + ", ".join(f"{k} {v}" for k, v in DFD.attrs.get("fuente", {}).items()))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Dólar MEP · {REF}", f"$ {num_es(mep_h)}", f"{mep_h / mep0 - 1:+.2%} vs {f_m0:%d/%m}")
    k2.metric(f"Dólar cable · {REF}", f"$ {num_es(ccl_h)}", f"{ccl_h / ccl0 - 1:+.2%} vs {f_m0:%d/%m}")
    k3.metric("Canje (cable / MEP)", f"{ccl_h / mep_h - 1:.2%}", f"{(ccl_h / mep_h - ccl0 / mep0) * 100:+.2f} pp", delta_color="off")
    k4.metric("A3500 (mayorista BCRA)", f"$ {num_es(a35)}", f"{a35 / a350 - 1:+.2%}")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Minorista (BCRA)", f"$ {num_es(mino)}", f"{mino / mino0 - 1:+.2%}")
    k6.metric("Brecha MEP", f"{mep_h / a35 - 1:.2%}", "MEP sobre A3500", delta_color="off")
    k7.metric("Brecha cable", f"{ccl_h / a35 - 1:.2%}", "cable sobre A3500", delta_color="off")
    k8.metric("Techo de la banda (aprox.)", f"$ {num_es(techo)}", f"A3500 {1 - a35 / techo:.1%} por debajo", delta_color="off")
    desde_d = SD["mep"].dropna().index[-per_d] if per_d else SD.index[0]
    Sv = SD[SD.index >= desde_d]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for col, nom, c, ls in [("mep", f"MEP ({REF})", AZUL, "-"), ("cable", f"Cable ({REF})", "#eb6834", "-"), ("a3500", "A3500 (BCRA)", "#1baf7a", "-"),
                            ("minorista", "Minorista (BCRA)", VIOLETA, "--")]:
        x = Sv[col].dropna()
        ax.plot(x.index, x.values, color=c, ls=ls, lw=1.6 if ls == "-" else 1.1, label=nom)
    ax.set_ylabel("$ por dólar"); ax.legend(fontsize=8, ncol=4); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    for col, nom, c in [("canje", "Canje (cable / MEP)", "black"), ("brecha_mep", "Brecha MEP vs A3500", AZUL), ("brecha_cable", "Brecha cable vs A3500", "#eb6834")]:
        x = Sv[col].dropna()
        ax.plot(x.index, x.values * 100, color=c, lw=1.5 if col == "canje" else 1.1, label=nom)
    ax.axhline(0, color=GRIS, ls="--", lw=.8); ax.set_ylabel("%"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
    st.caption("Canje = cable / MEP − 1: lo que cuesta pasar dólares del país al exterior. Brecha = dólar financiero / A3500 − 1.")
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Variaciones")
        VA = DL.variaciones(SD, PAN["fecha"])
        HZ = ["1 día", "1 semana", "1 mes", "3 meses", "En el año", "12 meses"]
        niv = VA[VA["tipo"] == "nivel"]
        mostrar(niv[["ultimo"] + HZ + ["min_12m", "max_12m"]], {"ultimo": "$ {:,.2f}", **{h: "{:+.1%}" for h in HZ}, "min_12m": "$ {:,.0f}", "max_12m": "$ {:,.0f}"},
                {"ultimo": "Último", "min_12m": "Mín. 12 m", "max_12m": "Máx. 12 m"}, calor={h: "var_periodo" for h in HZ})
        spr = VA[VA["tipo"] == "spread"].copy()
        for h in HZ:
            spr[h] = spr[h] * 100
        mostrar(spr[["ultimo"] + HZ + ["min_12m", "max_12m"]], {"ultimo": "{:.2%}", **{h: "{:+.2f} pp" for h in HZ}, "min_12m": "{:.1%}", "max_12m": "{:.1%}"},
                {"ultimo": "Último", "min_12m": "Mín. 12 m", "max_12m": "Máx. 12 m"}, calor={h: "pp" for h in HZ})
    with v2:
        st.subheader("Dólar implícito en cada bono")
        DB = DL.dolar_por_bono(PAN, M.USD_LOCAL + M.USD_NY + M.BOPREAL)
        mostrar(DB, {"mep": "$ {:,.2f}", "cable": "$ {:,.2f}", "canje": "{:.2%}", "vs_mep_al30": "{:+.2%}", "monto_d": "{:,.2f}", "monto_c": "{:,.2f}"},
                {"mep": "MEP", "cable": "Cable", "canje": "Canje", "vs_mep_al30": "vs MEP AL30", "monto_d": "Monto D (US$ M)", "monto_c": "Monto C (US$ M)"})
        st.caption("MEP de un bono = precio en pesos / precio D; cable = pesos / precio C. Negativo contra el AL30 = dólares más baratos con ese bono.")
    st.subheader("Qué dólar oficial descuenta el mercado")
    dl_ = RP["dl"][RP["dl"]["dias"] >= 15]
    mostrar(dl_[["vence", "dias", "dolar_be", "deval_mensual", "deval_total"]],
            {"vence": "{:%d/%m/%Y}", "dias": "{:.0f}", "dolar_be": "$ {:,.2f}", "deval_mensual": "{:.2%}", "deval_total": "{:.1%}"},
            {"vence": "Vence", "dias": "Días", "dolar_be": "A3500 de equilibrio", "deval_mensual": "Deval. mensual", "deval_total": "Deval. total"})
    st.caption(f"A3500 al vencimiento que hace rendir igual a cada bono dólar linked y a la tasa fija del mismo plazo. A3500 hoy: $ {num_es(RP['tc_hoy'])}. "
               "No es un pronóstico.")

# ── Licitaciones del Tesoro y calendario ──
with tabL:
    SUB, COL = L.resultados()
    PR = L.proxima()
    st.subheader(f"Próxima licitación: {pd.Timestamp(PR['lic']):%d/%m/%Y}")
    st.caption(f"Llamado del {pd.Timestamp(PR['llamado']):%d/%m/%Y} · ofertas de {PR['hora']} hs · liquida el {pd.Timestamp(PR['liq']):%d/%m/%Y} · {PR['nota']}")
    menu = []
    for t, nom, paga, tipo, venc in PR["menu"]:
        k_ = next((k for k, _, _ in P.TIPOS if len(RP[k]) and t in RP[k].index), None)
        r_ = RP[k_].loc[t] if k_ else None
        if r_ is None:
            otro = ""
        elif k_ in ("lecap", "boncap"):
            otro = f"TEM {r_['tem']:.2%}"
        elif k_ == "tamar":
            otro = f"margen s/ TAMAR {r_['margen_s_tamar']:.2f}%"
        else:
            otro = f"piso {r_['piso']:.2%} {r_['piso_tipo']} · paga {r_['paga']}"
        menu.append({"ticker": t, "instrumento": f"{nom} ({tipo})", "paga": paga, "vence": venc, "precio": r_["precio"] if r_ is not None else np.nan,
                     "tirea": r_["tirea"] if r_ is not None and "tirea" in r_ else np.nan, "otro": otro})
    mostrar(pd.DataFrame(menu).set_index("ticker"), {"precio": "{:,.2f}", "tirea": "{:.2%}"},
            {"instrumento": "Instrumento", "paga": "Paga", "vence": "Vence", "precio": "Precio hoy", "tirea": "TIREA hoy", "otro": "Otro dato"})
    st.subheader("¿Las licitaciones pagan premio?")
    l1, l2, l3 = st.columns(3)
    fam_l = l1.selectbox("Tipo de instrumento", ["all"] + list(L.FAMILIAS), format_func=lambda f: "Todos" if f == "all" else L.FAMILIAS[f])
    med_l = l2.radio("Medir el premio en", ["pb", "px"], horizontal=True, format_func={"pb": "Puntos de TIREA", "px": "% del precio"}.get)
    solo_reap = l3.checkbox("Solo reaperturas", value=False)
    C_ = COL if fam_l == "all" else COL[COL["familia"] == fam_l]
    if solo_reap:
        C_ = C_[C_["nueva"] == 0]
    RS = L.resumen(C_, med_l)
    u_ = (lambda v: f"{v:+.0f} pb") if med_l == "pb" else (lambda v: f"{v:+.2f}%")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Premio mediano", u_(RS["mediana"]), "TIREA de corte − secundario" if med_l == "pb" else "ganancia inmediata", delta_color="off")
    m2.metric("Salieron con premio", f"{RS['con_premio']} de {RS['n']}", f"{RS['con_premio'] / max(1, RS['n']):.0%}", delta_color="off")
    m3.metric("Ganaban el día de liquidación", f"{RS['ganan']} de {RS['n_post']}", f"mediana {RS['mediana_post']:+.2%}", delta_color="off")
    m4.metric("Adjudicado en 2026", f"$ {num_es(SUB['adj'].sum(), 1)} bill.", "valor efectivo", delta_color="off")
    ult_s = SUB.iloc[-1]
    m5.metric("Última licitación", f"{ult_s['f']:%d/%m/%Y}", f"{num_es(ult_s['adj'])} de {num_es(ult_s['ofer'])} billones ofertados", delta_color="off")
    Cg = C_.dropna(subset=["premio_pb"])
    valor = lambda x: x["premio_pb"] if med_l == "pb" else x["premio_precio"] * 100
    if len(Cg):
        lo_, hi_ = np.nanpercentile(valor(Cg), 2), np.nanpercentile(valor(Cg), 98)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        for f, nom in L.FAMILIAS.items():
            x = Cg[Cg["familia"] == f]
            if x.empty:
                continue
            v = valor(x).clip(min(lo_, 0), max(hi_, 0))
            ax.scatter(pd.to_datetime(x["fecha"]), v, s=30, edgecolors=L.COLORES[f], facecolors=[L.COLORES[f] if n == 0 else "none" for n in x["nueva"]],
                       label=nom, zorder=3)
        medl = Cg.groupby("fecha").apply(lambda g: valor(g).median())
        ax.plot(pd.to_datetime(medl.index), medl.values, color="black", lw=1, alpha=.6, label="Mediana de cada licitación")
        ax.axhline(0, color=GRIS, ls="--", lw=.8); ax.set_ylabel("Premio (pb)" if med_l == "pb" else "Premio (% del precio)")
        ax.legend(fontsize=7, ncol=4); ax.grid(alpha=.2); fig.tight_layout(); st.pyplot(fig)
    st.caption("Premio = TIREA de corte menos la TIREA del secundario al cierre del día de la licitación (BYMA, 24 hs), las dos con el mismo método. Positivo: el Tesoro "
               "pagó más que el mercado. Huecos: instrumentos nuevos, comparados contra la curva de su familia. Valores extremos recortados al percentil 2-98.")
    T_ = C_.sort_values(["fecha", "familia"], ascending=[False, True]).copy()
    T_["tipo"] = [L.FAMILIAS[f] + (" · nueva" if n else "") for f, n in zip(T_["familia"], T_["nueva"])]
    T_["adj_of"] = T_["vno_adjudicado"] / T_["vno_ofertado"]
    T_ = T_.set_index("ticker")
    mostrar(T_[["fecha", "tipo", "valor_efectivo", "adj_of", "precio", "tirea_corte", "tirea_secundario", "premio_pb", "premio_precio", "tirea_al_liquidar",
                "resultado_al_liquidar"]],
            {"fecha": "{:%d/%m/%Y}", "valor_efectivo": "{:,.0f}", "adj_of": "{:.0%}", "precio": "{:,.2f}", "tirea_corte": "{:.2%}", "tirea_secundario": "{:.2%}",
             "premio_pb": "{:+.0f}", "premio_precio": "{:+.2%}", "tirea_al_liquidar": "{:.2%}", "resultado_al_liquidar": "{:+.2%}"},
            {"fecha": "Licitación", "tipo": "Tipo", "valor_efectivo": "Adjudicado (millones)", "adj_of": "Adj. / ofert.", "precio": "Precio de corte",
             "tirea_corte": "TIREA de corte", "tirea_secundario": "TIREA secundario", "premio_pb": "Premio (pb)", "premio_precio": "Premio (% precio)",
             "tirea_al_liquidar": "TIREA al liquidar", "resultado_al_liquidar": "Resultado al liquidar"},
            calor={"premio_pb": "premio_pb", "premio_precio": "pct", "resultado_al_liquidar": "pct"})
    st.caption("Fuente: comunicados «Resultado de la licitación» de la Secretaría de Finanzas y planilla «Colocaciones de deuda» al 31/08/2026; "
               "secundario: BYMA, cierres de 24 hs. Montos en millones de pesos (de dólares en los BONAR). Datos hasta la licitación del 28/09/2026.")

    st.subheader("Calendario: licitaciones, cortes de cupón y vencimientos")
    hoy_c = max(date.today(), D["fecha"])
    meses = pd.period_range(pd.Timestamp(hoy_c).to_period("M") - 1, periods=16, freq="M")
    MESES_L = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    c1, c2, c3 = st.columns([1, 2, 1.2])
    mes_c = c1.selectbox("Mes", meses, index=1, format_func=lambda p: f"{MESES_L[p.month - 1]} {p.year}")
    tipos_c = c2.multiselect("Qué mostrar", ["lic", "cor", "ren", "ven", "fer"], default=["lic", "cor", "ren", "ven", "fer"],
                             format_func={"lic": "Licitaciones", "cor": "Cortes de cupón", "ren": "Pagos de renta", "ven": "Amortizaciones y vencimientos",
                                          "fer": "Feriados"}.get)
    mon_c = c3.multiselect("Bonos", ["usd", "ars"], default=["usd", "ars"], format_func={"usd": "En dólares", "ars": "En pesos"}.get)
    EV = L.eventos(date(2026, 1, 1), date(2028, 12, 31))
    grupo_t = {"lla": "lic", "lic": "lic", "liq": "lic", "cor": "cor", "ren": "ren", "amo": "ven", "ven": "ven", "fer": "fer"}
    EVf = EV[EV["tipo"].map(grupo_t).isin(tipos_c) & (EV["moneda"].isin(mon_c) | (EV["moneda"] == "") | EV["tipo"].isin(["lla", "lic", "liq"]))]
    st.markdown(L.mes_html(EVf, mes_c.year, mes_c.month, hoy_c), unsafe_allow_html=True)
    st.caption(L.RESPALDO["calendario"]["corte"] + " Las fechas que caen en feriado o fin de semana pasan al día hábil siguiente. Cronograma: Secretaría de "
               "Finanzas; feriados: BCRA (incluye el 9 y 10/11 por la visita del papa).")
    st.subheader("Agenda de las próximas semanas")
    AG = L.agrupar(EVf[(EVf["fecha"] >= hoy_c) & (EVf["fecha"] <= hoy_c + timedelta(days=45))])
    AG["tipo"] = AG["tipo"].map(L.TIPOS_EV)
    mostrar(AG.set_index("fecha"), {}, {"tipo": "Qué pasa", "ticker": "Bonos", "detalle": "Detalle"})

# ── 2. inflación breakeven (despejada, como Bonerío Local y Chamomberg) ──
def grafico_trayectoria(T):
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    x = list(range(len(T)))
    pub = [i for i, f in enumerate(T["fuente"]) if f.startswith("INDEC")]
    bes = [i for i, f in enumerate(T["fuente"]) if not f.startswith("INDEC")]
    if pub:
        ax.plot(pub, T["inflacion"].iloc[pub] * 100, color=VERDE, lw=2, marker="o", label="IPC publicado (INDEC)")
    if bes:
        xs = ([pub[-1]] if pub else []) + bes
        ax.plot(xs, T["inflacion"].iloc[xs] * 100, color=VIOLETA, lw=2, ls="--", marker="D", ms=5, label="Breakeven despejado (mercado)")
    rem = T["rem"] * 100
    if rem.notna().any():
        ax.plot(x, rem, color=GRIS, lw=1.4, ls=":", marker=".", label="REM (mediana de analistas)")
    for i in bes:
        ax.annotate(f"{T['inflacion'].iloc[i]:.2%}", (i, T["inflacion"].iloc[i] * 100), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=VIOLETA)
    ax.set_xticks(x, [B.mes_corto(m) for m in T.index]); ax.set_ylabel("Inflación mensual %")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    return fig


with tab2:
    st.caption("La inflación breakeven es la que hace que un bono CER pague lo mismo que uno a tasa fija al mismo plazo. El CER de los próximos días "
               "ya está publicado (crece con el IPC de dos meses antes), así que la inflación que hay que «despejar» es solo la de los meses que el INDEC "
               "todavía no publicó. Es el mismo método de Bonerío Local (inflación de cada bono) y de Chamomberg (trayectoria mes a mes).")
    b1, b2, b3, b4 = st.columns([1.35, 1.6, 1.6, 0.9])
    modelo_be = b1.selectbox("Curva de tasa fija", list(P.MODELOS_CURVA), format_func=P.MODELOS_CURVA.get,
                             help="Contra qué tasa fija se compara cada CER. «Automática» ajusta lineal, logarítmica, raíz cuadrada, cuadrática y "
                                  "Nelson-Siegel a las LECAP y BONCAP y se queda con la de menor AIC (buen ajuste sin exceso de parámetros).")
    anclas_be = b2.radio("Bonos que anclan la trayectoria", ["mes_completo", "todos"],
                         format_func={"mes_completo": "Los que cierran un mes de IPC", "todos": "Todos los CER cupón cero"}.get,
                         help="«Los que cierran un mes»: solo los CER cuya fecha de referencia cae el día 15, así cada tramo de meses queda exactamente "
                              "identificado (como Chamomberg). «Todos»: más puntos, pero más ruido.")
    contra_be = b3.radio("Contra qué tasa fija se despeja", ["curva", "par"],
                         format_func={"curva": "La curva ajustada", "par": "La LECAP/BONCAP del mismo vencimiento"}.get)
    min_be = b4.number_input("Meses por tramo", 1, 6, 3, disabled=anclas_be != "todos",
                             help="Con «Todos»: un bono despeja su tramo solo si trae al menos estos meses nuevos; si no, sus meses pasan al bono siguiente.")
    RB = calcular_pesos(en_vivo, plazo_p, precio_p, tamar_sup, infl_sup, dev_sup, modelo_be)
    cer_b = RB["cer"]
    beb = cer_b[np.isfinite(cer_b["be_mensual"].astype(float))]
    T = P.trayectoria_be(RB, anclas_be, contra_be, int(min_be))
    if len(beb):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Breakeven del CER más corto", f"{beb['be_mensual'].iloc[0]:.2%}", f"{beb.index[0]} · REM {beb['rem_mensual'].iloc[0]:.2%}", delta_color="off")
        k2.metric("Breakeven del CER más largo", f"{beb['be_mensual'].iloc[-1]:.2%}", f"{beb.index[-1]} · REM {beb['rem_mensual'].iloc[-1]:.2%}", delta_color="off")
        k3.metric("Mercado − REM (mediana)", f"{beb['be_rem_pb'].median():+.0f} pb", "por mes", delta_color="off",
                  help="Positivo: el mercado espera más inflación que los analistas del REM para esos mismos meses.")
        k4.metric("Curva de tasa fija", RB["curva_be_nombre"], f"error medio {RB['curva_be_error_pb']:.0f} pb" if RB["curva_be_error_pb"] else "pasa por cada bono", delta_color="off")

    st.subheader("Inflación breakeven de cada bono CER")
    t = cer_b[cer_b["dias"] <= max(400, RB["max_cero"] + 100)].copy()
    t["be_t"] = [f"{v:.2%}" if np.isfinite(v) else ("n/a" if n == "CER ya publicado" else "—") for v, n in zip(t["be_mensual"], t["be_nota"])]
    t["be_anual"] = (1 + t["be_mensual"].astype(float)) ** 12 - 1
    t["par_t"] = [f"{p_} · {v:.2%}" if isinstance(p_, str) and np.isfinite(v) else "–" for p_, v in zip(t.get("par", pd.Series(index=t.index)), t.get("be_par", pd.Series(np.nan, index=t.index)))]
    mostrar(t[["vence", "dias", "precio", "tir_real", "tna_real", "fija_be", "be_t", "be_anual", "rem_mensual", "be_rem_pb", "par_t"]],
            {"vence": "{:%d/%m/%Y}", "dias": "{:.0f}", "precio": "{:,.2f}", "tir_real": "{:.2%}", "tna_real": "{:.2%}", "fija_be": "{:.2%}",
             "be_anual": "{:.1%}", "rem_mensual": "{:.2%}", "be_rem_pb": "{:+.0f}"},
            {"vence": "Vence", "dias": "Días", "precio": "Precio", "tir_real": "TIR real (TEA)", "tna_real": "TNA real 30/365", "fija_be": "Tasa fija al plazo (TIREA)",
             "be_t": "Inflación BE mensual", "be_anual": "BE anual", "rem_mensual": "REM mensual", "be_rem_pb": "BE − REM (pb)", "par_t": "Con la letra del mismo vencimiento"})
    st.caption("Inflación BE: inflación mensual pareja de los meses sin IPC publicado que iguala lo que paga el CER con la tasa fija de la curva a su plazo. "
               "n/a: el CER hasta el vencimiento ya está publicado; —: vence después de la LECAP/BONCAP más larga. REM: promedio del sendero del REM para "
               "esos mismos meses. La última columna repite el cálculo contra el precio de la letra con igual vencimiento (S30N6, S29E7, T30A7, T31Y7, T30J7).")
    if len(beb):
        st.pyplot(grafico_be(cer_b, RB["max_cero"]))

    st.subheader("Trayectoria de inflación mensual que descuenta el mercado")
    if T.empty or not (T["fuente"] == "Breakeven despejado").any():
        st.info("No hay bonos CER suficientes para despejar la trayectoria con esas opciones.")
    else:
        st.pyplot(grafico_trayectoria(T))
        T2 = T.copy(); T2.index = [B.mes_corto(m) for m in T2.index]
        mostrar(T2[["inflacion", "tea", "fuente", "bono", "rem", "dif_rem_pb"]],
                {"inflacion": "{:.2%}", "tea": "{:.1%}", "rem": "{:.2%}", "dif_rem_pb": "{:+.0f}"},
                {"inflacion": "π mensual", "tea": "TEA", "fuente": "Fuente", "bono": "Bono que lo define", "rem": "REM", "dif_rem_pb": "Mercado − REM (pb)"})
        st.caption("Se ordenan los CER por su fecha de referencia (vencimiento menos 10 hábiles). El primero fija una inflación pareja para los meses sin IPC "
                   "que alcanza; el siguiente toma esos meses como dados y despeja solo los nuevos, y así: queda una inflación por tramos. Cada tramo "
                   "depende del precio de un solo bono, así que conviene leerlo como promedio del tramo y no como el dato exacto de cada mes.")
    with st.expander("¿Por qué antes daba entre 1,44% y 1,84%?"):
        st.markdown("La versión anterior comparaba cada par tasa fija/CER desde hoy hasta el vencimiento y repartía la diferencia entre **todos** los meses, "
                    "incluidos los que ya tienen IPC publicado (agosto, 1,7%). Eso diluye la inflación de los meses que de verdad no se conocen: en un bono "
                    "de dos meses, casi la mitad del plazo ya está fijado. Despejando solo los meses desconocidos, el número sube y queda en línea con lo que "
                    "muestran Bonerío Local y Chamomberg.")
        mostrar(tabla_be[["vence", "cer", "be_mensual", "be_rem_mensual"]], {"vence": "{:%d/%m/%Y}", "be_mensual": "{:.2%}", "be_rem_mensual": "{:.2%}"},
                {"vence": "Vence", "cer": "Par CER", "be_mensual": "BE promedio de todo el plazo (método anterior)", "be_rem_mensual": "Igual con el REM"})

# ── Riesgo país (EMBI de JP Morgan) y spread de cada bono contra el Tesoro de EE.UU. ──
def num_ar(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


with tabR:
    rp = RIESGO
    hoy_rp = rp.index[-1]
    ref = lambda f: float(rp[rp.index <= f].iloc[-1]) if (rp.index <= f).any() else np.nan
    u = float(rp.iloc[-1])
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Riesgo país", f"{num_ar(u)} pb", f"al {hoy_rp:%d/%m/%Y}", delta_color="off")
    for col, nombre, f in [(r2, "1 día", rp.index[-2]), (r3, "1 mes", hoy_rp - pd.DateOffset(months=1)), (r4, "en el año", pd.Timestamp(hoy_rp.year - 1, 12, 31)),
                           (r5, "12 meses", hoy_rp - pd.DateOffset(months=12))]:
        a = ref(f)
        col.metric(f"Cambio {nombre}", f"{u - a:+.0f} pb", f"{u / a - 1:+.1%} (antes {num_ar(a)})", delta_color="inverse")
    periodo = st.radio("Período", ["6 meses", "1 año", "3 años", "5 años", "Desde 1999"], index=1, horizontal=True, key="rp_periodo")
    desde = {"6 meses": hoy_rp - pd.DateOffset(months=6), "1 año": hoy_rp - pd.DateOffset(years=1), "3 años": hoy_rp - pd.DateOffset(years=3),
             "5 años": hoy_rp - pd.DateOffset(years=5), "Desde 1999": rp.index[0]}[periodo]
    x = rp[rp.index >= desde]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x.index, x.values, color=AZUL, lw=1.6)
    ax.fill_between(x.index, x.values, x.min() * 0.95, color=AZUL, alpha=.08)
    ax.scatter([x.index[-1]], [x.iloc[-1]], color=AZUL, zorder=3)
    ax.annotate(f"{num_ar(x.iloc[-1])} pb", (x.index[-1], x.iloc[-1]), xytext=(-60, 10), textcoords="offset points", fontsize=8)
    for f_, v_, txt in [(x.idxmax(), x.max(), "máximo"), (x.idxmin(), x.min(), "mínimo")]:
        ax.annotate(f"{txt} {num_ar(v_)}", (f_, v_), xytext=(0, 8 if txt == "máximo" else -14), textcoords="offset points", fontsize=7, ha="center", color=GRIS)
    ax.set_ylabel("Puntos básicos"); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
    st.markdown("**Qué es.** Es la diferencia entre lo que rinden los bonos del gobierno argentino en dólares y lo que rinden los bonos del Tesoro de "
                "Estados Unidos, que se toman como libres de riesgo. Lo calcula JP Morgan (índice EMBI) y se mide en puntos básicos: "
                f"100 pb = 1 punto porcentual. Con {num_ar(u)} pb, un bono argentino rinde unos {u / 100:.1f} puntos más por año que uno de EE.UU.\n\n"
                "**Qué te dice.** Es cómo ve el mercado de afuera a la Argentina hacia adelante: cuánto premio pide para prestarle en vez de comprar "
                "bonos de EE.UU. Sube cuando aumenta la probabilidad de que no se paguen las deudas, cuando faltan dólares para los vencimientos o hay "
                "incertidumbre política; baja cuando el país acumula reservas, ordena las cuentas o paga sin problemas.\n\n"
                "**Por qué importa.** Marca el costo de financiarse: con riesgo país alto el Tesoro y las empresas (ONs) tienen que pagar más para "
                "colocar deuda en dólares, y por debajo de 500 pb suele abrirse la puerta de los mercados internacionales. Para un inversor, una baja del "
                "riesgo país es una suba de precio de los bonos en dólares (y al revés): con una duración de 5 años, 100 pb menos son cerca de 5% de ganancia.")

    st.subheader("El mismo cálculo con la curva de la app")
    UST, F_UST_FECHA, F_UST = cargar_ust(en_vivo, D["fecha"])
    SP = X.spread_ust(tabla_usd, UST)
    glob = SP[SP["familia"] == "Globales"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Spread mediano de los Globales", f"{glob['spread_pb'].median():.0f} pb", f"riesgo país {num_ar(ref(pd.Timestamp(D['fecha'])))} pb al {D['fecha']:%d/%m}", delta_color="off",
              help="Los Globales (ley Nueva York) son los más parecidos a la canasta del EMBI.")
    s2.metric("Bonares − Globales (mediana)", f"{SP.loc[SP['familia'] == 'Bonares', 'spread_pb'].median() - glob['spread_pb'].median():+.0f} pb", "prima por ley local", delta_color="off")
    s3.metric("Tesoro de EE.UU. a 5 años", f"{np.interp(5, UST.index, UST.values):.2%}", F_UST, delta_color="off")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    mac = SP["md"] * (1 + SP["tir"])
    for fam, col, mk in [("Bonares", B.COLORES["bonar"], "o"), ("Tesoro", B.COLORES["tesoro"], "^"), ("Globales", B.COLORES["global"], "s")]:
        sel = SP["familia"] == fam
        ax.scatter(mac[sel], SP.loc[sel, "tir"] * 100, color=col, marker=mk, label=fam, zorder=3)
    etiquetar(ax, mac, SP["tir"] * 100, SP.index)
    g = np.linspace(0, max(8, mac.max() + 0.5), 100)
    ax.plot(g, np.interp(g, UST.index, UST.values) * 100, color="black", lw=1.5, label="Tesoro de EE.UU. (curva par)")
    for t_, x_, y_ in zip(SP.index, mac, SP["tir"]):
        ax.vlines(x_, np.interp(x_, UST.index, UST.values) * 100, y_ * 100, color=GRIS, lw=0.6, alpha=.6)
    ax.set_xlabel("Duración (años)"); ax.set_ylabel("Tasa anual %"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig)
    mostrar(SP[["familia", "tir", "md", "ust", "spread_pb"]].sort_values("spread_pb"),
            {"tir": "{:.2%}", "md": "{:.2f}", "ust": "{:.2%}", "spread_pb": "{:+.0f}"},
            {"familia": "Familia", "tir": "TIR", "md": "MD", "ust": "Tesoro EE.UU. al mismo plazo", "spread_pb": "Spread (pb)"})
    st.caption(f"Fuentes: {F_RIESGO}; {F_UST}. Spread = TIR del bono − tasa del Tesoro de EE.UU. a su duración (curva par pasada a tasa efectiva anual). "
               "No da exactamente el número de JP Morgan, que usa su propia canasta, ponderaciones y precios, pero cuenta la misma historia: "
               "los bonos cortos (AL29, GD29, AO27) pagan mucho menos spread que los largos.")

# ── Devaluación esperada: futuros de dólar de A3 (ex Matba Rofex) ──
with tabFU:
    tc_ser = DP["a3500"].sort_index()
    f_fut = FUT["fecha"].iloc[0]
    spot_f = float(tc_ser[tc_ser.index <= pd.Timestamp(f_fut)].iloc[-1]) if (tc_ser.index <= pd.Timestamp(f_fut)).any() else float(tc_ser.iloc[-1])
    TF = X.devaluacion_futuros(FUT, spot_f, f_fut)
    ipc_pub0, rem0 = D["macro"]["ipc"].dropna().to_dict(), D["rem_mensual"]["mediana"].dropna().to_dict()
    ipc_f = lambda m: (ipc_pub0.get(m) or rem0.get(m) or list(rem0.values())[-1]) / 100
    TF["techo"] = [X.techo_proyectado(v, DL.techo_banda, ipc_f) for v in TF["vence"]]
    TF["dist_techo"] = TF["ajuste"] / TF["techo"] - 1
    cf_be = RP0["curva_be"]
    TF["tasa_pesos"] = [cf_be(d) if d <= RP0["max_cero"] + 30 else np.nan for d in TF["dias"]]
    TF["usd_cubierta"] = ((1 + TF["tasa_pesos"]) ** (TF["dias"] / 365) / (TF["ajuste"] / spot_f)) ** (365 / TF["dias"]) - 1
    st.caption("Un futuro de dólar es un contrato para comprar o vender dólares a fin de mes a un precio fijado hoy. Su precio contra el dólar oficial "
               "(A3500) es la devaluación que el mercado descuenta: si el futuro de diciembre vale \\$1.590 y el dólar hoy \\$1.516, el mercado espera "
               "(o se cubre por) una suba de 4,9% hasta fin de diciembre. Se compara con el dólar breakeven de los bonos dólar linked y con el techo de la banda.")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Dólar A3500", f"$ {num_ar(spot_f, 2)}", f"{tc_ser.index[tc_ser.index <= pd.Timestamp(f_fut)][-1]:%d/%m/%Y}", delta_color="off")
    f2.metric(f"Futuro {TF['contrato'].iloc[0]}", f"$ {num_ar(TF['ajuste'].iloc[0], 1)}", f"{TF['dev_directa'].iloc[0]:+.2%} hasta el {TF['vence'].iloc[0]:%d/%m}", delta_color="off")
    d6 = X.futuro_en(TF, spot_f, 182)
    f3.metric("Devaluación a 6 meses", f"{(d6 / spot_f) ** (30 / 182) - 1:.2%}" if np.isfinite(d6) else "–", f"mensual · dólar $ {num_ar(d6, 0)}" if np.isfinite(d6) else "", delta_color="off")
    dic = TF[TF["vence"].map(lambda v: v.month == 12)]
    if len(dic):
        f4.metric(f"Dólar a fin de año ({dic['contrato'].iloc[0]})", f"$ {num_ar(dic['ajuste'].iloc[0], 1)}", f"{dic['dev_directa'].iloc[0]:+.1%} · TNA {dic['tna'].iloc[0]:.1%}", delta_color="off")
    f5.metric("Vs techo de la banda", f"{TF['dist_techo'].iloc[-1]:.1%}", f"{TF['contrato'].iloc[-1].replace('DLR/', '')}: techo $ {num_ar(TF['techo'].iloc[-1], 0)}", delta_color="off",
              help="Negativo: el mercado ve al dólar por debajo del techo de la banda cambiaria (el BCRA vende dólares si lo toca).")

    dlb = RP0["dl"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = [pd.Timestamp(f_fut)] + list(pd.to_datetime(TF["vence"]))
    ax.plot(xs, [spot_f] + list(TF["ajuste"]), color=AZUL, lw=2, marker="o", ms=4, label="Dólar futuro (A3)")
    ax.plot(xs, [DL.techo_banda(f_fut)] + list(TF["techo"]), color=ROJO, lw=1.4, ls="--", label="Techo de la banda (proyectado con IPC y REM)")
    if len(dlb):
        ok = (dlb["dias"] >= 15) & (dlb["dias"] <= TF["dias"].max() + 45)
        ax.scatter(pd.to_datetime(dlb.loc[ok, "vence"]), dlb.loc[ok, "dolar_be"], color=VERDE, zorder=4, s=30, label="Dólar breakeven de cada dólar linked")
        etiquetar(ax, pd.to_datetime(dlb.loc[ok, "vence"]), dlb.loc[ok, "dolar_be"], dlb.index[ok])
    ax.scatter([pd.Timestamp(f_fut)], [spot_f], color="black", zorder=5, label="A3500 hoy")
    ax.set_ylabel("Pesos por dólar"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    xi = np.arange(len(TF)); w = .38
    ax.bar(xi - w / 2, TF["mensual"] * 100, w, color=AZUL, label="Devaluación mensual promedio hasta el vencimiento")
    ax.bar(xi + w / 2, TF["fwd_mensual"] * 100, w, color=VIOLETA, alpha=.75, label="Devaluación de ese mes (forward entre contratos)")
    if len(dlb):
        dl_ok = dlb[(dlb["dias"] >= 15) & (dlb["dias"] <= TF["dias"].max())]
        ax.plot(np.interp(dl_ok["dias"], TF["dias"], xi), dl_ok["deval_mensual"] * 100, color=VERDE, marker="D", ms=4, lw=1, label="Implícita en dólar linked")
    ax.set_xticks(xi, TF["contrato"].str.replace("DLR/", ""), fontsize=8); ax.set_ylabel("% mensual"); ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout(); st.pyplot(fig)
    mostrar(TF.set_index("contrato")[["vence", "dias", "ajuste", "dev_directa", "tna", "tasa_a3", "tea", "mensual", "fwd_mensual", "techo", "dist_techo", "usd_cubierta", "volumen", "interes_abierto"]],
            {"vence": "{:%d/%m/%Y}", "dias": "{:.0f}", "ajuste": "{:,.1f}", "dev_directa": "{:+.2%}", "tna": "{:.1%}", "tasa_a3": "{:.1%}", "tea": "{:.1%}",
             "mensual": "{:.2%}", "fwd_mensual": "{:.2%}", "techo": "{:,.0f}", "dist_techo": "{:+.1%}", "usd_cubierta": "{:.1%}", "volumen": "{:,.0f}", "interes_abierto": "{:,.0f}"},
            {"vence": "Vence", "dias": "Días", "ajuste": "Ajuste $", "dev_directa": "Devaluación directa", "tna": "TNA", "tasa_a3": "TNA según A3", "tea": "TEA",
             "mensual": "Mensual promedio", "fwd_mensual": "Mensual de ese tramo", "techo": "Techo banda $", "dist_techo": "vs techo", "usd_cubierta": "LECAP cubierta, en USD",
             "volumen": "Volumen (contratos)", "interes_abierto": "Interés abierto"})
    st.caption(f"Fuente: {F_FUT}; A3500 del BCRA. Cada contrato vence el último día hábil del mes. Devaluación directa = futuro / A3500 − 1; "
               "TNA = directa × 365 / días (A3 la informa contra su propio dólar spot, por eso difiere un poco); mensual = (futuro / A3500)^(30/días) − 1; "
               "la de ese tramo compara cada contrato con el anterior. «LECAP cubierta, en USD»: tasa anual en dólares de comprar una LECAP al plazo del "
               "contrato y vender dólar futuro (la curva de tasa fija contra la devaluación implícita): si es alta, el carry trade cubierto rinde.")
    if len(dlb):
        st.markdown("**Dólar linked contra futuros.** Si el dólar breakeven de un dólar linked queda por encima del futuro al mismo plazo, esos bonos "
                    "descuentan más devaluación que el mercado de futuros (el dólar linked está caro como cobertura); si queda por debajo, está barato.")
        cmp_ = dlb[dlb["dias"] >= 15][["vence", "dias", "dolar_be", "deval_mensual"]].copy()
        cmp_["futuro"] = [X.futuro_en(TF, spot_f, d) for d in cmp_["dias"]]
        cmp_["dif"] = cmp_["dolar_be"] / cmp_["futuro"] - 1
        mostrar(cmp_, {"vence": "{:%d/%m/%Y}", "dias": "{:.0f}", "dolar_be": "{:,.1f}", "deval_mensual": "{:.2%}", "futuro": "{:,.1f}", "dif": "{:+.2%}"},
                {"vence": "Vence", "dias": "Días", "dolar_be": "Dólar breakeven DL", "deval_mensual": "Devaluación mensual DL", "futuro": "Futuro al mismo plazo",
                 "dif": "DL vs futuro"}, na="después del último futuro")

# ── Obligaciones negociables (data912) ──
with tabN:
    mep_on = PAN["t1"]["AL30"][0] / PAN["t1"]["AL30D"][0] if PAN["t1"].get("AL30D") and PAN["t1"]["AL30D"][0] else np.nan
    TO = X.tabla_ons(ONS, mep_on)
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("ONs con precio", f"{TO['precio_mep'].notna().sum()} en MEP", f"{len(TO)} en total", delta_color="off")
    n2.metric("Volumen en ONs", f"US$ {num_ar(TO['volumen_usd'].sum() / 1e6, 1)} M", "pesos, MEP y cable", delta_color="off")
    n3.metric("MEP implícito (mediana)", f"$ {num_ar(TO['mep_implicito'].median(), 2)}", f"AL30: $ {num_ar(mep_on, 2)}", delta_color="off",
              help="Precio en pesos / precio en dólares de cada ON. Si una ON da un MEP más bajo, conviene usarla para comprar dólares (o al revés).")
    n4.metric("Canje (mediana)", f"{TO['canje'].median():.1%}", "cable sobre MEP", delta_color="off")
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
    mostrar(V_.head(150)[["emisor", "precio_mep", "var", "precio_cable", "precio_pesos", "mep_implicito", "canje", "spread_puntas", "volumen_usd"]],
            {"precio_mep": "{:.2f}", "var": "{:+.2%}", "precio_cable": "{:.2f}", "precio_pesos": "{:,.0f}", "mep_implicito": "{:,.2f}", "canje": "{:+.1%}",
             "spread_puntas": "{:.2%}", "volumen_usd": "{:,.0f}"},
            {"emisor": "Emisor", "precio_mep": "Precio MEP (D)", "var": "Var.", "precio_cable": "Precio cable (C)", "precio_pesos": "Precio $ (O)",
             "mep_implicito": "MEP implícito", "canje": "Canje", "spread_puntas": "Spread compra-venta", "volumen_usd": "Volumen US$"}, calor={"var": "var"})
    top = TO.sort_values("volumen_usd", ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([f"{t} · {e}" if e else t for t, e in zip(top.index, top["emisor"])], top["volumen_usd"] / 1e6, color=AZUL)
    ax.set_xlabel("Volumen del día, millones de US$"); ax.grid(alpha=.25, axis="x"); fig.tight_layout(); st.pyplot(fig)
    st.caption(f"Fuente: {F_ONS}. Cada ON cotiza en tres monedas: terminada en O en pesos, en D en dólar MEP y en C en dólar cable, por 100 de "
               "valor nominal. El emisor se muestra para los tickers verificados en informes de mercado (VALO); el resto, por su código.")

    st.subheader("Calculadora de TIR de una ON")
    st.caption("data912 publica precios pero no las condiciones de emisión. Cargá las del aviso de resultado o de la ficha técnica de BYMA "
               "(cupón, frecuencia, vencimiento y cómo amortiza) y la app calcula la TIR y la duración al precio de hoy.")
    con_d = TO[TO["precio_mep"].notna()].sort_values("volumen_usd", ascending=False)
    k1, k2, k3, k4, k5, k6 = st.columns([1.2, .8, .9, 1, .8, .8])
    on_sel = k1.selectbox("ON", con_d.index.tolist(), format_func=lambda t: f"{t}D · {TO.loc[t, 'emisor']}" if TO.loc[t, "emisor"] else f"{t}D")
    cup = k2.number_input("Cupón anual %", 0.0, 30.0, 8.0, 0.125) / 100
    frec = k3.selectbox("Paga", [2, 4, 12, 1], format_func={1: "anual", 2: "semestral", 4: "trimestral", 12: "mensual"}.get)
    venc_on = k4.date_input("Vencimiento", date(2030, 6, 30), format="DD/MM/YYYY")
    cuotas_on = k5.number_input("Cuotas de capital que faltan", 1, 40, 1, help="1 = paga todo al vencimiento (bullet).")
    resid_on = k6.number_input("Residual por 100 VN", 1.0, 100.0, 100.0, 1.0)
    liq_on = B.liquidacion_24hs(date.today() if en_vivo else D["fecha"])
    fl_on = X.flujos_on(venc_on, cup, frec, int(cuotas_on), liq_on, resid_on)
    px_on = float(TO.loc[on_sel, "precio_mep"])
    y_on, md_on = X.tir_on(px_on, fl_on, liq_on)
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Precio MEP", f"{px_on:.2f}")
    j2.metric("TIR", f"{y_on:.2%}" if np.isfinite(y_on) else "–")
    j3.metric("Duración modificada", f"{md_on:.2f}" if np.isfinite(md_on) else "–")
    j4.metric("Paridad (precio / residual)", f"{px_on / resid_on:.1%}")
    if fl_on:
        st.dataframe(pd.DataFrame(fl_on, columns=["Fecha", "Cupón", "Amortización"]).assign(Total=lambda t: t["Cupón"] + t["Amortización"])
                     .style.format({"Fecha": "{:%d/%m/%Y}", "Cupón": "{:.3f}", "Amortización": "{:.2f}", "Total": "{:.3f}"}), hide_index=True, height=220)
    st.caption(f"Liquidación {liq_on:%d/%m/%Y} (24 hs). Se toma el precio en dólar MEP como precio sucio (con intereses corridos), igual que los "
               "soberanos. Las fechas de pago se cuentan hacia atrás desde el vencimiento.")

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

with tabPF:
    st.caption("Cargá tus bonos y nominales (valor nominal, no pesos) y mirá cuánto vale la cartera, cuánto riesgo de tasa tiene, qué pasa en distintos "
               "escenarios, cuánto podría perder en un día malo (VaR) y qué tipo de cartera es. Cada número tiene su explicación al pasar el mouse por el ? "
               "y en el glosario de abajo.")
    tc_pf = (PAN["t1"]["AL30"][0] / PAN["t1"]["AL30C"][0]) if moneda == "ccl" else (PAN["t1"]["AL30"][0] / PAN["t1"]["AL30D"][0])
    UPF = PF.universo(tabla_usd, liq_usd, RP0, tc_pf, moneda)
    orden_g = {g: i for i, g in enumerate(PF.GRUPOS)}
    opciones_pf = sorted(UPF, key=lambda t: (orden_g[UPF[t]["grupo"]], UPF[t]["vence"]))
    e1, e2 = st.columns([1, 2.2])
    with e1:
        moneda_pf = st.radio("Valuar en", ["ARS", "USD"], horizontal=True, key="pf_moneda",
                             format_func={"ARS": "Pesos", "USD": "Dólares " + ("cable" if moneda == "ccl" else "MEP")}.get)
        st.caption(f"Dólar {'cable' if moneda == 'ccl' else 'MEP'} $ {num_ar(tc_pf, 2)} (AL30). Precios de hoy: hard dollar en dólar {'cable' if moneda == 'ccl' else 'MEP'}, "
                   "bonos en pesos con los supuestos por defecto de «Tesoro en pesos» (24 hs).")
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
                                                                                    help="Hard dollar, LECAP, BONCAP, CER, TAMAR, duales y dólar linked"),
                                           "Nominales": st.column_config.NumberColumn("Nominales (VN)", min_value=0, step=1000, format="%d")})
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
        fig.tight_layout(); st.pyplot(fig)
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
            fig.tight_layout(); col.pyplot(fig)

        st.subheader("Duración y key rate durations")
        KR = PF.krd_cartera(CPF)
        gk = [g for g in PF.GRUPOS if g in KR.columns and KR[g].abs().sum() > 0]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        base = np.zeros(len(KR))
        for g in gk:
            ax.bar(KR.index, KR[g], bottom=base, color=PF.COLORES_G[g], label=PF.GRUPOS[g]); base += KR[g].values
        ax.set_ylabel("Key rate duration (años)"); ax.set_xlabel("Plazo (nodo de la curva)"); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig)
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
        fig.tight_layout(); st.pyplot(fig)
        mostrar(EPF.assign(grupo=EPF["grupo"].map(PF.GRUPOS))[["grupo", "valor", "cambio", "impacto", "pct_impacto"]].sort_values("impacto"),
                {"valor": "{:,.0f}", "cambio": "{:+.2%}", "impacto": "{:+,.0f}", "pct_impacto": "{:.1%}"},
                {"grupo": "Grupo", "valor": f"Valor hoy {mon}", "cambio": "Cambio de precio", "impacto": f"Impacto {mon}", "pct_impacto": "% del impacto"},
                calor={"cambio": "var"})
        st.caption("Cada grupo se mueve con su propia curva: hard dollar en TIR en dólares, tasa fija en TIREA nominal, CER en tasa real y dólar linked "
                   "en TIR en dólares oficiales. Editá cualquier celda eligiendo «Propio» para armar tu escenario.")

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
        H_, FH_ = historicos_pf(en_vivo, sims)
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
                fig.tight_layout(); st.pyplot(fig)
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

        with st.expander("Glosario: qué significa cada número"):
            for k, v in GLOSARIO_PF.items():
                st.markdown(f"**{k}.** {v}")


# ── 4. LLM ──
with tab4:
    st.subheader("Resumen ejecutivo")
    st.caption("El LLM solo redacta: recibe los números ya calculados en un JSON y tiene prohibido inventar datos.")
    try:  # publicada en Streamlit Community Cloud, la clave se guarda en Settings › Secrets y no hace falta escribirla
        clave_guardada = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        clave_guardada = ""
    if clave_guardada:
        os.environ["ANTHROPIC_API_KEY"] = clave_guardada
        st.caption("Clave de Anthropic cargada desde los Secrets de la app.")
        clave = ""
    else:
        clave = st.text_input("ANTHROPIC_API_KEY (opcional)", type="password", help="Sin clave se usa una plantilla determinística.")
    if st.button("Generar resumen"):
        if clave:
            os.environ["ANTHROPIC_API_KEY"] = clave
        senal = B.senales_fija_vs_cer(D, curva, tabla_be)                       # retorno de cada par si se cumple el REM
        fw_fija = M.forwards(RP, "fija", 14)
        fw_usd = M.forwards(RP, "tesoro", 90, tabla_usd)
        extra = {"tasas_forward": {
            "pesos_tasa_fija_TEM_%": {k: round(float(v) * 100, 2) for k, v in fw_fija["fwd_mensual"].items()},
            f"dolares_tesoro_anual_%_{'cable' if moneda == 'ccl' else 'mep'}": {k: round(float(v) * 100, 2) for k, v in fw_usd["fwd_anual"].items()}}}
        T0 = P.trayectoria_be(RP0)
        comp_be = pd.DataFrame({"mercado": T0["inflacion"].where(T0["fuente"] != "INDEC (publicado)"), "rem": T0["rem"]}).dropna(subset=["mercado"])
        be_ctx = BE0[["be_mensual", "rem_mensual"]].rename(columns={"rem_mensual": "be_rem_mensual"})
        rp_1m = float(RIESGO[RIESGO.index <= RIESGO.index[-1] - pd.DateOffset(months=1)].iloc[-1])
        UST_c, _, _ = cargar_ust(en_vivo, D["fecha"])
        sp_g = X.spread_ust(tabla_usd, UST_c)
        extra["riesgo_pais"] = {"pb": int(RIESGO.iloc[-1]), "fecha": f"{RIESGO.index[-1]:%d/%m/%Y}", "cambio_1m_pb": int(RIESGO.iloc[-1] - rp_1m),
                                "spread_globales_pb": int(round(sp_g.loc[sp_g["familia"] == "Globales", "spread_pb"].median()))}
        tc_s = DP["a3500"].sort_index()
        sp_f = float(tc_s[tc_s.index <= pd.Timestamp(FUT["fecha"].iloc[0])].iloc[-1])
        TFc = X.devaluacion_futuros(FUT, sp_f, FUT["fecha"].iloc[0])
        dic_ = TFc[TFc["vence"].map(lambda v: v.month == 12)]
        fila_f = dic_.iloc[0] if len(dic_) else TFc.iloc[-1]
        extra["dolar_futuro"] = {"a3500": round(sp_f, 2), "contrato": fila_f["contrato"], "precio": round(float(fila_f["ajuste"]), 1),
                                 "devaluacion_%": round(float(fila_f["dev_directa"]) * 100, 2), "mensual_%": round(float(fila_f["mensual"]) * 100, 2)}
        ctx = B.contexto_resultados(tabla_usd, be_ctx, comp_be, senal, fecha=D["fecha"], extra=extra)
        with st.expander("Contexto enviado al modelo (JSON)"):
            st.json(ctx)
        try:
            st.markdown(B.resumen_llm(ctx).replace("$", "\\$")); st.caption("Generado por Claude (API de Anthropic).")
        except Exception as e:
            st.markdown(B.resumen_plantilla(ctx).replace("$", "\\$")); st.caption(f"Plantilla de respaldo ({e}).")
