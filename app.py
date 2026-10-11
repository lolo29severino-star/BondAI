"""
BondAI · prototipo en Streamlit
Ejecutar:  pip install -r requirements.txt   →   streamlit run app.py
Publicada en internet: ver README.md (GitHub + Streamlit Community Cloud).
Usa las mismas funciones que el notebook (bondai.py, pesos.py, mercado.py, dolares.py y licitaciones.py deben estar en la misma carpeta).
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
import licitaciones as L
import mercado as M
import pesos as P

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


@st.cache_data(show_spinner="Entrenando y validando los modelos de Machine Learning…")
def modelos(en_vivo):
    d = cargar(en_vivo)
    X, y = B.construir_features(d["macro"])
    bt = B.validacion_temporal(X, y, h=1)
    mae = pd.DataFrame({f"h={h}": B.metricas_error(B.validacion_temporal(X, y, h=h))["MAE (pp)"] for h in (1, 3, 6)})
    pron = B.pronostico_directo(X, y)
    return X, y, bt, mae, pron



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


def borrar_precios():
    """Borra de la memoria solo los precios y tasas (no el volumen ni los modelos, que tardan en recalcularse)."""
    for f in (cargar, calcular, cargar_pesos, cargar_paneles, cargar_dolares):
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
DP = cargar_pesos(en_vivo)
for k, v in DP["fuente"].items():
    st.sidebar.write(f"· pesos/{k}: {v}")
PAN = cargar_paneles(en_vivo)
st.sidebar.write(f"· paneles 24 hs: {PAN['fuente']}")
st.sidebar.caption(f"Fecha de análisis: {D['fecha']:%d/%m/%Y} · datos bajados a las {D['bajado']:%H:%M} (hora argentina)")
st.sidebar.caption("Herramienta académica. No es recomendación de inversión.")

tabla_usd, liq_usd, curva, tabla_be, cer_ref, liq_ars, mensual = calcular(en_vivo, moneda)

st.title("BondAI · Renta fija argentina con IA")
c1, c2, c3, c4 = st.columns(4)
c1.metric("TIR mediana soberanos USD", f"{tabla_usd['tir'].median():.2%}")
jun = tabla_be.iloc[-1]
c2.metric(f"Breakeven {jun.name}/{jun['cer']}", f"{jun['be_mensual']:.2%} mensual", f"{jun['be_anual']:.1%} anual", delta_color="off")
c3.metric("REM próximo mes", f"{D['rem_mensual']['mediana'].iloc[0]:.2f}%", f"{D['rem_mensual'].index[0]}", delta_color="off")
c4.metric("Último IPC", f"{D['macro']['ipc'].dropna().iloc[-1]:.1f}%", D["macro"]["ipc"].dropna().index[-1], delta_color="off")

tab1, tabP, tabV, tabO, tabF, tabD, tabL, tab2, tab3, tab4 = st.tabs(["Curva en dólares", "Tesoro en pesos", "Variación diaria", "Volumen",
                                                                      "Flujos de fondos", "Dólares", "Licitaciones",
                                                                      "Inflación breakeven", "Pronóstico con ML", "Resumen con IA"])

# ── 1. curva USD + Monte Carlo ──
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

    st.subheader("Simulador Monte Carlo")
    a, b, c, d_, e = st.columns(5)
    bono = a.selectbox("Bono", tabla_usd.index.tolist(), index=tabla_usd.index.tolist().index("AE38"))
    meses = b.slider("Horizonte (meses)", 1, 24, 6)
    tir0 = float(tabla_usd.loc[bono, "tir"])
    objetivo = c.number_input("TIR objetivo %", value=round(tir0 * 100, 2), step=0.25) / 100
    sigma = d_.number_input("Volatilidad TIR (pb/año)", value=164, step=10) / 1e4
    kappa = e.number_input("Reversión κ", value=1.5, step=0.25)
    usd = B.universo_usd()
    salida = min(B.sumar_meses(liq_usd, meses), usd[bono]["venc"] - timedelta(days=1))
    r, _ = B.monte_carlo_bono(usd[bono], liq_usd, tabla_usd.loc[bono, "precio"], salida, tir0, objetivo, kappa, sigma)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retorno esperado", f"{r.mean():.2%}")
    m2.metric("Percentil 5", f"{np.percentile(r, 5):.2%}")
    m3.metric("Percentil 95", f"{np.percentile(r, 95):.2%}")
    m4.metric("Prob. retorno > 0", f"{(r > 0).mean():.0%}")
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.hist(r * 100, bins=60, color=B.COLORES["bonar"], alpha=.8); ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"Retorno total en USD de {bono} al {salida:%d/%m/%Y}, %")
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
    RP = P.calcular(DP, plazo_p, precio_p, tamar_sup, infl_sup, ipc_publicado=IPC_PUB, rem_mensual=REM_MED, dev=dev_sup)
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
        "cer": "Valor técnico = 100 × CER de 10 hábiles antes de la liquidación / CER base (10 hábiles antes de la emisión). Inflación BE: se toma tal cual el CER que el BCRA ya publicó y se busca la inflación mensual pareja de los meses todavía no publicados que hace que el bono pague lo mismo que la LECAP/BONCAP interpolada a su plazo (n/a: el CER hasta el vencimiento ya está publicado; —: vence después de la letra a tasa fija más larga). REM: promedio mensual del sendero del REM para esos mismos meses. TNA 30/365 = ((1 + TIR)^(30/365) − 1) × 365/30; la nominal usa el sendero IPC publicado → REM → supuesto. DICP, DIP0, PARP, PAP0 y CUAP son los bonos del canje 2005 (base CER 31/12/2003).",
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
                st.markdown(f"A3500 hoy **${RP['tc_hoy']:,.2f}** · Dólar breakeven {ult['vence']:%m/%Y}: **${ult['dolar_be']:,.0f}** ({ult['deval_total']:+.1%})")
                cols = ["vence", "dias", "precio", "precio_usd", "var", "tir_usd", "dolar_be", "deval_mensual", "tirea_esperada", "vs_fija_pb", "md", "vol"]
                mostrar(df[cols],
                        {**COM, "precio": "{:,.0f}", "precio_usd": "{:.2f}", "tir_usd": "{:.2%}", "dolar_be": "${:,.2f}", "deval_mensual": "{:.2%}",
                         "tirea_esperada": "{:.2%}", "vs_fija_pb": "{:+.0f}"},
                        {**NOM, "precio": "Precio $", "precio_usd": "Precio USD", "tir_usd": "TIR USD", "dolar_be": "Dólar BE", "deval_mensual": "Deval. mensual",
                         "tirea_esperada": "TIREA $ esperada", "vs_fija_pb": "vs tasa fija (pb)"},
                        na="= curva", calor={"var": "var"})
                st.pyplot(grafico_pesos(df[df["dias"] >= 15], "deval_mensual", None, "#1baf7a", "Devaluación mensual implícita"))
            st.caption(NOTAS[k])

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
    VD, info = M.variacion(grupo, PAN, DP, tamar=tamar_sup, infl=infl_sup, duales=duales_v, ipc_publicado=IPC_PUB, rem_mensual=REM_MED)
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

# ── 2. breakeven ──
with tab2:
    st.subheader("Pares tasa fija vs CER con el mismo vencimiento")
    st.caption(f"Contado inmediato {liq_ars:%d/%m/%Y}. BE según REM: breakeven que daría el sendero mensual del REM con el mismo plazo.")
    mostrar(tabla_be[["vence", "cer", "tirea_fija", "tir_real", "be_mensual", "be_anual", "be_acum", "be_rem_mensual", "dif_vs_rem_pb"]],
            {"vence": "{:%d/%m/%Y}", "tirea_fija": "{:.2%}", "tir_real": "{:.2%}", "be_mensual": "{:.2%}", "be_anual": "{:.1%}", "be_acum": "{:.2%}",
             "be_rem_mensual": "{:.2%}", "dif_vs_rem_pb": "{:+.0f}"},
            {"vence": "Vence", "cer": "Par CER", "tirea_fija": "TIREA fija", "tir_real": "TIR real", "be_mensual": "BE mensual",
             "be_anual": "BE anual", "be_acum": "BE acumulada", "be_rem_mensual": "BE según REM", "dif_vs_rem_pb": "Mercado − REM (pb)"},
            na="sin REM")
    st.subheader("Inflación implícita mes a mes vs REM")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    xs = np.arange(len(mensual)); w = .38
    ax.bar(xs - w / 2, mensual["mercado"] * 100, w, color=B.COLORES["mercado"], label="Mercado")
    ax.bar(xs + w / 2, mensual["rem"] * 100, w, color=B.COLORES["rem"], label="REM mediana")
    ax.set_xticks(xs, [B.mes_corto(m) for m in mensual.index]); ax.set_ylabel("% mensual"); ax.legend()
    st.pyplot(fig)
    mostrar(mensual[["mercado", "rem", "rem_p25", "rem_p75"]], {c: "{:.2%}" for c in ["mercado", "rem", "rem_p25", "rem_p75"]},
            {"mercado": "Mercado", "rem": "REM mediana", "rem_p25": "REM P25", "rem_p75": "REM P75"})

# ── 3. ML ──
with tab3:
    X, y, bt, mae, pron = modelos(en_vivo)
    comp = B.comparar_expectativas(pron, mensual, y)
    st.subheader("Validación fuera de muestra")
    st.caption("Error absoluto medio en puntos porcentuales de inflación mensual. Cada pronóstico se hizo solo con datos anteriores.")
    mostrar(mae.sort_values("h=1"), {c: "{:.3f}" for c in mae.columns})
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(bt.index.to_timestamp(), bt["Real"], color="black", lw=2, label="IPC real")
    ax.plot(bt.index.to_timestamp(), bt["Paseo aleatorio"], ls="--", color=B.COLORES["gris"], label="Paseo aleatorio")
    ax.plot(bt.index.to_timestamp(), bt["Ensamble ML"], color=B.COLORES["modelo"], label="Ensamble ML")
    ax.legend(fontsize=8); st.pyplot(fig)
    st.subheader("Mercado, analistas y modelo")
    mostrar(comp[["h", "modelo", "mercado", "rem", "rem_p25", "rem_p75"]],
            {"h": "{:.0f}", "modelo": "{:.2%}", "mercado": "{:.2%}", "rem": "{:.2%}", "rem_p25": "{:.2%}", "rem_p75": "{:.2%}"},
            {"modelo": "Modelo ML", "mercado": "Mercado", "rem": "REM mediana", "rem_p25": "REM P25", "rem_p75": "REM P75"})
    st.caption("h = meses hacia adelante desde el último IPC publicado. Más allá de 3 meses el modelo no supera al paseo aleatorio en el backtest.")
    st.subheader("Señal tasa fija vs CER")
    senal = B.senales_fija_vs_cer(D, curva, tabla_be, comp)
    mostrar(senal, {"vence": "{:%d/%m/%Y}", "BE mensual": "{:.2%}", "Tasa fija": "{:+.2%}", "CER con modelo": "{:+.2%}", "CER con REM": "{:+.2%}"},
            {"Tasa fija": "Retorno tasa fija", "CER con modelo": "Retorno CER (modelo)", "CER con REM": "Retorno CER (REM)"})

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
        X, y, bt, mae, pron = modelos(en_vivo)
        comp = B.comparar_expectativas(pron, mensual, y)
        senal = B.senales_fija_vs_cer(D, curva, tabla_be, comp)
        usd = B.universo_usd()
        r, _ = B.monte_carlo_bono(usd["AE38"], liq_usd, tabla_usd.loc["AE38", "precio"], B.sumar_meses(liq_usd, 6),
                                  tabla_usd.loc["AE38", "tir"], tabla_usd.loc["AE38", "tir"])
        res_mc = pd.Series({"Retorno esperado": r.mean(), "Mediana": np.median(r), "Percentil 5": np.percentile(r, 5),
                            "Percentil 95": np.percentile(r, 95), "Prob. retorno > 0": (r > 0).mean()})
        ctx = B.contexto_resultados(tabla_usd, tabla_be, comp, senal, B.metricas_error(bt), res_mc, D["fecha"])
        with st.expander("Contexto enviado al modelo (JSON)"):
            st.json(ctx)
        try:
            st.markdown(B.resumen_llm(ctx)); st.caption("Generado por Claude (API de Anthropic).")
        except Exception as e:
            st.markdown(B.resumen_plantilla(ctx).replace("\n\n", "\n\n")); st.caption(f"Plantilla de respaldo ({e}).")
