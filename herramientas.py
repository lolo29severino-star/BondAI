"""
herramientas.py — tres herramientas de Bond Analytics que se apoyan en los cálculos de los otros módulos.

  1. Fichas de las ONs de los grandes emisores y su curva contra la de los soberanos   → spread_ons, EMISORES_TOP   (pestaña «ONs»)
  2. Retorno de los bonos en dólares hasta una fecha según la TIR a la que se venda      → retorno_hasta, tabla_retornos   (pestaña «Curva en dólares»)
  3. Tu cartera contra un índice de bonos                                                → comparar_indice   (pestaña «Mi portafolio»)
"""
import numpy as np
import pandas as pd

import bondai as B

# ───────────────────────────── 1. ONs de los grandes emisores ─────────────────────────────
EMISORES_TOP = ["YPF", "Pampa Energía", "Vista Energy", "Telecom", "IRSA", "Pan American Energy"]


def spread_ons(ON, tabla_usd):
    """Agrega a cada ON la TIR de la curva de Globales (Nelson-Siegel, ley Nueva York) a su misma duración y el spread en pb.
    Fuera del rango de duraciones de los Globales se usa el extremo más cercano (la curva no se extrapola)."""
    if ON is None or ON.empty:
        return ON
    g = tabla_usd[tabla_usd["familia"] == "Globales"]
    lo, hi = float(g["md"].min()), float(g["md"].max())
    ns = tabla_usd.attrs.get("Nueva York")
    out = ON.copy()
    out["tir_soberano"] = [B.nelson_siegel(np.clip(m, lo, hi), *ns) if ns else np.nan for m in out["md"]]
    out["spread_pb"] = (out["tir"] - out["tir_soberano"]) * 1e4
    return out


# ───────────────────────────── 2. retornos de los bonos en dólares ─────────────────────────────
def precio_a_tir(b, fecha, tir):
    """Precio sucio por 100 VN originales en 'fecha' si el mercado exige 'tir' (efectiva anual): flujos posteriores descontados."""
    fl = [((f - fecha).days / 365, c + a) for f, c, a in b["flujos"] if f > fecha]
    return float(sum(x / (1 + tir) ** t for t, x in fl))


def retorno_hasta(b, liq, precio, horizonte, tir_salida):
    """Retorno en dólares de comprar hoy a 'precio' y vender en 'horizonte' a 'tir_salida': (cupones y amortizaciones cobrados
    + precio de venta) / precio de compra − 1. Los pagos cobrados no se reinvierten. Si el bono vence antes, se cobra todo."""
    cobros = sum(c + a for f, c, a in b["flujos"] if liq < f <= horizonte)
    venta = precio_a_tir(b, horizonte, tir_salida) if horizonte < b["venc"] else 0.0
    total = (cobros + venta) / precio - 1
    dias = max((min(horizonte, b["venc"]) - liq).days, 1)
    return total, (1 + total) ** (365 / dias) - 1, cobros, venta


def tabla_retornos(tabla_usd, liq, horizonte, escenarios, modo="pb"):
    """Retorno total hasta 'horizonte' de cada bono en dólares para cada escenario de TIR de salida.
    modo 'pb': cambio de TIR en puntos básicos sobre la TIR de hoy; modo 'tir': TIR de salida en %."""
    U = B.universo_usd()
    filas = []
    for t, r in tabla_usd.iterrows():
        b = U[t]
        fila = {"ticker": t, "familia": r["familia"], "precio": r["precio"], "tir": r["tir"], "md": r["md"]}
        for e in escenarios:
            y = r["tir"] + e / 1e4 if modo == "pb" else e / 100
            fila[e] = retorno_hasta(b, liq, float(r["precio"]), horizonte, y)[0]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("ticker")


# ───────────────────────────── 3. cartera contra un índice ─────────────────────────────
def comparar_indice(rc, ri):
    """rc, ri: retornos diarios de la cartera y del índice (mismo índice de fechas). Devuelve (acumulados, estadísticas)."""
    df = pd.DataFrame({"cartera": rc, "indice": ri}).dropna()
    if len(df) < 5:
        return pd.DataFrame(), {}
    acum = (1 + df).cumprod()
    dd = lambda s: float((s / s.cummax() - 1).min())
    beta = float(np.cov(df["cartera"], df["indice"])[0, 1] / df["indice"].var()) if df["indice"].var() > 0 else np.nan
    st = {k: {"retorno": float(acum[k].iloc[-1] - 1), "vol": float(df[k].std() * np.sqrt(252)), "max_dd": dd(acum[k])} for k in ("cartera", "indice")}
    st["beta"], st["corr"] = beta, float(df.corr().iloc[0, 1])
    st["ruedas"], st["desde"], st["hasta"] = len(df), df.index[0], df.index[-1]
    return acum, st
