"""
mercado.py — el panel del día de BYMA y el ajuste de curvas de rendimiento por plazo.

Uso desde la app:
    import mercado as M
    pan = M.cargar_paneles(usar_api=True)            # panel de 24 hs de BYMA (o el respaldo del cierre del 09/10/2026)
    f, nombre, desvio = M.ajustar_curva(dias, tasas) # mejor curva (logarítmica, raíz, lineal o Nelson-Siegel) según el AIC

El panel da el precio de hoy de cada símbolo (sirve para el dólar MEP de los bonos y los precios de las ONs) y la curva se usa en
los gráficos de «Tesoro en pesos» para marcar qué bonos rinden más o menos que el resto de su curva.
"""
import base64
import json
import zlib
from datetime import date

import numpy as np
import requests
import urllib3

import pesos as P

urllib3.disable_warnings()

BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
FECHA_RESPALDO = date(2026, 10, 9)
RESPALDO = json.loads(zlib.decompress(base64.b64decode("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")))  # panel al cierre del 09/10/2026

COLORES = {"fija": "#2a78d6", "cer": "#1baf7a", "tamar": "#d99a00", "dl": "#7a5cf0", "dual": "#e0569a",
           "local": "#eb6834", "ny": "#c94040", "bop": "#13a9c9"}


# ───────────────────────────── paneles ─────────────────────────────
def _panel(ruta):
    r = requests.post(f"{BYMA_API}/{ruta}", json={"excludeZeroPxAndQty": False, "page_size": 5000, "T0": False, "T1": True, "T2": False},
                      headers={"Content-Type": "application/json"}, verify=False, timeout=20)
    r.raise_for_status()
    j = r.json()
    return j["data"] if isinstance(j, dict) else j


def cargar_paneles(usar_api=True):
    """Panel de 24 hs: símbolo -> [vwap, último, cierre, cierre anterior, bid, ask, monto, variación]."""
    try:
        assert usar_api
        filas = _panel("public-bonds") + _panel("lebacs")
        pan = {x["symbol"]: [x.get("vwap") or 0, x.get("trade") or 0, x.get("closingPrice") or 0, x.get("previousClosingPrice") or 0,
                             x.get("bidPrice") or 0, x.get("offerPrice") or 0, x.get("volumeAmount") or 0, x.get("imbalance") or 0] for x in filas}
        if "AL30D" not in pan:
            raise RuntimeError("panel incompleto")
        return {"t1": pan, "fecha": date.today(), "fuente": "API BYMA"}
    except Exception:
        return {"t1": RESPALDO["paneles"]["t1"], "fecha": FECHA_RESPALDO, "fuente": "respaldo"}


# ───────────────────────────── ajuste de curvas ─────────────────────────────
NOMBRES = {"log": "Logarítmica", "sqrt": "Raíz cuadrada", "lin": "Lineal", "ns": "Nelson-Siegel"}


def ajustar_curva(x, y, tipo="auto", escala=1.0):
    """Curva de rendimiento por plazo. tipo: auto (mejor AIC entre log, raíz, lineal y Nelson-Siegel), log, sqrt, lin o ns.
    Devuelve (función, nombre, desvío de los residuos) o None."""
    x, y = np.asarray(x, float) / escala, np.asarray(y, float)
    ok = (x > 0) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None
    def lineal(g):
        u = g(x)
        A = np.column_stack([np.ones_like(u), u])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        f = lambda v, b=b: b[0] + b[1] * g(np.asarray(v, float))
        return f, float(((y - f(x)) ** 2).sum()), 2
    cand = {"log": lambda: lineal(np.log), "sqrt": lambda: lineal(np.sqrt), "lin": lambda: lineal(lambda v: v)}
    def ns():
        if len(x) < 4:
            return None
        g = P.ajustar_ns(list(zip(x, y)))
        if g is None:
            return None
        f = lambda v: np.array([g(z) for z in np.atleast_1d(np.asarray(v, float))])
        return f, float(((y - f(x)) ** 2).sum()), 4
    cand["ns"] = ns
    tipos = (["log", "sqrt", "lin"] + (["ns"] if len(x) >= 6 else [])) if tipo == "auto" else [tipo]
    lo, hi = y.min(), y.max(); rango = (hi - lo) or abs(hi) or 1
    grilla = np.linspace(x.min(), x.max(), 61)
    mejor = None
    for k in tipos:
        r = cand[k]()
        if not r:
            continue
        f, sse, npar = r
        if tipo == "auto":
            v = f(grilla)
            if not np.all(np.isfinite(v)) or v.min() < lo - .25 * rango or v.max() > hi + .25 * rango:
                continue
        aic = len(x) * np.log(max(sse, 1e-14) / len(x)) + 2 * npar
        if mejor is None or aic < mejor[0]:
            mejor = (aic, k, f, sse, npar)
    if not mejor:
        return None
    _, k, f, sse, npar = mejor
    return (lambda v: f(np.asarray(v, float) / escala)), ("Auto: " if tipo == "auto" else "") + NOMBRES[k], float(np.sqrt(sse / max(1, len(x) - npar)))
