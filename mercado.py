"""
mercado.py — variación diaria contra el cierre anterior y volumen operado por rueda (pestañas nuevas de la app BondAI).

Uso desde la app:
    import mercado as M
    pan = M.cargar_paneles(usar_api=True)                  # paneles de BYMA en 24 hs (o el respaldo del cierre del 09/10/2026)
    tabla, info = M.variacion("cer", pan, datos_pesos)     # rendimiento de hoy contra el del cierre anterior
    V = M.cargar_volumen(usar_api=False)                   # 100 ruedas de volumen por bono (respaldo); con True baja el histórico
    esp, fechas = M.volumen_periodo(V, 20)                 # monto, volumen relativo y variación de precio de cada bono

Mismos cálculos que las pestañas "Variación diaria" y "Volumen" de la terminal web Bonos HD.
"""
import base64
import json
import time
import zlib
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import urllib3

import bondai as B
import pesos as P

urllib3.disable_warnings()

BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
FECHA_RESPALDO = date(2026, 10, 9)
RESPALDO = json.loads(zlib.decompress(base64.b64decode("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")))  # paneles al cierre del 09/10/2026 y 100 ruedas de volumen

FAMILIAS = {"fija": "Tasa fija", "cer": "CER", "tamar": "TAMAR", "dl": "Dólar linked", "dual": "Duales",
            "local": "Bonares y Tesoro (ley local)", "ny": "Globales (ley NY)", "bop": "Bopreal"}
CORTO = {"fija": "Tasa fija", "cer": "CER", "tamar": "TAMAR", "dl": "Dólar linked", "dual": "Duales", "local": "Ley local", "ny": "Ley NY", "bop": "Bopreal"}
COLORES = {"fija": "#2a78d6", "cer": "#1baf7a", "tamar": "#d99a00", "dl": "#7a5cf0", "dual": "#e0569a",
           "local": "#eb6834", "ny": "#c94040", "bop": "#13a9c9"}
GRUPOS = {"hd": "Hard dollar", "fija": "Tasa fija", "cer": "CER", "tamar": "TAMAR", "dl": "Dólar linked"}
USD_LOCAL = ["AL29", "AL30", "AL35", "AE38", "AL41", "AO27", "AO28", "AO29", "AN29"]
USD_NY = ["GD29", "GD30", "GD35", "GD38", "GD41", "GD46"]
BOPREAL = ["BPOA7", "BPOB7", "BPOC7", "BPOD7", "BPOA8", "BPOB8"]


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


def _px(x):
    return (x[0] or x[1] or x[3]) if x else 0


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


# ───────────────────────────── variación diaria ─────────────────────────────
def _habil_anterior(f):
    f -= timedelta(days=1)
    while not P.habil(f):
        f -= timedelta(days=1)
    return f


def variacion(grupo, paneles, datos_pesos, tamar=None, infl=None, duales=True, ipc_publicado=None, rem_mensual=None, margenes=None):
    """Rendimiento de hoy contra el del cierre anterior, bono por bono.
    grupo: hd (Bonares, Globales y Tesoro en dólar MEP), fija, cer, tamar o dl. El rendimiento de ayer usa el cierre anterior
    y la liquidación del día hábil previo. Devuelve (DataFrame, info) con x (MD o días), y, y0, precio, p0, var y cambio en pb."""
    filas = []
    if grupo == "hd":
        usd = B.universo_usd()
        pan = paneles["t1"]
        liq = P.siguiente_habil(paneles["fecha"]); liq0 = _habil_anterior(liq)
        for t, b in usd.items():
            x = pan.get(t + "D")
            if not x or not (_px(x) > 0 and x[3] > 0):
                continue
            hoy, ayer = B.metricas_bono(b, liq, _px(x)), B.metricas_bono(b, liq0, x[3])
            filas.append({"ticker": t, "x": hoy["md"], "dias": (b["venc"] - liq).days, "y": hoy["tir"], "y0": ayer["tir"], "precio": _px(x),
                          "p0": x[3], "vence": b["venc"], "grupo": "ny" if t.startswith("GD") else "local", "dual": False})
        info = {"metrica": "TIR", "xlab": "Duración modificada (años)", "ylab": "TIR (%)", "xlog": False, "liq": liq, "escala": 1.0,
                "unidad": "dólar MEP"}
    else:
        kw = dict(plazo="24HS", precio="px", tamar=tamar, infl=infl, ipc_publicado=ipc_publicado, rem_mensual=rem_mensual, margenes=margenes)
        R1 = P.calcular(datos_pesos, **kw)
        ayer = dict(datos_pesos, fecha=_habil_anterior(datos_pesos["fecha"]),
                    px={t: {k: [v[2] or v[0]] * 3 + list(v[3:]) for k, v in d.items() if isinstance(v, list)} for t, d in datos_pesos["px"].items()})
        R0 = P.calcular(ayer, **kw)
        claves = {"fija": ["lecap", "boncap"], "cer": ["cer"], "tamar": ["tamar"], "dl": ["dl"]}[grupo]
        met = {"fija": "tirea", "cer": "tna_real", "tamar": "margen_s_tamar", "dl": "tir_usd"}[grupo]
        div = 100 if grupo == "tamar" else 1
        def agregar(df, df0, m, dual=False):
            for t, r in df.iterrows():
                if t not in df0.index:
                    continue
                r0 = df0.loc[t]
                filas.append({"ticker": t, "x": r["dias"], "dias": r["dias"], "y": r[m] / div, "y0": r0[m] / div, "precio": r["precio"],
                              "p0": r0["precio"], "vence": r["vence"], "grupo": "a", "dual": dual})
        for k in claves:
            agregar(R1[k], R0[k], met)
        if duales and grupo in ("tamar", "cer") and len(R1["dual"]):
            d1, d0 = R1["dual"], R0["dual"]
            if grupo == "cer":
                d1, d0 = d1[d1["clase"] == "cer"].copy(), d0[d0["clase"] == "cer"].copy()
                for d in (d1, d0):
                    d["piso_tna"] = [P.tna30(v) for v in d["piso"]]
                agregar(d1, d0, "piso_tna", True)
            else:
                agregar(d1, d0, "margen_s_tamar", True)
        nombre = {"fija": "TEA", "cer": "TNA real", "tamar": "Margen s/ TAMAR", "dl": "TIR en dólares"}[grupo]
        info = {"metrica": nombre, "xlab": "Plazo (días, escala logarítmica)" if grupo in ("fija", "cer") else "Plazo (días)",
                "ylab": nombre + " (%)", "xlog": grupo in ("fija", "cer"), "liq": R1["liq"], "escala": 365.0, "unidad": "pesos",
                "tamar_sup": R1["tamar_sup"]}
    df = pd.DataFrame(filas)
    if df.empty:
        return df, info
    df = df.set_index("ticker").sort_values("x")
    df["var"] = df["precio"] / df["p0"] - 1
    df["cambio_pb"] = (df["y"] - df["y0"]) * 1e4
    return df, info


def resumen_variacion(df):
    """Suben/bajan, variación media, mediana del cambio de rendimiento, mayor suba y mayor baja."""
    sube, baja = int((df["var"] > 5e-5).sum()), int((df["var"] < -5e-5).sum())
    return {"suben": sube, "bajan": baja, "igual": len(df) - sube - baja, "media": float(df["var"].mean()),
            "mediana_pb": float(df["cambio_pb"].median()), "max": df["var"].idxmax(), "max_v": float(df["var"].max()),
            "min": df["var"].idxmin(), "min_v": float(df["var"].min())}


# ───────────────────────────── volumen ─────────────────────────────
def simbolos():
    """Tickers que suman al volumen de cada bono y familia. Los alias X sólo se usan para el precio si el ticker base no operó."""
    s = {}
    for t in USD_LOCAL + USD_NY:
        s[t] = ("local" if t in USD_LOCAL else "ny", [(t, "ars"), (t + "D", "mep"), (t + "C", "ccl")])
    for t in BOPREAL:
        s[t] = ("bop", [(t, "ars"), ("BP" + t[3:5] + "D", "mep"), ("BP" + t[3:5] + "C", "ccl")])
    familia = {}
    for t, *_ in P.FIJA:
        familia[t] = "fija"
    familia.update({"TO26": "fija", "TY30P": "fija"})
    for t in [c[0] for c in P.CERZ] + [b[0] for b in P.BONCER] + [x[0] for x in P.LARGOS]:
        familia[t] = "cer"
    for t in [x[0] for x in P.TAMAR]:
        familia[t] = "tamar"
    for d in P.DUALES:
        familia[d["t"]] = "dual"
    for t in [x[0] for x in P.DL]:
        familia[t] = "dl"
    for t, f in familia.items():
        s[t] = (f, [(t, "ars")] + [(a, "x") for a in P.ALIAS.get(t, [])])
    return s


def _historico(sesion, simbolo, desde, hasta):
    ts = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone(timedelta(hours=-3))).timestamp())
    r = sesion.get(f"{BYMA_API}/chart/historical-series/history", params={"symbol": simbolo + " 24HS", "resolution": "D",
                   "from": ts(desde), "to": ts(hasta) + 86400}, verify=False, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        return {}
    return {datetime.fromtimestamp(t, tz=timezone.utc).strftime("%y%m%d"): (c, h, l, v)
            for t, c, h, l, v in zip(j["t"], j["c"], j["h"], j["l"], j["v"])}


def cargar_volumen(usar_api=False, ruedas=100, progreso=None):
    """Monto operado por rueda (millones de pesos) de cada bono y su precio hace 0, 5, 20 y 60 ruedas.
    Con usar_api=True baja el histórico de BYMA de a un símbolo por vez (unos 160, tarda alrededor de un minuto: BYMA corta
    las conexiones si se le piden varias en paralelo). Si falla, usa el respaldo del cierre del 09/10/2026."""
    if not usar_api:
        return dict(RESPALDO["volumen"], fuente="respaldo")
    try:
        S = simbolos()
        todos = sorted({s for _, ss in S.values() for s, _ in ss})
        hasta, desde = date.today(), date.today() - timedelta(days=int(ruedas * 1.6) + 120)
        ses = requests.Session(); ses.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://open.bymadata.com.ar/"})
        H = {}
        for i, s in enumerate(todos):
            for intento in range(3):
                try:
                    H[s] = _historico(ses, s, desde, hasta); break
                except Exception:
                    time.sleep(0.5 * (intento + 1))
            time.sleep(0.12)
            if progreso:
                progreso((i + 1) / len(todos), s)
        cal = sorted(d for d, x in H.get("AL30", {}).items() if x[3] > 0)
        if len(cal) < 70:
            raise RuntimeError("histórico incompleto")
        fechas = cal[-ruedas:]
        fx = {d: {"ars": 1.0, "mep": H["AL30"][d][0] / H["AL30D"][d][0] if d in H.get("AL30D", {}) else np.nan,
                  "ccl": H["AL30"][d][0] / H["AL30C"][d][0] if d in H.get("AL30C", {}) else np.nan} for d in cal}
        b = {}
        for t, (fam, ss) in S.items():
            m = []
            for d in fechas:
                tot = 0.0
                for s, k in ss:
                    x = H.get(s, {}).get(d)
                    if k == "x" or not x or not x[3] > 0 or not np.isfinite(fx[d][k]):
                        continue
                    tot += x[3] * x[0] / 100 * fx[d][k] / 1e6
                m.append(round(tot))
            ult, p = 0.0, []
            for d in cal:
                x = H.get(t, {}).get(d)
                if x and x[3] > 0:
                    ult = x[0]
                else:
                    for s, k in ss:
                        a = H.get(s, {}).get(d)
                        if k == "x" and a and a[3] > 0:
                            ult = (a[1] + a[2]) / 2; break
                p.append(ult)
            c = [round(p[-1 - k], 4) if len(p) > k else 0 for k in (0, 5, 20, 60)]
            b[t] = [fam, c, m]
        return {"d": fechas, "b": b, "fuente": "API BYMA"}
    except Exception:
        return dict(RESPALDO["volumen"], fuente="respaldo")


def fecha_rueda(z):
    return date(2000 + int(z[:2]), int(z[2:4]), int(z[4:6]))


def monto_hoy(paneles, t, fam):
    """Monto operado en la rueda del panel (millones de pesos) y precio en pesos, para sumar la rueda de hoy (parcial)."""
    pan = paneles["t1"]
    al, ald, alc = pan.get("AL30"), pan.get("AL30D"), pan.get("AL30C")
    mep = _px(al) / _px(ald) if al and ald and _px(ald) else np.nan
    ccl = _px(al) / _px(alc) if al and alc and _px(alc) else np.nan
    if fam in ("local", "ny", "bop"):
        d, c = (t + "D", t + "C") if fam != "bop" else ("BP" + t[3:5] + "D", "BP" + t[3:5] + "C")
        m = (pan.get(t, [0] * 8)[6] + pan.get(d, [0] * 8)[6] * mep + pan.get(c, [0] * 8)[6] * ccl) / 1e6
    else:
        m = pan.get(t, [0] * 8)[6] / 1e6
    return (m if np.isfinite(m) else 0.0), _px(pan.get(t))


def volumen_periodo(V, ruedas=20, paneles=None):
    """Para cada bono: monto del período, del período anterior, volumen relativo (promedio diario del período sobre el de las
    60 ruedas anteriores) y variación de precio. Con paneles de una rueda posterior al histórico, la suma como 'hoy (parcial)'."""
    fechas = [fecha_rueda(z) for z in V["d"]]
    hoy = paneles is not None and paneles["fecha"] > fechas[-1]
    if hoy:
        fechas = fechas + [paneles["fecha"]]
    N = len(fechas); ini = N - ruedas
    k = {5: 1, 20: 2, 60: 3}[ruedas]
    filas = []
    for t, (fam, c, m) in V["b"].items():
        m = list(m); c0 = c[0]
        if hoy:
            mh, ph = monto_hoy(paneles, t, fam)
            m.append(mh)
            c0 = ph or c0
        per, prev = m[ini:], m[max(0, ini - ruedas):ini]
        primero = next((i for i, v in enumerate(m) if v > 0), len(m))
        base = [v for i, v in enumerate(m[max(0, ini - 60):ini], start=max(0, ini - 60)) if i >= primero]
        bm = np.mean(base) if len(base) >= 5 else np.nan
        mp = float(sum(per))
        filas.append({"ticker": t, "familia": fam, "monto": mp, "monto_previo": float(sum(prev)), "por_rueda": per,
                      "relativo": (mp / ruedas) / bm if bm and bm > 0 else np.nan,
                      "var": c0 / c[k] - 1 if c0 and c[k] else np.nan})
    return pd.DataFrame(filas).set_index("ticker"), fechas[ini:], hoy
