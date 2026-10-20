"""
dolares.py — dólar MEP, cable, A3500 y minorista para la pestaña «Dólares» de la app BondAI.

Uso desde la app:
    import dolares as DL
    df = DL.cargar_dolares(usar_api=True)       # cierres diarios de BYMA (AL30, GD30 y sus D y C) y A3500 / minorista del BCRA
    s = DL.series(df, "al30")                   # MEP, cable, canje y brechas por día
    tabla = DL.variaciones(s, hoy)              # cuánto se movió cada dólar en 1 día, 1 semana, 1 mes, etc.

Definiciones (las mismas de la terminal web Bonos HD):
    MEP   = precio del bono en pesos / precio en dólares MEP (ticker terminado en D)
    cable = precio en pesos / precio en dólares cable (ticker terminado en C)
    canje = cable / MEP − 1        brecha = dólar financiero / A3500 − 1
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

urllib3.disable_warnings()

BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
BCRA_API = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
FECHA_RESPALDO = date(2026, 10, 9)
RESPALDO = json.loads(zlib.decompress(base64.b64decode("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")))  # 10/10/2024 al 09/10/2026
SIMBOLOS = ["AL30", "AL30D", "AL30C", "GD30", "GD30D", "GD30C"]
# techo de la banda cambiaria: se mueve todos los días con la inflación de dos meses antes (Infobae, 01/10/2026)
TECHO_BANDA = [(date(2026, 9, 30), 1919.63), (date(2026, 10, 31), 1951.50)]


def _historico(sesion, simbolo, desde, hasta):
    """Velas diarias de BYMA (segmento 24 hs). Devuelve {fecha: cierre}; si el cierre cae fuera del rango del día usa el medio."""
    ts = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone(timedelta(hours=-3))).timestamp())
    r = sesion.get(f"{BYMA_API}/chart/historical-series/history", params={"symbol": simbolo + " 24HS", "resolution": "D",
                   "from": ts(desde), "to": ts(hasta) + 86400}, verify=False, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        return {}
    out = {}
    for t, c, h, l, v in zip(j["t"], j["c"], j["h"], j["l"], j["v"]):
        f = datetime.fromtimestamp(t, tz=timezone.utc).date()
        out[f] = c if (v and c and (not h or l <= c <= h)) else ((h + l) / 2 if v and h and l else np.nan)
    return out


def _bcra(id_var, desde, hasta):
    r = requests.get(f"{BCRA_API}/{id_var}", params={"desde": desde, "hasta": hasta, "limit": 3000}, timeout=20)
    r.raise_for_status()
    return pd.Series({pd.Timestamp(x["fecha"]): x["valor"] for x in r.json()["results"][0]["detalle"]}).sort_index()


def cargar_dolares(usar_api=True):
    """DataFrame diario con los cierres de AL30, AL30D, AL30C, GD30, GD30D y GD30C (BYMA, 24 hs), el A3500 (BCRA, variable 5)
    y el minorista (BCRA, variable 4). Si alguna fuente falla, usa el respaldo del 10/10/2024 al 09/10/2026."""
    R = RESPALDO
    bym = pd.DataFrame({k: R[k.lower()] for k in SIMBOLOS}, index=pd.to_datetime(R["d"]))
    bc = pd.DataFrame({"a3500": R["bcra"]["a3500"], "minorista": R["bcra"]["mino"]}, index=pd.to_datetime(R["bcra"]["d"]))
    fuente = {"bonos": "respaldo", "bcra": "respaldo"}
    if usar_api:
        try:
            hoy = date.today(); desde = hoy - timedelta(days=730)
            ses = requests.Session(); ses.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://open.bymadata.com.ar/"})
            cols = {}
            for s in SIMBOLOS:  # de a uno: BYMA corta las conexiones si se le piden varias en paralelo
                cols[s] = _historico(ses, s, desde, hoy); time.sleep(0.15)
            nuevo = pd.DataFrame(cols); nuevo.index = pd.to_datetime(nuevo.index)
            if len(nuevo.dropna()) > 100:
                bym, fuente["bonos"] = nuevo.sort_index(), "API BYMA"
        except Exception:
            pass
        try:
            hoy = date.today()
            a = _bcra(5, (hoy - timedelta(days=760)).isoformat(), hoy.isoformat())
            m = _bcra(4, (hoy - timedelta(days=760)).isoformat(), hoy.isoformat())
            bc, fuente["bcra"] = pd.DataFrame({"a3500": a, "minorista": m}), "API BCRA"
        except Exception:
            pass
    df = bym.join(bc, how="outer").sort_index()
    df[["a3500", "minorista"]] = df[["a3500", "minorista"]].ffill(limit=4)  # el BCRA publica días hábiles bancarios
    df = df[df["AL30"].notna() | df["a3500"].notna()]
    df.attrs["fuente"] = fuente
    return df


def series(df, ref="al30"):
    """MEP, cable, canje y brechas por día con el bono de referencia elegido (al30 o gd30)."""
    T = ref.upper()
    s = pd.DataFrame(index=df.index)
    s["mep"] = df[T] / df[T + "D"]
    s["cable"] = df[T] / df[T + "C"]
    s["canje"] = s["cable"] / s["mep"] - 1
    s["a3500"], s["minorista"] = df["a3500"], df["minorista"]
    s["brecha_mep"] = s["mep"] / s["a3500"] - 1
    s["brecha_cable"] = s["cable"] / s["a3500"] - 1
    return s


def valor_al(serie, fecha):
    """Último valor disponible a esa fecha (o antes)."""
    x = serie.dropna()
    x = x[x.index <= pd.Timestamp(fecha)]
    return (x.index[-1].date(), float(x.iloc[-1])) if len(x) else (None, np.nan)


def variaciones(s, hoy):
    """Variación de cada dólar (en %) y de canje y brechas (en puntos porcentuales) contra 1 día, 1 semana, 1 mes, 3 meses,
    el cierre de 2025 y 12 meses antes, más el mínimo y el máximo de 12 meses."""
    hoy = pd.Timestamp(hoy)
    horizontes = {"1 día": None, "1 semana": hoy - pd.Timedelta(days=7), "1 mes": hoy - pd.DateOffset(months=1), "3 meses": hoy - pd.DateOffset(months=3),
                  "En el año": pd.Timestamp("2025-12-31"), "12 meses": hoy - pd.DateOffset(months=12)}
    filas = []
    for col, nombre, tipo in [("mep", "Dólar MEP", "nivel"), ("cable", "Dólar cable", "nivel"), ("a3500", "A3500", "nivel"), ("minorista", "Minorista", "nivel"),
                              ("canje", "Canje", "spread"), ("brecha_mep", "Brecha MEP", "spread"), ("brecha_cable", "Brecha cable", "spread")]:
        x = s[col].dropna(); x = x[x.index <= hoy]
        if x.empty:
            continue
        ult = float(x.iloc[-1]); fila = {"dolar": nombre, "tipo": tipo, "ultimo": ult}
        for k, f in horizontes.items():
            ant = float(x.iloc[-2]) if f is None and len(x) > 1 else valor_al(x, f)[1] if f is not None else np.nan
            fila[k] = ult / ant - 1 if tipo == "nivel" else ult - ant
        x12 = x[x.index > hoy - pd.DateOffset(months=12)]
        fila["min_12m"], fila["max_12m"] = float(x12.min()), float(x12.max())
        filas.append(fila)
    return pd.DataFrame(filas).set_index("dolar")


def techo_banda(fecha):
    """Techo de la banda cambiaria interpolado (geométrico) entre los valores publicados de fin de septiembre y de octubre."""
    (f0, v0), (f1, v1) = TECHO_BANDA
    if fecha <= f0:
        return v0
    if fecha >= f1:
        return v1
    return v0 * (v1 / v0) ** ((fecha - f0).days / (f1 - f0).days)


def dolar_por_bono(paneles, bonos):
    """MEP y cable implícitos en cada bono con el panel de hoy: precio en pesos / precio D y / precio C (VWAP o último)."""
    px = lambda x: (x[0] or x[1] or x[2] or x[3]) if x else 0  # antes de la apertura (11 hs) solo hay cierre anterior
    P = paneles["t1"]
    mep_ref = px(P.get("AL30")) / px(P.get("AL30D")) if px(P.get("AL30D")) else np.nan
    filas = []
    for t in bonos:
        d_, c_ = (t + "D", t + "C") if not t.startswith("BPO") else ("BP" + t[3:5] + "D", "BP" + t[3:5] + "C")
        a, d, c = px(P.get(t)), px(P.get(d_)), px(P.get(c_))
        if not (a > 0):
            continue
        mep = a / d if d > 0 else np.nan
        cab = a / c if c > 0 else np.nan
        filas.append({"bono": t, "mep": mep, "cable": cab, "canje": cab / mep - 1, "vs_mep_al30": mep / mep_ref - 1,
                      "monto_d": (P.get(d_) or [0] * 8)[6] / 1e6, "monto_c": (P.get(c_) or [0] * 8)[6] / 1e6})
    cols = ["bono", "mep", "cable", "canje", "vs_mep_al30", "monto_d", "monto_c"]
    return pd.DataFrame(filas, columns=cols).set_index("bono")
