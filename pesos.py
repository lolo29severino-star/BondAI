"""
pesos.py — los bonos del Tesoro en pesos: LECAP, BONCAP y BONTE (tasa fija), CER, TAMAR, duales y dólar linked.

Lo usan: «Tesoro en pesos», «Mi portafolio», «Flujos de fondos» y el informe semanal.
Idea central: la tasa fija es la vara. Cada otro bono se compara con la curva de tasa fija al mismo plazo: qué inflación (CER),
qué TAMAR (TAMAR y duales) o qué dólar (dólar linked) lo empata.

Índice
  1. Calendario: feriados, días hábiles y fechas de pago
  2. Instrumentos: las condiciones de cada bono (emisión, vencimiento, TEM, CER base, margen TAMAR…)
  3. Bonos nuevos: registrar_nuevos() agrega los que el Tesoro emite en cada licitación
  4. Carga de datos: precios de BYMA (o el último de data912 con el mercado cerrado) y CER, TAMAR y A3500 del BCRA
     (con respaldo del 09/10/2026)
  5. Matemática: TIR, duración, curvas, conversión de tasas y proyección del CER
  6. Cálculo principal: calcular() valúa todos los bonos; dolar_esperado_futuros() arma el dólar que esperan los futuros
  7. Inflación mes a mes que descuenta el mercado (trayectoria_be)

Ejemplo:
    import pesos as P
    datos = P.cargar(usar_api=True)
    res = P.calcular(datos, plazo="24HS", tamar=None, infl=None, dev=None)   # res["lecap"], res["cer"], res["dl"]…: una tabla por tipo
"""
import base64
import json
import zlib
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import urllib3
from scipy.optimize import brentq

urllib3.disable_warnings()

BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
DATA912 = "https://data912.com/live"   # último precio operado, para cuando BYMA vacía su panel
UA = {"User-Agent": "Mozilla/5.0 (Bond Analytics, trabajo académico)"}
BCRA_API = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
FECHA_RESPALDO = date(2026, 10, 9)
RESPALDO = json.loads(zlib.decompress(base64.b64decode("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")))  # cierre del 09/10/2026

TIPOS = [("lecap", "LECAP", "Letras capitalizables a tasa fija"),
         ("boncap", "BONCAP y BONTE", "Bonos a tasa fija"),
         ("cer", "CER", "BONCER, LECER y bonos del canje 2005: ajustan el capital por inflación"),
         ("tamar", "TAMAR", "Tasa variable de plazo fijo mayorista"),
         ("dual", "Duales", "Pagan la mejor de dos patas"),
         ("dl", "Dólar linked", "Capital en dólares, pago en pesos al A3500")]

# ── 1. Calendario: feriados, días hábiles y fechas de pago ─────────────────────────────────────────────────────
FERIADOS = {date.fromisoformat(f) for f in (
    "2026-01-01 2026-02-16 2026-02-17 2026-03-23 2026-03-24 2026-04-02 2026-04-03 2026-05-01 2026-05-25 2026-06-15 2026-07-09 "
    "2026-07-10 2026-08-17 2026-10-12 2026-11-09 2026-11-10 2026-11-23 2026-12-07 2026-12-08 2026-12-25 2027-01-01 2027-02-08 2027-02-09 2027-03-24 "
    "2027-03-26 2027-04-02 2027-05-25 2027-06-21 2027-07-09 2027-08-16 2027-10-11 2027-11-22 2027-12-08 2027-12-25 "
    "2028-01-01 2028-03-24 2028-04-14 2028-05-01 2028-05-25 2028-06-20 2028-07-09 2028-12-08 2028-12-25 "
    "2029-01-01 2029-03-24 2029-03-30 2029-05-01 2029-05-25 2029-06-20 2029-07-09 2029-12-08 2029-12-25 "
    "2030-01-01 2030-03-24 2030-04-19 2030-05-01 2030-05-25 2030-06-20 2030-07-09 2030-12-08 2030-12-25").split()}
habil = lambda f: f.weekday() < 5 and f not in FERIADOS
D = date.fromisoformat


def siguiente_habil(f):
    f += timedelta(days=1)
    while not habil(f):
        f += timedelta(days=1)
    return f


ajustar_pago = lambda f: f if habil(f) else siguiente_habil(f)


def restar_habiles(f, n):
    while n:
        f -= timedelta(days=1)
        n -= habil(f)
    return f


def d360(a, b):
    d1, d2 = min(a.day, 30), b.day
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (b.year - a.year) * 360 + (b.month - a.month) * 30 + (d2 - d1)


def sumar_meses(f, n):
    m = f.month - 1 + n
    y, m = f.year + m // 12, m % 12 + 1
    ult = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
    return date(y, m, min(f.day, ult))


# ── 2. Instrumentos: las condiciones de cada bono ──────────────────────────────────────────────────────────────
FIJA = [("S16O6", "LECAP", "2026-07-31", "2026-10-16", .0205), ("S30O6", "LECAP", "2025-10-31", "2026-10-30", .0255),
        ("S13N6", "LECAP", "2026-06-30", "2026-11-13", .0210), ("S30N6", "LECAP", "2025-12-15", "2026-11-30", .0230),
        ("S29E7", "LECAP", "2026-08-31", "2027-01-29", .0225), ("T15E7", "BONCAP", "2025-01-31", "2027-01-15", .0205),
        ("T30A7", "BONCAP", "2025-10-31", "2027-04-30", .0255), ("T31Y7", "BONCAP", "2025-12-15", "2027-05-31", .0240),
        ("T30J7", "BONCAP", "2026-01-16", "2027-06-30", .0258)]
# fecha base = CER de 10 hábiles antes de la liquidación; TZXM8, TZXM9 y TZXO7 liquidaron el 01/04 y el 30/06/2026 (base implícita
# en la TIREA oficial de sus reaperturas), aunque la ficha de BYMA diga 31/03 y 29/06
CERZ = [("TZXO6", "BONCER", "2024-10-31", "2026-10-30", "2024-10-17"), ("X30N6", "LECER", "2025-12-15", "2026-11-30", "2025-11-28"),
        ("TZXD6", "BONCER", "2024-03-15", "2026-12-15", "2024-03-01"), ("X29E7", "LECER", "2026-08-14", "2027-01-29", "2026-07-31"),
        ("TZXM7", "BONCER", "2024-05-20", "2027-03-31", "2024-05-06"), ("TZXA7", "BONCER", "2025-11-28", "2027-04-30", "2025-11-12"),
        ("TZXY7", "BONCER", "2025-12-15", "2027-05-31", "2025-11-28"), ("TZX27", "BONCER", "2024-02-01", "2027-06-30", "2024-01-18"),
        ("TZXS7", "BONCER", "2026-03-31", "2027-09-30", "2026-03-13"), ("TZXO7", "BONCER", "2026-06-30", "2027-10-29", "2026-06-16"),
        ("TZXD7", "BONCER", "2024-03-15", "2027-12-15", "2024-03-01"), ("TZXM8", "BONCER", "2026-04-01", "2028-03-31", "2026-03-16"),
        ("TZX28", "BONCER", "2024-02-01", "2028-06-30", "2024-01-18"), ("TZXS8", "BONCER", "2026-03-31", "2028-09-29", "2026-03-13"),
        ("TZXD8", "BONCER", "2026-06-30", "2028-12-15", "2026-06-16"), ("TZXM9", "BONCER", "2026-04-01", "2029-03-28", "2026-03-16")]
# BONCER con cupón: ticker, emisión, vencimiento, fecha base, tasa real, primer pago, inicio de amortización, cuotas, condición
BONCER = [("TX26", "2020-09-04", "2026-11-09", "2020-08-21", .02, "2021-05-09", "2024-11-09", 5,
           "2% anual sobre saldo ajustado, pagos 9/5 y 9/11; amortiza en 5 cuotas semestrales"),
          ("TX28", "2020-09-04", "2028-11-09", "2020-08-21", .0225, "2021-05-09", "2024-05-09", 10,
           "2,25% anual sobre saldo ajustado, pagos 9/5 y 9/11; amortiza en 10 cuotas semestrales"),
          ("TX31", "2022-05-31", "2031-11-30", "2022-05-13", .025, "2022-11-30", "2027-05-30", 10,
           "2,5% anual sobre saldo ajustado, pagos 30/5 y 30/11; amortiza en 10 cuotas semestrales")]
# Canje 2005: base CER del 31/12/2003; flujos por 100 VN originales sobre el capital capitalizado (Discount 127%, Cuasipar 138,86%).
# ticker, tipo, vencimiento, (desde, hasta, mes1, mes2) de las fechas de pago, tasa según el inicio del período, inicio de amortización, capitalización
LARGOS = [("DICP", "Discount", "2033-12-31", ("2013-12-31", "2033-12-31", 6, 12), lambda ini: .0583, "2024-06-30", 1.27),
          ("DIP0", "Discount", "2033-12-31", ("2013-12-31", "2033-12-31", 6, 12), lambda ini: .0583, "2024-06-30", 1.27),
          ("PARP", "Par", "2038-12-31", ("2019-03-31", "2038-09-30", 3, 9), lambda ini: .0177 if ini < date(2029, 3, 31) else .0248, "2029-09-30", 1.0),
          ("PAP0", "Par", "2038-12-31", ("2019-03-31", "2038-09-30", 3, 9), lambda ini: .0177 if ini < date(2029, 3, 31) else .0248, "2029-09-30", 1.0),
          ("CUAP", "Cuasipar", "2045-12-31", ("2013-12-31", "2045-12-31", 6, 12), lambda ini: .0331, "2036-06-30", 1.38859)]
COND_LARGOS = {"Discount": "5,83% anual sobre el capital capitalizado (127% del VN), pagos 30/6 y 31/12; amortiza en 20 cuotas de jun-24 a dic-33",
               "Par": "1,77% anual hasta mar-29 y 2,48% después, pagos 31/3 y 30/9; amortiza en 20 cuotas de sep-29 a dic-38",
               "Cuasipar": "3,31% anual sobre el capital capitalizado (138,86% del VN), pagos 30/6 y 31/12; amortiza en 20 cuotas de jun-36 a dic-45"}
BASE_CANJE = "2003-12-31"
# TAMAR: margen de la licitación en puntos de TNA. TMG27: ficha técnica de BYMA; el resto, comunicados de resultado de la
# Secretaría de Finanzas (TMF27 11/02/26, TMF28 15/04/26, TMG28 28/04/26, TML27 26/06/26)
TAMAR = [("TMF27", "2026-02-13", "2027-02-26", 6.5), ("TML27", "2026-06-30", "2027-07-30", 5.4), ("TMG27", "2026-04-01", "2027-08-31", 6.0),
         ("TMF28", "2026-04-20", "2028-02-25", 6.5), ("TMG28", "2026-04-30", "2028-08-31", 6.9)]
DUALES = [dict(t="TTD26", clase="fija", em="2025-01-29", v="2026-12-15", tem=.0214, m=0.0, cond="Máximo entre TEM fija 2,14% y TAMAR"),
          dict(t="TMVE8", clase="dolar", em="2026-07-31", v="2028-01-31", m=0.0, tc0=1499.8387, cond="Máximo entre USD al A3500 y USD al TC inicial capitalizado a TAMAR"),
          dict(t="TXMJ8", clase="cer", em="2026-05-15", v="2028-06-30", m=3.0, cond="Máximo entre capital ajustado por CER y TAMAR + 3%"),
          dict(t="TXMD8", clase="cer", em="2026-06-12", v="2028-12-15", m=3.0, cond="Máximo entre capital ajustado por CER y TAMAR + 3%"),
          dict(t="TXMJ9", clase="cer", em="2026-04-30", v="2029-06-29", m=3.0, cond="Máximo entre capital ajustado por CER y TAMAR + 3%"),
          dict(t="TXMD9", clase="cer", em="2026-06-12", v="2029-12-14", m=3.0, cond="Máximo entre capital ajustado por CER y TAMAR + 3%"),
          dict(t="TXMJ0", clase="cer", em="2026-06-12", v="2030-06-28", m=3.0, cond="Máximo entre capital ajustado por CER y TAMAR + 3%")]
DL = [("D30O6", "2026-08-14", "2026-10-30"), ("D30N6", "2026-09-15", "2026-11-30"), ("D15E7", "2026-07-31", "2027-01-15"),
      ("D31M7", "2026-05-29", "2027-03-31"), ("D10Y7", "2026-07-16", "2027-05-10"), ("TZV27", "2026-02-27", "2027-06-30"),
      ("TZV28", "2026-03-31", "2028-06-30"), ("TZVD8", "2026-06-12", "2028-12-15")]
# tickers con sufijo X: mismo ISIN que el base (verificado con la ficha técnica). Se usa el precio del ticker base y el del
# alias sólo si el base no operó (como hace Bonerío Local); el volumen es siempre el del ticker base
ALIAS = {"TZXD8": ["T2D8X"], "TXMD8": ["TXD8X"], "TZXD6": ["TXD6X"], "TZXM7": ["TXM7X"], "TZXS7": ["TXS7X"], "TZXO7": ["TXO7X"],
         "TZXD7": ["TXD7X"], "TZXM8": ["TXM8X"], "TZX28": ["TZX8X"], "TZXS8": ["TXS8X"], "TZXM9": ["TXM9X"], "TZXO6": ["TXO6X"],
         "TZXA7": ["TXA7X"], "TZXY7": ["TXY7X"], "TZX27": ["TZX7X"], "TX26": ["TX26X"], "TX28": ["TX28X"], "TX31": ["TX31X"],
         "T15E7": ["TE7X"], "T30A7": ["TA7X"], "T31Y7": ["TY7X"], "T30J7": ["TJ7X"], "TO26": ["TO26X"], "TY30P": ["TY30X"],
         "TMF27": ["TMF7X"], "TML27": ["TML7X"], "TMG27": ["TMG7X"], "TMF28": ["TMF8X"], "TMG28": ["TMG8X"], "TTD26": ["TTD6X"],
         "TXMJ8": ["TXJ8X"], "TXMJ9": ["TXJ9X"], "TXMD9": ["TXD9X"], "TXMJ0": ["TXJ0X"], "TMVE8": ["TVE8X"], "TZV27": ["TZV7X"],
         "TZV28": ["TZV8X"], "TZVD8": ["TVD8X"]}
TICKERS = ([f[0] for f in FIJA] + ["TO26", "TY30P"] + [c[0] for c in CERZ] + [b[0] for b in BONCER] + [x[0] for x in LARGOS] + [t[0] for t in TAMAR]
           + [d["t"] for d in DUALES] + [d[0] for d in DL])

# ── 3. Bonos nuevos: se agregan solos desde las licitaciones (licitaciones.instrumentos_nuevos) ────────────────
NUEVOS = {}   # ticker -> condiciones de los bonos que el Tesoro emitió después del respaldo y que se agregaron solos


def registrar_nuevos(lista):
    """Agrega a FIJA, CERZ, TAMAR y DL (y a TICKERS, para bajar su precio) los bonos nuevos que emitió el Tesoro, con las
    condiciones leídas del resultado de su licitación. Cada elemento es un dict con fam ('fija', 'cer', 'tamar' o 'dl'), t
    (ticker), venc y em (fechas ISO) y, según la familia, tem (TEM de emisión), fb (fecha del CER base) o m (margen TAMAR).
    Es idempotente: un ticker que ya está no se vuelve a agregar. Devuelve los tickers agregados en esta llamada."""
    agregados = []
    for x in lista or []:
        t, fam, v, em = x.get("t"), x.get("fam"), x.get("venc"), x.get("em")
        if not t or not v or not em or t in TICKERS:
            continue
        if fam == "fija" and x.get("tem"):
            FIJA.append((t, "LECAP" if t.startswith("S") else "BONCAP", em, v, float(x["tem"])))
        elif fam == "cer" and x.get("fb"):
            CERZ.append((t, "LECER" if t.startswith("X") else "BONCER", em, v, x["fb"]))
        elif fam == "tamar":
            TAMAR.append((t, em, v, x.get("m")))
        elif fam == "dl":
            DL.append((t, em, v))
        else:
            continue
        TICKERS.append(t); NUEVOS[t] = dict(x); agregados.append(t)
    return agregados


# ── 4. Carga de datos: BYMA y BCRA, con respaldo ───────────────────────────────────────────────────────────────
def _bcra(id_var, desde, hasta):
    r = requests.get(f"{BCRA_API}/{id_var}", params={"desde": desde, "hasta": hasta, "Limit": 3000}, timeout=20)
    r.raise_for_status()
    return pd.Series({pd.Timestamp(x["fecha"]): x["valor"] for x in r.json()["results"][0]["detalle"]}).sort_index()


def _panel(ruta, plazo):
    filtro = {"CI": {"T0": True, "T1": False, "T2": False}, "24HS": {"T0": False, "T1": True, "T2": False}}[plazo]
    r = requests.post(f"{BYMA_API}/{ruta}", json={"excludeZeroPxAndQty": False, "page_size": 5000, **filtro},
                      headers={"Content-Type": "application/json"}, verify=False, timeout=20)
    r.raise_for_status()
    j = r.json()
    return j["data"] if isinstance(j, dict) else j


def _precios_byma():
    out = {}
    for clave, plazo in (("t0", "CI"), ("t1", "24HS")):
        filas = {x["symbol"]: x for x in _panel("public-bonds", plazo) + _panel("lebacs", plazo) if x.get("denominationCcy") == "ARS"}
        operado = lambda x: (x.get("vwap") or x.get("trade") or 0) > 0
        for t in TICKERS:
            base = filas.get(t)
            cands = [filas[c] for c in [t] + ALIAS.get(t, []) if c in filas]
            if not cands:
                continue
            if base and operado(base):
                x = base
            else:
                alt = [c for c in cands if operado(c)]
                x = max(alt, key=lambda c: c.get("volumeAmount") or 0) if alt else (base or cands[0])
            px = x.get("vwap") or x.get("trade") or x.get("closingPrice") or x.get("previousClosingPrice")
            prev = (base or x).get("previousClosingPrice") or x.get("previousClosingPrice") or 0
            out.setdefault(t, {})[clave] = [px, x.get("trade") or 0, prev, x.get("bidPrice") or 0, x.get("offerPrice") or 0,
                                            ((base or {}).get("volumeAmount") or 0) / 1e6, px / prev - 1 if prev else 0, x["symbol"]]
    if len(out) < 30:
        raise RuntimeError("BYMA devolvió pocos bonos en pesos")
    return out


def _precios_data912():
    """Último precio operado de cada bono en pesos (data912): fines de semana, feriados o antes de la apertura BYMA devuelve
    el panel vacío. data912 da un solo precio, así que se usa el mismo para contado inmediato y 24 hs."""
    filas = {}
    for ruta in ("arg_bonds", "arg_notes"):
        r = requests.get(f"{DATA912}/{ruta}", headers=UA, timeout=(5, 12))
        r.raise_for_status()
        filas.update({x["symbol"]: x for x in r.json() if x.get("symbol")})
    out = {}
    for t in TICKERS:
        x = next((filas[c] for c in [t] + ALIAS.get(t, []) if float((filas.get(c) or {}).get("c") or 0) > 0), None)
        if x:
            px, var = float(x["c"]), float(x.get("pct_change") or 0) / 100
            fila = [px, px, px / (1 + var) if var > -1 else px, x.get("px_bid") or 0, x.get("px_ask") or 0, (x.get("v") or 0) / 1e6, var, x["symbol"]]
            out[t] = {"t0": fila, "t1": list(fila)}
    if len(out) < 30:
        raise RuntimeError("data912 devolvió pocos bonos en pesos")
    return out


def cargar(usar_api=True):
    """Precios (BYMA o data912), CER, TAMAR y A3500 (BCRA). Si una fuente falla, usa el respaldo del 09/10/2026 para esa parte."""
    R, hoy, fuente = RESPALDO, date.today(), {}
    try:
        assert usar_api
        px = _precios_byma(); fuente["precios"] = "API BYMA"; fecha = hoy
    except Exception:
        try:   # mercado cerrado: último precio de data912, para comprar el próximo día hábil
            assert usar_api
            px = _precios_data912(); fuente["precios"] = "data912 (último precio)"; fecha = ajustar_pago(hoy)
        except Exception:
            px = {t: v for t, v in R["px"].items()}; fuente["precios"] = "respaldo"; fecha = FECHA_RESPALDO
    try:
        assert usar_api
        cer = _bcra(30, "2020-08-01", (hoy + timedelta(days=40)).isoformat()); fuente["cer"] = "API BCRA"
        try:
            cer[pd.Timestamp(BASE_CANJE)] = float(_bcra(30, BASE_CANJE, BASE_CANJE).iloc[-1])
        except Exception:
            cer[pd.Timestamp(BASE_CANJE)] = R["cerBase"][BASE_CANJE]
        cer = cer.sort_index()
    except Exception:
        cer = pd.Series(R["cer"], index=pd.date_range(R["cer0"], periods=len(R["cer"])))
        for f, v in R["cerBase"].items():
            cer[pd.Timestamp(f)] = v
        cer = cer.sort_index(); fuente["cer"] = "respaldo"
    try:
        assert usar_api
        tamar = _bcra(44, "2024-12-01", hoy.isoformat()); fuente["tamar"] = "API BCRA"
    except Exception:
        tamar = pd.Series({pd.Timestamp(f): v for f, v in R["tamar"]}); fuente["tamar"] = "respaldo"
    try:
        assert usar_api
        tc = _bcra(5, (hoy - timedelta(days=20)).isoformat(), hoy.isoformat()); fuente["a3500"] = "API BCRA"
    except Exception:
        tc = pd.Series({pd.Timestamp(f): v for f, v in R["a3500"]}); fuente["a3500"] = "respaldo"
    return {"px": px, "cer": cer, "tamar": tamar, "a3500": tc, "fecha": fecha, "fuente": fuente,
            "ipc": dict(R["ipc"]), "rem": {r[0]: r[1] for r in R["rem"]}}


def _fechas_sem(desde, hasta, m1, m2):
    """Fin de mes de los meses m1 y m2 de cada año, entre dos fechas."""
    out = []
    for y in range(int(desde[:4]), int(hasta[:4]) + 1):
        for m in (m1, m2):
            f = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
            if D(desde) <= f <= D(hasta):
                out.append(f)
    return sorted(out)


# ── 5. Matemática: TIR, duración, curvas, tasas y CER ──────────────────────────────────────────────────────────
def _tir(flujos, s, P):
    pv = lambda y: sum(f / (1 + y) ** ((d - s).days / 365) for d, f in flujos if d > s) - P
    try:
        return brentq(pv, -0.9, 20)
    except ValueError:
        return np.nan


def _dur(flujos, s, y):
    v = [(((d - s).days / 365), f / (1 + y) ** ((d - s).days / 365)) for d, f in flujos if d > s]
    return sum(t * x for t, x in v) / sum(x for _, x in v)


def ajustar_ns(pts):
    """Nelson-Siegel: grilla en tau y mínimos cuadrados lineales en los betas (igual que la terminal web)."""
    if len(pts) < 3:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    mejor, tau = None, 0.08
    while tau <= 6:
        z = x / tau
        l = (1 - np.exp(-z)) / z
        A = np.column_stack([np.ones_like(x), l, l - np.exp(-z)])
        M = A.T @ A + np.diag([0, 1e-5, 1e-5])
        b = np.linalg.solve(M, A.T @ y)
        sse = float(((A @ b - y) ** 2).sum())
        if mejor is None or sse < mejor[0]:
            mejor = (sse, tau, b)
        tau *= 1.07
    _, tau, b = mejor

    def f(m):
        z = max(m, 0.01) / tau
        l = (1 - np.exp(-z)) / z
        return b[0] + b[1] * l + b[2] * (l - np.exp(-z))
    return f


def _interp(nodos, x):
    n = sorted(nodos)
    if not n:
        return np.nan
    if x <= n[0][0]:
        return n[0][1]
    if x >= n[-1][0]:
        return n[-1][1]
    for (x0, y0), (x1, y1) in zip(n, n[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return np.nan


MODELOS_CURVA = {"auto": "Automática (mejor ajuste)", "lineal": "Lineal", "log": "Logarítmica", "raiz": "Raíz cuadrada",
                 "cuad": "Cuadrática", "ns": "Nelson-Siegel", "interp": "Interpolada entre bonos"}


def ajustar_curva(pts, modelo="auto"):
    """Ajusta una curva tasa = f(años) a los puntos [(años, tasa)].

    Modelos: lineal (a + b·t), logarítmica (a + b·ln t), raíz cuadrada (a + b·√t), cuadrática (a + b·t + c·t²) y Nelson-Siegel.
    'auto' elige el de menor AIC = n·ln(SCE/n) + 2k: premia ajustar bien pero castiga cada parámetro de más, porque con
    pocos bonos el modelo más flexible siempre pasa más cerca de los puntos y predice peor entre ellos.
    'interp' une los puntos con rectas (sin ajuste). Devuelve (función, nombre, error medio en pb, {modelo: error en pb})."""
    pts = sorted((float(a), float(b)) for a, b in pts if np.isfinite(a) and np.isfinite(b))
    n = len(pts)
    if n == 0:
        return None, "", np.nan, {}
    x, y = np.array([max(a, 1e-4) for a, _ in pts]), np.array([b for _, b in pts])
    if modelo == "interp" or n < 3:
        return (lambda t: _interp(pts, t)), MODELOS_CURVA["interp"], 0.0, {}

    def mc(cols):
        b, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
        return b
    c = {}
    b = mc([np.ones(n), x]); c["lineal"] = (lambda t, b=b: b[0] + b[1] * t, 2)
    b = mc([np.ones(n), np.log(x)]); c["log"] = (lambda t, b=b: b[0] + b[1] * np.log(max(t, 1e-4)), 2)
    b = mc([np.ones(n), np.sqrt(x)]); c["raiz"] = (lambda t, b=b: b[0] + b[1] * np.sqrt(max(t, 0.0)), 2)
    if n >= 4:
        b = mc([np.ones(n), x, x ** 2]); c["cuad"] = (lambda t, b=b: b[0] + b[1] * t + b[2] * t * t, 3)
    if n >= 5:
        f = ajustar_ns(list(zip(x, y)))
        if f:
            c["ns"] = (f, 4)
    est = {}
    for k, (f, npar) in c.items():
        sce = float(sum((f(a) - v) ** 2 for a, v in zip(x, y)))
        est[k] = (np.sqrt(sce / n) * 1e4, n * np.log(sce / n + 1e-14) + 2 * npar)
    elegido = modelo if modelo in c else min(est, key=lambda k: est[k][1])
    return c[elegido][0], MODELOS_CURVA[elegido], est[elegido][0], {MODELOS_CURVA[k]: v[0] for k, v in est.items()}


tem_tamar = lambda tna: (1 + tna / 100 * 32 / 365) ** (365 / 32 / 12) - 1
tir_de = lambda pago, P, d: (pago / P) ** (365 / d) - 1 if P > 0 and d > 0 else np.nan
mens = lambda y: (1 + y) ** (30 / 365) - 1
tna30 = lambda y: ((1 + y) ** (30 / 365) - 1) * 365 / 30


def margen_tamar(pago, P, d, tamar_ref):
    """TNA x tal que la TAMAR TEM de x iguala el rendimiento mensual del bono; devuelve x − TAMAR supuesta (puntos de TNA)."""
    tm = (pago / P) ** (30 / d) - 1
    return ((1 + tm) ** (384 / 365) - 1) * 365 / 32 * 100 - tamar_ref


def _cer_en(cer, f):
    f = pd.Timestamp(f)
    return float(cer[cer.index <= f].iloc[-1])


@lru_cache(maxsize=4096)
def _ventanas(desde, hasta):
    """Exponentes del IPC de cada mes entre 'desde' (excluido) y 'hasta': ((mes de IPC, días / días de la ventana), ...)."""
    cuenta, f = {}, pd.Timestamp(desde)
    fin = pd.Timestamp(hasta)
    while f < fin:
        f += pd.Timedelta(days=1)
        ventana = pd.Period(f, "M") - (0 if f.day >= 16 else 1)
        k = (str(ventana - 1), ventana.days_in_month)
        cuenta[k] = cuenta.get(k, 0) + 1
    return tuple((m, n / dm) for (m, dm), n in cuenta.items())


def _proyectar_cer(cer, hasta, ipc):
    """CER proyectado con la regla del BCRA: del 16 del mes m al 15 del m+1 crece (1 + IPC de m-1)^(1/días del mes m).
    Se agrupan los días por mes de IPC (cacheado), así que cada proyección cuesta unas pocas multiplicaciones."""
    hasta = pd.Timestamp(hasta)
    if hasta <= cer.index[-1]:
        return _cer_en(cer, hasta)
    v = float(cer.iloc[-1])
    for m, e in _ventanas(cer.index[-1].date(), hasta.date()):
        v *= (1 + ipc(m)) ** e
    return v


# ── 6. Cálculo principal: calcular() valúa todos los bonos ─────────────────────────────────────────────────────
def dolar_esperado_futuros(tf, spot, fecha_futuros):
    """Dólar oficial (A3500) que esperan los futuros de dólar de A3 para cada fecha.
    tf: tabla de externos.devaluacion_futuros (columnas dias, ajuste y mensual), spot: A3500 del día de los futuros.
    Entre dos contratos se supone una devaluación diaria pareja; después del último contrato, el dólar sigue subiendo
    a la devaluación mensual promedio que descuenta ese último contrato."""
    xs = [0] + [int(x) for x in tf["dias"]]
    ys = [float(np.log(spot))] + [float(np.log(a)) for a in tf["ajuste"]]
    m_ult = float(tf["mensual"].iloc[-1])
    f0 = pd.Timestamp(fecha_futuros).date()

    def f(fecha):
        d = (pd.Timestamp(fecha).date() - f0).days
        if d <= xs[-1]:
            return float(np.exp(np.interp(max(d, 0), xs, ys)))
        return float(np.exp(ys[-1]) * (1 + m_ult) ** ((d - xs[-1]) / 30))
    f.ultimo_contrato = f0 + timedelta(days=xs[-1])
    f.mensual_despues = m_ult
    return f


def calcular(datos, plazo="CI", precio="px", tamar=None, infl=None, ipc_publicado=None, rem_mensual=None, dev=None, margenes=None, curva_be="auto",
             dolar_fut=None):
    """Devuelve un DataFrame por tipo de bono y los supuestos usados.
    plazo: 'CI' o '24HS'. precio: 'px' (VWAP), 'last' (último) o 'prev' (cierre anterior).
    tamar: TAMAR futura supuesta (TNA %, por defecto el promedio de las últimas 5 ruedas).
    infl: inflación mensual (decimal) después del último mes del REM.
    dev: devaluación mensual esperada (decimal) para dólar linked y TMVE8. Si es None se usa dolar_fut.
    dolar_fut: función fecha -> A3500 esperado (dolar_esperado_futuros, con los futuros de A3). Si también es None, se usa la
    devaluación implícita en la curva dólar linked (la que los empata con la tasa fija).
    margenes: dict ticker -> margen TAMAR (puntos de TNA) para reemplazar el de la licitación.
    curva_be: curva de tasa fija contra la que se despeja la inflación breakeven de cada CER: 'auto' (mejor ajuste, como
    Bonerío Local), un modelo de MODELOS_CURVA o 'interp' (rectas entre las LECAP/BONCAP)."""
    clave, idx = ("t0" if plazo == "CI" else "t1"), {"px": 0, "last": 1, "prev": 2}[precio]
    s = datos["fecha"] if plazo == "CI" else siguiente_habil(datos["fecha"])
    cer, tam, tc = datos["cer"], datos["tamar"].sort_index(), datos["a3500"].sort_index()
    tamar_hoy = float(tam.iloc[-5:].mean())
    tc_hoy, tc_fecha = float(tc.iloc[-1]), tc.index[-1].date()
    ipc_pub = {k: v / 100 for k, v in (ipc_publicado or datos["ipc"]).items()}
    rem = {k: v / 100 for k, v in (rem_mensual or datos["rem"]).items()}
    infl = infl if infl is not None else (list(rem.values())[-1] if rem else 0.017)
    tamar = tamar if tamar is not None else tamar_hoy
    ipc = lambda ym: ipc_pub.get(ym, rem.get(ym, infl))

    def q(t):
        r = datos["px"].get(t, {}).get(clave)
        if not r:
            return None
        p = r[idx] if r[idx] and r[idx] > 0 else (r[0] if r[0] and r[0] > 0 else r[2])
        return {"ticker": t, "precio": p, "var": r[6], "vol": r[5], "precio_de": r[7]} if p and p > 0 else None

    out = {k: [] for k, _, _ in TIPOS}
    flujos = {}  # ticker -> lista de pagos proyectados por 100 VN: {fecha, renta, capital, real (CER, en pesos de hoy), usd (dólar linked)}

    # tasa fija
    fija = {t: (tipo, D(em), D(v), [(ajustar_pago(D(v)), 100 * (1 + tem) ** (d360(D(em), D(v)) / 30))],
                f"TEM de emisión {tem:.2%} capitalizable") for t, tipo, em, v, tem in FIJA}
    fija["TO26"] = ("BONTE", D("2016-10-17"), D("2026-10-17"), [(ajustar_pago(D("2026-10-17")), 107.75)], "Tasa fija 15,5% anual, cupón semestral")
    ty, f = [], D("2025-11-30")
    while f <= D("2030-05-30"):
        ty.append((ajustar_pago(f), 14.75 + (100 if f == D("2030-05-30") else 0))); f = sumar_meses(f, 6)
    fija["TY30P"] = ("BONTE", D("2025-06-04"), D("2030-05-30"), ty, "Tasa fija 29,5% anual, cupones semestrales 30/5 y 30/11")
    for t, (tipo, em, v, fl, cond) in fija.items():
        r = q(t); fl = [x for x in fl if x[0] > s]
        if not r or not fl:
            continue
        y = _tir(fl, s, r["precio"]); d = (fl[-1][0] - s).days; mac = _dur(fl, s, y)
        cero = len(fl) == 1
        r.update(tipo=tipo, vence=v, dias=d, pago_final=fl[-1][1] if cero else np.nan, tna=(fl[-1][1] / r["precio"] - 1) * 365 / d if cero else np.nan,
                 tem=mens(y), tirea=y, md=mac / (1 + y), x=mac, condicion=cond)
        out["lecap" if tipo == "LECAP" else "boncap"].append(r)
        flujos[t] = [{"fecha": d_, "renta": f_ - (100 if (cero or d_ == fl[-1][0]) else 0), "capital": 100.0 if (cero or d_ == fl[-1][0]) else 0.0} for d_, f_ in fl]
    fit_fija = ajustar_ns([(r["x"], r["tirea"]) for r in out["lecap"] + out["boncap"] if r["dias"] >= 15])
    curva_fija = fit_fija or (lambda yrs: _interp([(r["x"], r["tirea"]) for r in out["lecap"] + out["boncap"]], yrs))
    for r in out["lecap"] + out["boncap"]:
        r["spread_pb"] = (r["tirea"] - curva_fija(r["x"])) * 1e4 if r["dias"] >= 30 else np.nan

    # CER
    ref = restar_habiles(s, 10); cref = _cer_en(cer, ref)
    cer_ult = cer.index[-1].date()
    # flujos reales por 100 VN: (fecha de pago, flujo, residual, inicio del período, tasa del período)
    cerb = {t: (tipo, D(em), D(v), _cer_en(cer, fb), [(ajustar_pago(D(v)), 100.0, 100.0, D(em), 0.0, 100.0)], f"Cupón cero; base CER {fb}", False)
            for t, tipo, em, v, fb in CERZ}
    for t, em, v, fb, tasa, p1, am0, n, cond in BONCER:
        fl, res, ini, f = [], 100.0, D(em), D(p1)
        while f <= D(v):
            am = 100 / n if f >= D(am0) else 0
            fl.append((ajustar_pago(f), res * tasa * d360(ini, f) / 360 + am, res, ini, tasa, am)); res -= am; ini = f; f = sumar_meses(f, 6)
        cerb[t] = ("BONCER", D(em), D(v), _cer_en(cer, fb), fl, cond, False)
    for t, tipo, v, (fd, fh, m1, m2), tasa_de, am0, capf in LARGOS:
        fechas = _fechas_sem(fd, fh, m1, m2) + ([D(v)] if tipo == "Par" else [])
        fl, res = [], 100 * capf
        for ini, f in zip(fechas, fechas[1:]):
            am = 5 * capf if f >= D(am0) else 0
            fl.append((ajustar_pago(f), res * tasa_de(ini) * d360(ini, f) / 360 + am, res, ini, tasa_de(ini), am)); res -= am
        cerb[t] = (tipo, D(BASE_CANJE), D(v), _cer_en(cer, BASE_CANJE), fl, f"{tipo} en pesos, canje 2005: {COND_LARGOS[tipo]}. Base CER 31/12/2003", True)
    # tasa fija cupón cero (LECAP y BONCAP) por días, para la breakeven
    cero = sorted((r["dias"], r["tirea"]) for r in out["lecap"] + out["boncap"] if not pd.isna(r["pago_final"]))
    max_cero = cero[-1][0] if cero else 0
    f_be, nombre_be, err_be, errores_be = ajustar_curva([(d / 365, y) for d, y in cero if d >= 15], curva_be)
    tasa_be = (lambda d: f_be(d / 365)) if f_be else (lambda d: _interp(cero, d))
    par_de = {r["vence"]: r for r in out["lecap"] + out["boncap"] if not pd.isna(r["pago_final"])}  # tasa fija cupón cero por vencimiento
    be_aux = []

    def despejar(vt, r1, pago_obj, fijos=None):
        """Inflación mensual pareja para los meses de IPC no publicados (y no fijados en 'fijos') que iguala el pago del CER con pago_obj."""
        pago_cer = lambda x: vt / cref * _proyectar_cer(cer, r1, lambda ym: ipc_pub.get(ym, (fijos or {}).get(ym, x)))
        lo, hi = -0.05, 0.3
        for _ in range(55):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if pago_cer(mid) > pago_obj else (mid, hi)
        return (lo + hi) / 2
    for t, (tipo, em, v, base, fl, cond, largo) in cerb.items():
        r = q(t); k = cref / base
        fut = [x for x in fl if x[0] > s]
        if not r or not fut:
            continue
        flx = [(d, f * k) for d, f, *_ in fut]
        cero_cupon = len(fut) == 1
        vt = fut[0][2] * (1 + fut[0][4] * d360(fut[0][3], s) / 360) * k
        y = _tir(flx, s, r["precio"]); mac = _dur(flx, s, y); dias = (flx[-1][0] - s).days
        hz = dias if cero_cupon else max(1, round(mac * 365))
        r1 = restar_habiles(s + timedelta(days=hz), 10)
        cp = _proyectar_cer(cer, r1, ipc)
        tir_nom = (1 + y) * (cp / cref) ** (365 / hz) - 1
        # Breakeven: inflación mensual pareja, para los meses de IPC no publicados, que iguala el bono con la tasa fija al mismo plazo
        d_fut = (r1 - cer_ult).days
        be, rem_m, nota = np.nan, np.nan, ""
        if d_fut < 15:
            # El CER que se usa para el pago (10 hábiles antes del vencimiento) ya está publicado: el bono paga una cantidad de pesos
            # conocida y no queda inflación por despejar. Se compara como tasa fija contra la letra del mismo día.
            nota = "CER ya publicado"
            if cero_cupon:
                tirea_c = ((vt / cref * cp) / r["precio"]) ** (365 / hz) - 1
                par = par_de.get(v)
                r.update(tirea_conocida=tirea_c, par=par["ticker"] if par is not None else None,
                         vs_par_pb=(tirea_c - par["tirea"]) * 1e4 if par is not None else np.nan)
        elif hz > max_cero:
            nota = "vence después de la LECAP/BONCAP más larga"
        elif not cero_cupon:
            nota = "bono con cupones"
        else:
            fija_be = tasa_be(hz)
            pago_fija = r["precio"] * (1 + fija_be) ** (hz / 365)
            be = despejar(vt, r1, pago_fija)
            par = par_de.get(v)
            pago_par = r["precio"] * (1 + par["tirea"]) ** (hz / 365) if par is not None else np.nan
            be_par = despejar(vt, r1, pago_par) if par is not None else np.nan
            r.update(fija_be=fija_be, par=par["ticker"] if par is not None else None, be_par=be_par)
            be_aux.append({"ticker": t, "vence": v, "r1": r1, "vt": vt, "precio": r["precio"], "hz": hz,
                           "pago_curva": pago_fija, "pago_par": pago_par, "par": par["ticker"] if par is not None else None})
        if d_fut >= 15:
            rem_m = (cp / float(cer.iloc[-1])) ** (30 / d_fut) - 1
        r.update(tipo=tipo, vence=v, dias=dias, valor_tecnico=vt, paridad=r["precio"] / vt, tir_real=y, tna_real=tna30(y), tna_nominal=tna30(tir_nom),
                 md=mac / (1 + y), x=mac, plazo_be=hz, be_mensual=be, rem_mensual=rem_m, be_rem_pb=(be - rem_m) * 1e4, be_nota=nota,
                 canje=largo, condicion=cond)
        out["cer"].append(r)
        flujos[t] = []
        for d_, f_, _res, _ini, _tasa, am in fut:
            kp = _proyectar_cer(cer, restar_habiles(d_, 10), ipc) / base
            flujos[t].append({"fecha": d_, "renta": (f_ - am) * kp, "capital": am * kp, "real": f_ * k})
    fit_cer = ajustar_ns([(r["x"], r["tir_real"]) for r in out["cer"] if r["dias"] >= 30])
    for r in out["cer"]:
        r["spread_pb"] = (r["tir_real"] - fit_cer(r["x"])) * 1e4 if fit_cer and r["dias"] >= 30 else np.nan

    # TAMAR (promedio de toda la vida del título; observado hasta hoy, supuesto constante después)
    obs = [(f.date(), float(v)) for f, v in tam.items()]
    ult = obs[-1][0]

    def tamar_prom(em, v, sup):
        a, b = restar_habiles(em, 10), restar_habiles(v, 10)
        vals = [x for f, x in obs if a <= f <= b]
        f, nr = max(a, ult + timedelta(days=1)), 0
        while f <= b:
            nr += habil(f); f += timedelta(days=1)
        hasta = sum(vals) / len(vals) if vals else sup
        return hasta, (sum(vals) + sup * nr) / ((len(vals) + nr) or 1)

    def pata_tamar(em, v, m, sup):
        hasta, total = tamar_prom(em, v, sup)
        return 100 * (1 + tem_tamar(total + m)) ** (d360(em, v) / 30), 100 * (1 + tem_tamar(hasta + m)) ** (d360(em, s) / 30), hasta

    def biseccion(fn, objetivo):
        lo, hi = 0.0, 150.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if fn(mid) > objetivo else (mid, hi)
        return (lo + hi) / 2

    for t, em, v, mg in TAMAR:
        r = q(t)
        if not r or ajustar_pago(D(v)) <= s:   # sin precio o ya vencido
            continue
        mg = (margenes or {}).get(t, mg)
        em, v = D(em), D(v); d = (ajustar_pago(v) - s).days; m = mg or 0.0
        pago, devengado, prom = pata_tamar(em, v, m, tamar)
        y = tir_de(pago, r["precio"], d); obj = curva_fija(d / 365)
        r.update(tipo="TAMAR", vence=v, dias=d, margen=mg, tamar_prom=prom, valor_tecnico=devengado, paridad=r["precio"] / devengado, tirea=y, tem=mens(y),
                 tamar_be=biseccion(lambda x: tir_de(pata_tamar(em, v, m, x)[0], r["precio"], d), obj), vs_fija_pb=(y - obj) * 1e4,
                 md=d / 365 / (1 + y), x=d / 365, margen_desconocido=mg is None, margen_s_tamar=margen_tamar(pago, r["precio"], d, tamar),
                 condicion="TAMAR promedio de toda la vida, capitalizable mensual" + (f" + margen {mg:g}% (licitación)" if mg is not None else "; sin margen (se toma 0)"))
        out["tamar"].append(r)
        flujos[t] = [{"fecha": ajustar_pago(v), "renta": pago - 100, "capital": 100.0}]

    # dólar linked
    for t, em, v in DL:
        r = q(t)
        if not r or ajustar_pago(D(v)) <= s:   # sin precio o ya vencido
            continue
        v = D(v); d = (ajustar_pago(v) - s).days
        tcbe = r["precio"] / 100 * (1 + curva_fija(d / 365)) ** (d / 365)
        y = tir_de(100 * tc_hoy, r["precio"], d)
        r.update(tipo="Dólar linked", vence=v, dias=d, precio_usd=r["precio"] / tc_hoy, tir_usd=y, dolar_be=tcbe,
                 deval_mensual=(tcbe / tc_hoy) ** (30 / d) - 1, deval_total=tcbe / tc_hoy - 1, md=d / 365 / (1 + y), x=d / 365,
                 condicion="Paga USD 100 en pesos al A3500 de 3 hábiles antes del vencimiento")
        out["dl"].append(r)
    devs = [(r["dias"], r["deval_mensual"]) for r in out["dl"] if r["dias"] >= 15]
    dev_de = lambda d: dev if dev is not None else (_interp(devs, d) if devs else 0.015)
    if dev is not None:
        fuente_dolar = f"devaluación elegida ({dev:.2%} por mes)"
    elif dolar_fut is not None:
        fuente_dolar = "futuros de dólar de A3"
    else:
        fuente_dolar = "devaluación implícita en la curva dólar linked"

    def dolar_al(v, d):
        """A3500 esperado al vencimiento: la devaluación elegida, el futuro de A3 o la implícita, en ese orden."""
        if dev is None and dolar_fut is not None:
            return dolar_fut(restar_habiles(v, 3))   # pagan al A3500 de 3 hábiles antes del vencimiento
        return tc_hoy * (1 + dev_de(d)) ** (d / 30)
    for r in out["dl"]:
        tc_fin = dolar_al(r["vence"], r["dias"])
        te = tir_de(100 * tc_fin, r["precio"], r["dias"]); fija_ = curva_fija(r["dias"] / 365)
        r.update(dolar_esperado=tc_fin, suba_esperada=tc_fin / tc_hoy - 1, deval_usada=(tc_fin / tc_hoy) ** (30 / r["dias"]) - 1,
                 extrapolado=bool(dev is None and dolar_fut is not None and restar_habiles(r["vence"], 3) > getattr(dolar_fut, "ultimo_contrato", date.max)),
                 tirea_esperada=te, tirea_fija=fija_, cobro_100k=100_000 * 100 * tc_fin / r["precio"],
                 vs_fija_pb=(te - fija_) * 1e4 if (dev is not None or dolar_fut is not None) else np.nan)
        flujos[r["ticker"]] = [{"fecha": ajustar_pago(r["vence"]), "renta": 0.0, "capital": 100 * tc_fin, "usd": 100.0, "tc": tc_fin}]

    # duales
    for b in DUALES:
        r = q(b["t"])
        if not r or ajustar_pago(D(b["v"])) <= s:   # sin precio o ya vencido
            continue
        em, v = D(b["em"]), D(b["v"]); d = (ajustar_pago(v) - s).days; n = d360(em, v) / 30
        pb, _, _ = pata_tamar(em, v, b["m"], tamar)
        if b["clase"] == "fija":
            pa, nombre = 100 * (1 + b["tem"]) ** n, "Tasa fija"; piso, piso_txt = tir_de(pa, r["precio"], d), "fija"
        elif b["clase"] == "cer":
            base = _cer_en(cer, restar_habiles(em, 10))
            pa, nombre = 100 * _proyectar_cer(cer, restar_habiles(v, 10), ipc) / base, "CER"
            piso, piso_txt = tir_de(100 * cref / base, r["precio"], d), "real"
        else:
            pa, nombre = 100 * dolar_al(v, d), "Dólar"
            pb *= b["tc0"]; piso, piso_txt = tir_de(100 * tc_hoy, r["precio"], d), "USD"
        k = b["tc0"] if b["clase"] == "dolar" else 1
        cruce = biseccion(lambda x: pata_tamar(em, v, b["m"], x)[0] * k, pa)
        pago = max(pa, pb); y = tir_de(pago, r["precio"], d)
        r.update(tipo="Dual", patas=f"{nombre}/TAMAR", vence=v, dias=d, pago_pata1=pa, pago_tamar=pb, paga=nombre if pa >= pb else "TAMAR",
                 tirea=y, piso=piso, piso_tipo=piso_txt, tamar_cruce=cruce, vs_fija_pb=(y - curva_fija(d / 365)) * 1e4, md=d / 365 / (1 + y),
                 margen_s_tamar=margen_tamar(pago, r["precio"], d, tamar), clase=b["clase"],
                 x=d / 365, condicion=b["cond"])
        out["dual"].append(r)
        flujos[b["t"]] = [{"fecha": ajustar_pago(v), "renta": 0.0, "capital": pago, "pata1": pa, "pata_tamar": pb}]

    res = {k: (pd.DataFrame(v).sort_values("vence").set_index("ticker") if v else pd.DataFrame()) for k, v in out.items()}
    res.update(curva_fija=curva_fija, curva_cer=fit_cer, liq=s, tamar_hoy=tamar_hoy, tamar_sup=tamar, infl_sup=infl, tc_hoy=tc_hoy, tc_fecha=tc_fecha,
               max_cero=max_cero, cer_ultimo=cer_ult, dev_sup=dev, fuente_dolar=fuente_dolar, curva_be=tasa_be, curva_be_nombre=nombre_be, curva_be_error_pb=err_be,
               curva_be_errores=errores_be, be_aux=be_aux, despejar=despejar, cref=cref, ipc_pub=ipc_pub, rem=rem,
               flujos={t: pd.DataFrame(v) for t, v in flujos.items() if v})
    return res


# ── 7. Inflación mes a mes que descuenta el mercado ────────────────────────────────────────────────────────────
def _dias_por_mes_ipc(desde, hasta):
    """Días del CER posteriores a 'desde' y hasta 'hasta' que crecen con el IPC de cada mes ('AAAA-MM' -> días)."""
    out, f = {}, pd.Timestamp(desde)
    while f < pd.Timestamp(hasta):
        f += pd.Timedelta(days=1)
        m = str(pd.Period(f, "M") - (0 if f.day >= 16 else 1) - 1)
        out[m] = out.get(m, 0) + 1
    return out


def trayectoria_be(res, anclas="mes_completo", contra="curva", min_meses=3, meses_publicados=3):
    """Sendero de inflación mensual que descuenta el mercado, despejado bono por bono ("breakeven despejado").

    Los CER cupón cero se ordenan por su fecha de referencia (vencimiento menos 10 hábiles). El primero fija una inflación
    pareja para los meses sin IPC publicado que cubre; el siguiente toma esos meses como dados y despeja solo los meses
    nuevos, y así sucesivamente: queda una inflación por tramos.
    anclas: 'mes_completo' (solo los CER cuya fecha de referencia cierra un mes entero de IPC, el día 15: cada tramo queda
    exactamente identificado, como en Chamomberg) o 'todos' (todos los CER; un mes que un bono cubre en menos de la mitad
    pasa al bono siguiente, y un bono que trae menos de min_meses meses nuevos cede sus meses al siguiente, salvo el último).
    contra: 'curva' (la curva de tasa fija ajustada al plazo del CER) o 'par' (solo los CER con una LECAP/BONCAP del mismo
    vencimiento, contra el precio de esa letra)."""
    ipc_pub, rem, despejar = res["ipc_pub"], res["rem"], res["despejar"]
    cer_ult = pd.Timestamp(res["cer_ultimo"])
    obj = lambda a: a["pago_par"] if contra == "par" else a["pago_curva"]
    fijos, filas = {}, []
    cands = []
    for a in sorted(res["be_aux"], key=lambda z: z["r1"]):
        if not np.isfinite(obj(a)):
            continue
        dias = _dias_por_mes_ipc(cer_ult, a["r1"])
        ult = max(dias)
        cubre = dias[ult] / (pd.Period(ult, "M") + 1).days_in_month
        if anclas == "mes_completo" and cubre < 0.97:
            continue
        cands.append((a, dias, ult, cubre))
    for i, (a, dias, ult, cubre) in enumerate(cands):
        faltan = [m for m in sorted(dias) if m not in ipc_pub and m not in fijos]
        if not faltan or (len(faltan) == 1 and cubre < 0.5):
            continue
        nuevos = faltan if (cubre >= 0.5 or len(faltan) == 1) else faltan[:-1]
        if anclas == "todos" and len(nuevos) < min_meses and i < len(cands) - 1:
            continue  # tramo corto: sus meses se despejan con el bono siguiente
        pi = despejar(a["vt"], a["r1"], obj(a), fijos)
        for m in nuevos:
            fijos[m] = pi
            filas.append({"mes": m, "inflacion": pi, "fuente": "Breakeven despejado",
                          "bono": a["ticker"] + (f" / {a['par']}" if contra == "par" else "")})
    publicados = sorted(ipc_pub)[-meses_publicados:]
    filas = [{"mes": m, "inflacion": ipc_pub[m], "fuente": "INDEC (publicado)", "bono": ""} for m in publicados] + filas
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df = df.set_index("mes").sort_index()
    df["tea"] = (1 + df["inflacion"]) ** 12 - 1
    df["rem"] = [rem.get(m, np.nan) for m in df.index]
    df["dif_rem_pb"] = (df["inflacion"] - df["rem"]) * 1e4
    return df
