"""
bondai.py — los datos básicos (BYMA, BCRA y REM) y los bonos soberanos en dólares.

Lo usan: la pestaña «Curva en dólares» y, a través de ella, «Mi portafolio», «Flujos de fondos» y el informe semanal.

Índice
  1. Bajar datos: series del BCRA, paneles de BYMA (o el último precio de data912 con el mercado cerrado) y el REM;
     cargar_datos() junta todo (o usa el respaldo del 09/10/2026)
  2. Fechas: días 30/360, sumar meses, días hábiles y feriados
  3. Flujos de fondos: cupones y amortizaciones de cada Bonar, Global y bono del Tesoro en dólares
  4. Métricas y curvas: TIR, duración, convexidad, valor técnico, paridad y curva Nelson-Siegel por legislación

Ejemplo:
    import bondai as B
    D = B.cargar_datos(usar_api=True)                 # precios, CER, IPC, tipo de cambio y REM
    tabla, liq, bonos = B.analisis_usd(D, "mep")      # una fila por bono con TIR, MD, convexidad, paridad y spread
"""
import base64
import json
import warnings
import zlib
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3
from scipy.optimize import brentq, least_squares

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

FECHA_ANALISIS = date(2026, 10, 9)  # día de los precios del respaldo; con API se usa la fecha de hoy
COLORES = {"bonar": "#2a78d6", "global": "#eb6834", "tesoro": "#1baf7a", "mercado": "#4a3aa7",
           "rem": "#1baf7a", "modelo": "#eb6834", "ipc": "#2a78d6", "gris": "#75839a"}
MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
mes_corto = lambda m: f"{MESES_ES[pd.Period(m, 'M').month - 1]}-{str(pd.Period(m, 'M').year)[2:]}"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


# Respaldo: datos reales del 09/10/2026 (BYMA + BCRA), comprimidos con zlib y codificados en base64.
RESPALDO_B64 = "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"
RESPALDO = json.loads(zlib.decompress(base64.b64decode(RESPALDO_B64)))
print("Respaldo:", RESPALDO["fecha_datos"], "·", ", ".join(RESPALDO["fuentes"].values())[:120], "…")


BCRA_API = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
DATA912 = "https://data912.com/live"   # último precio operado: sirve cuando BYMA vacía su panel (fines de semana, feriados, antes de abrir)
UA = {"User-Agent": "Mozilla/5.0 (Bond Analytics, trabajo académico)"}


# ── 1. Bajar datos: BCRA, BYMA y REM (con respaldo) ────────────────────────────────────────────────────────────
def bcra_serie(id_variable, desde, hasta):
    """Serie diaria o mensual del BCRA como pd.Series indexada por fecha."""
    filas, offset = [], 0
    while True:
        r = requests.get(f"{BCRA_API}/{id_variable}",
                         params={"desde": desde, "hasta": hasta, "Limit": 3000, "Offset": offset}, timeout=20)
        r.raise_for_status()
        detalle = r.json()["results"][0]["detalle"]
        filas += detalle
        if len(detalle) < 3000:
            break
        offset += 3000
    s = pd.Series({pd.Timestamp(x["fecha"]): x["valor"] for x in filas}).sort_index()
    return s


def bcra_mensual(id_variable, desde="2016-01-01", hasta=None, como="mean"):
    """Agrega una serie del BCRA a frecuencia mensual (índice 'AAAA-MM')."""
    hasta = hasta or date.today().isoformat()
    s = bcra_serie(id_variable, desde, hasta)
    m = s.groupby(s.index.to_period("M")).agg(como)
    m.index = m.index.astype(str)
    return m


def byma_panel(ruta, plazo="24HS"):
    """Panel de BYMA (public-bonds o lebacs). plazo: 'CI' (contado inmediato) o '24HS'."""
    filtro = {"CI": {"T0": True, "T1": False, "T2": False}, "24HS": {"T0": False, "T1": True, "T2": False}}[plazo]
    r = requests.post(f"{BYMA_API}/{ruta}", json={"excludeZeroPxAndQty": False, "page_size": 5000, **filtro},
                      headers={"Content-Type": "application/json"}, verify=False, timeout=20)
    r.raise_for_status()
    datos = r.json()
    df = pd.DataFrame(datos["data"] if isinstance(datos, dict) else datos)
    if df.empty or "symbol" not in df:
        raise ValueError("BYMA devolvió el panel vacío (fin de semana, feriado o antes de la apertura)")
    col = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else pd.Series(0.0, index=df.index)
    df["precio"] = col("vwap")                        # precio promedio del día; si no operó: último, cierre o cierre anterior
    for c in ("trade", "closingPrice", "previousClosingPrice"):
        df["precio"] = df["precio"].where(df["precio"] > 0, col(c))
    return df.set_index("symbol")


def data912_precios():
    """Último precio operado de cada bono y letra (data912). Lo guarda aunque el mercado esté cerrado."""
    filas = []
    for ruta in ("arg_bonds", "arg_notes"):
        r = requests.get(f"{DATA912}/{ruta}", headers=UA, timeout=(5, 12))
        r.raise_for_status()
        filas += r.json()
    s = pd.Series({x["symbol"]: float(x.get("c") or 0) for x in filas if x.get("symbol")})
    return s[s > 0]


def rem_mensual_bcra(anio, mes):
    """Medianas mensuales del REM leídas del Excel que publica el BCRA (hoja 'Cuadros de resultados')."""
    url = ("https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/informes/"
           f"relevamiento-expectativas-mercado-tablas-{anio}-{mes:02d}.xlsx")
    tabla = pd.read_excel(url, sheet_name="Cuadros de resultados", header=None)
    filas = tabla[tabla[2] == "var. % mensual"].head(7)  # primer bloque: IPC nivel general
    out = pd.DataFrame({"mes": pd.to_datetime(filas[1].astype(float), unit="D", origin="1899-12-30").dt.strftime("%Y-%m"),
                        "mediana": filas[3].astype(float), "p25": filas[10].astype(float), "p75": filas[9].astype(float)})
    return out.set_index("mes")


def cargar_datos(usar_api=True):
    """Devuelve un dict con todos los insumos. Si una API falla, usa el respaldo para esa parte."""
    d, R = {"fuente": {}}, RESPALDO
    # 1) macro mensual
    try:
        assert usar_api
        macro = pd.DataFrame({"ipc": bcra_mensual(27, como="last"), "rem12": bcra_mensual(29, como="last"),
                              "tc": bcra_mensual(5), "badlar": bcra_mensual(7), "base": bcra_mensual(15)})
        d["fuente"]["macro"] = "API BCRA"
    except Exception:
        macro = pd.DataFrame({k: pd.Series(dict(v)) for k, v in R["macro"].items()})
        d["fuente"]["macro"] = "respaldo"
    d["macro"] = macro.sort_index()
    # 2) CER diario
    try:
        assert usar_api
        cer = bcra_serie(30, "2020-08-01", (date.today() + timedelta(days=40)).isoformat())
        d["fuente"]["cer"] = "API BCRA"
    except Exception:
        cer = pd.Series(R["cer"]["valores"], index=pd.date_range(R["cer"]["desde"], periods=len(R["cer"]["valores"])))
        for f, v in R["cer_base"].items():
            cer[pd.Timestamp(f)] = v
        cer = cer.sort_index()
        d["fuente"]["cer"] = "respaldo"
    d["cer"] = cer
    # 3) precios de bonos: BYMA mientras opera el mercado; data912 (último precio) si BYMA tiene el panel vacío; si no, respaldo
    try:
        assert usar_api
        pb24, pbci, leci = byma_panel("public-bonds", "24HS"), byma_panel("public-bonds", "CI"), byma_panel("lebacs", "CI")
        assert pb24["precio"].get("AL30D", 0) > 0, "panel de BYMA incompleto"
        d["precios_usd"] = pb24["precio"].to_dict()      # incluye tickers D (MEP), C (cable) y en pesos
        d["precios_pesos"] = pd.concat([pbci["precio"], leci["precio"]]).to_dict()
        d["fuente"]["bonos"] = "API BYMA"
    except Exception:
        try:
            assert usar_api
            px = data912_precios()
            assert px.get("AL30D", 0) > 0, "data912 sin AL30D"
            d["precios_usd"] = d["precios_pesos"] = px.to_dict()
            d["fuente"]["bonos"] = "data912 (último precio)"
        except Exception:
            pu = {}
            for t, v in R["bonos_usd"].items():
                pu[t + "D"], pu[t + "C"], pu[t] = v["mep"], v["ccl"], v["ars"]
            d["precios_usd"], d["precios_pesos"] = pu, R["pesos"]
            d["fuente"]["bonos"] = "respaldo"
    # 4) REM mes a mes
    try:
        assert usar_api
        hoy = date.today()
        anio, mes = (hoy.year, hoy.month - 1) if hoy.month > 1 else (hoy.year - 1, 12)
        d["rem_mensual"] = rem_mensual_bcra(anio, mes)
        d["fuente"]["rem"] = "Excel BCRA"
    except Exception:
        d["rem_mensual"] = pd.DataFrame(R["rem_mensual"], columns=["mes", "mediana", "p25", "p75", "top10", "n", "nucleo"]).set_index("mes")
        d["fuente"]["rem"] = "respaldo"
    d["fecha"] = FECHA_ANALISIS if d["fuente"]["bonos"] == "respaldo" else date.today()
    return d


# ── 2. Fechas: 30/360, meses, días hábiles y feriados ──────────────────────────────────────────────────────────
def d360(a, b):
    """Días entre dos fechas con convención 30/360."""
    d1, d2 = min(a.day, 30), b.day
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (b.year - a.year) * 360 + (b.month - a.month) * 30 + (d2 - d1)


def sumar_meses(f, n):
    y, m = divmod(f.month - 1 + n, 12)
    y += f.year
    import calendar
    return date(y, m + 1, min(f.day, calendar.monthrange(y, m + 1)[1]))


FERIADOS = {date(2026, 1, 1), date(2026, 2, 16), date(2026, 2, 17), date(2026, 3, 23), date(2026, 3, 24), date(2026, 4, 2),
            date(2026, 4, 3), date(2026, 5, 1), date(2026, 5, 25), date(2026, 6, 15), date(2026, 7, 9), date(2026, 7, 10),
            date(2026, 8, 17), date(2026, 10, 12), date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 23), date(2026, 12, 7), date(2026, 12, 8), date(2026, 12, 25),
            date(2027, 1, 1), date(2027, 2, 8), date(2027, 2, 9), date(2027, 3, 24), date(2027, 3, 26), date(2027, 4, 2),
            date(2027, 5, 25), date(2027, 6, 21), date(2027, 7, 9), date(2027, 8, 16), date(2027, 10, 11)}
habil = lambda f: f.weekday() < 5 and f not in FERIADOS


def liquidacion_24hs(f):
    """Siguiente día hábil (para el plazo de 24 hs)."""
    f += timedelta(days=1)
    while f.weekday() >= 5 or f in FERIADOS:
        f += timedelta(days=1)
    return f


# ── 3. Flujos de fondos de los bonos en dólares ────────────────────────────────────────────────────────────────
def bono_2020(ticker):
    """Bonares y Globales del canje 2020: pagos 9/1 y 9/7, cupón step-up, amortización según ficha."""
    s = ticker[-2:]
    tramos = {"29": [(2020, .01)], "30": [(2020, .00125), (2021, .005), (2023, .0075), (2027, .0175)],
              "35": [(2020, .00125), (2021, .01125), (2022, .015), (2023, .03625), (2024, .04125), (2027, .0475), (2028, .05)],
              "38": [(2020, .00125), (2021, .02), (2022, .03875), (2023, .0425), (2024, .05)],
              "41": [(2020, .00125), (2021, .025), (2022, .035), (2029, .04875)],
              "46": [(2020, .00125), (2021, .01125), (2022, .015), (2023, .03625), (2024, .04125), (2027, .04375), (2028, .05)]}[s]
    venc = {"29": date(2029, 7, 9), "30": date(2030, 7, 9), "35": date(2035, 7, 9), "38": date(2038, 1, 9),
            "41": date(2041, 7, 9), "46": date(2046, 7, 9)}[s]
    amort_ini, n_cuotas = {"29": (date(2025, 1, 9), 10), "30": (date(2025, 1, 9), 12), "35": (date(2031, 1, 9), 10),
                           "38": (date(2027, 7, 9), 22), "41": (date(2028, 1, 9), 28), "46": (date(2025, 1, 9), 44)}[s]
    amort = {sumar_meses(amort_ini, 6 * i): (96 if s == "30" else 100) / n_cuotas for i in range(n_cuotas)}
    if s == "30":
        amort[date(2024, 7, 9)] = 4.0
    tasa = lambda ini: [t for (anio, t) in tramos if ini >= (date(2020, 9, 4) if anio == 2020 else date(anio, 7, 9))][-1]
    flujos, periodos, residual, ini, pago = [], [], 100.0, date(2020, 9, 4), date(2021, 7, 9)
    while pago <= venc:
        cup = residual * tasa(ini) * d360(ini, pago) / 360
        am = amort.get(pago, 0.0)
        flujos.append((pago, cup, am)); periodos.append((ini, pago, tasa(ini), residual))
        residual -= am; ini = pago; pago = sumar_meses(pago, 6)
    familia = "Bonares" if ticker.startswith("A") else "Globales"
    return {"ticker": ticker, "familia": familia, "flujos": flujos, "periodos": periodos, "venc": venc}


def bono_tesoro(ticker, emision, venc, tasa, mensual=True):
    """Bonos del Tesoro en USD emitidos en 2025-26: AO27/AO28/AO29 (6%, cupón mensual) y AN29 (6,5% semestral)."""
    import calendar
    fechas = []
    if mensual:
        y, m = emision.year, emision.month
        while (y, m) <= (venc.year, venc.month):
            ult = date(y, m, calendar.monthrange(y, m)[1] if m != 12 else 30)
            while ult.weekday() >= 5:
                ult -= timedelta(days=1)
            if emision < ult < venc:
                fechas.append(ult)
            m += 1
            if m > 12:
                y, m = y + 1, 1
        fechas.append(venc)
    else:
        f = date(2026, 5, 30)
        while f <= venc:
            fechas.append(f); f = sumar_meses(f, 6) if f.month == 5 else date(f.year + 1, 5, 30)
    flujos, periodos, ini = [], [], emision
    for i, f in enumerate(fechas):
        frac = (d360(ini, f) / 360) if (i == 0 or not mensual) else 1 / 12
        flujos.append((f, 100 * tasa * frac, 100.0 if f == venc else 0.0)); periodos.append((ini, f, tasa, 100.0)); ini = f
    return {"ticker": ticker, "familia": "Tesoro", "flujos": flujos, "periodos": periodos, "venc": venc}


def universo_usd():
    u = {t: bono_2020(t) for t in ["AL29", "AL30", "AL35", "AE38", "AL41", "GD29", "GD30", "GD35", "GD38", "GD41", "GD46"]}
    u["AO27"] = bono_tesoro("AO27", date(2026, 2, 27), date(2027, 10, 29), .06)
    u["AO28"] = bono_tesoro("AO28", date(2026, 3, 31), date(2028, 10, 31), .06)
    u["AO29"] = bono_tesoro("AO29", date(2026, 7, 16), date(2029, 10, 31), .06)
    u["AN29"] = bono_tesoro("AN29", date(2025, 12, 12), date(2029, 11, 30), .065, mensual=False)
    return u


# ── 4. Métricas y curvas: TIR, duración, convexidad, paridad y Nelson-Siegel ───────────────────────────────────
def metricas_bono(b, liq, precio):
    """TIR, MD, convexidad, valor técnico y paridad. Precio por 100 VN originales (sucio, como cotiza en BYMA)."""
    fl = [(f, c + a) for f, c, a in b["flujos"] if f > liq]
    t = np.array([(f - liq).days / 365 for f, _ in fl]); F = np.array([x for _, x in fl])
    pv = lambda y: float(np.sum(F / (1 + y) ** t))
    y = brentq(lambda y: pv(y) - precio, -0.5, 5.0)
    vp = F / (1 + y) ** t
    mac = float(np.sum(t * vp) / precio)
    residual = 100 - sum(a for f, _, a in b["flujos"] if f <= liq)
    ic = next((res * tasa * d360(ini, liq) / 360 for ini, fin, tasa, res in b["periodos"] if ini <= liq < fin), 0.0)
    cupon = next((res * tasa for ini, fin, tasa, res in b["periodos"] if ini <= liq < fin), 0.0)      # cupón anual vigente por 100 VN
    return {"precio": precio, "tir": y, "md": mac / (1 + y), "convexidad": float(np.sum(t * (t + 1) * vp) / (precio * (1 + y) ** 2)),
            "residual": residual, "ic": ic, "vt": residual + ic, "paridad": precio / (residual + ic), "current_yield": cupon / precio}


def nelson_siegel(m, b0, b1, b2, tau):
    x = np.maximum(m, 1e-3) / tau
    return b0 + b1 * (1 - np.exp(-x)) / x + b2 * ((1 - np.exp(-x)) / x - np.exp(-x))


def ajustar_ns(x, y):
    """Ajuste no lineal de Nelson-Siegel por mínimos cuadrados (scipy.optimize.least_squares)."""
    res = least_squares(lambda p: nelson_siegel(np.asarray(x), *p) - np.asarray(y), x0=[y.mean(), -0.02, 0.0, 1.0],
                        bounds=([-1, -1, -1, 0.05], [1, 1, 1, 15]))
    return res.x


def analisis_usd(datos, moneda="mep"):
    """Métricas de cada bono en dólar MEP (tickers D) o cable (tickers C), curvas Nelson-Siegel por legislación y spread contra la curva."""
    usd, liq = universo_usd(), liquidacion_24hs(datos["fecha"])
    suf = "C" if moneda == "ccl" else "D"
    filas = [{"ticker": t, "familia": b["familia"], **metricas_bono(b, liq, datos["precios_usd"][t + suf])}
             for t, b in usd.items() if datos["precios_usd"].get(t + suf)]
    tabla = pd.DataFrame(filas).set_index("ticker").sort_values(["familia", "md"])
    tabla["ley"] = np.where(tabla["familia"] == "Globales", "Nueva York", "Local")
    for ley in ["Local", "Nueva York"]:
        sub = tabla[tabla["ley"] == ley]
        p = ajustar_ns(sub["md"].values, sub["tir"].values)
        tabla.loc[sub.index, "tir_curva"] = nelson_siegel(sub["md"].values, *p)
        tabla.attrs[ley] = [float(v) for v in p]  # parámetros Nelson-Siegel (lista: pandas compara attrs al concatenar)
    tabla["spread_pb"] = (tabla["tir"] - tabla["tir_curva"]) * 1e4
    return tabla, liq, usd
