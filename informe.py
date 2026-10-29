"""
informe.py — informe semanal de renta fija argentina en PDF, para la pestaña «Informe semanal con IA» de Bond Analytics.

Qué hace:
  1. Baja los cierres diarios de BYMA (24 hs) de las últimas semanas de cada bono (o usa el respaldo del 09/10/2026).
  2. Toma la última rueda disponible y la de una semana antes, y calcula para cada bono la variación del precio y de su tasa
     (TIR en dólares para Bonares, Globales y Tesoro USD; TIREA para LECAP y BONCAP; TIR real para los CER cupón cero).
  3. Suma el riesgo país, los dólares (MEP, cable, A3500 y brecha), la TAMAR, la licitación de la semana, el dólar futuro, la
     inflación breakeven, las ONs de los grandes emisores y la agenda de las dos semanas siguientes.
  4. Si hay un portafolio cargado en «Mi portafolio», calcula cómo le fue en la semana (retorno, aporte de cada bono, precios) y
     sus indicadores: duración, DV01, convexidad, tipo de cartera, exposición y VaR (contexto_portafolio).
  5. Un LLM (Claude) redacta el resumen usando SOLO esos números, para alguien que no sabe de finanzas: frases cortas, pocos números
     y cada término explicado. Sin clave se usa una plantilla que sigue las mismas reglas.
  6. Arma un PDF fácil de leer: tarjetas con los números de la semana, el resumen con títulos, gráficos con «cómo leerlo», tu
     portafolio y un anexo con las tablas.

Uso desde la app:
    import informe as I
    S = I.semana(usar_api, cer, tamar, riesgo, dolares, licitaciones, extra)   # números de la semana
    ctx = {**S["contexto"], "portafolio": I.contexto_portafolio(C, R, H, U, S["ini"], S["fin"])}   # (opcional) tu cartera
    texto, origen = I.redactar(ctx)                                            # Claude o plantilla
    pdf = I.pdf(S, texto, origen, ctx)                                         # bytes del PDF
"""
import io
import json
import os
import re
import time
from datetime import date, datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bondai as B
import pesos as P
import portafolio as PF

AZUL, ROJO, VERDE, GRIS, VIOLETA = "#2a78d6", "#d64545", "#1baf7a", "#8a96aa", "#7a5cf0"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


# ───────────────────────────── datos ─────────────────────────────
def instrumentos():
    """{ticker: (grupo, símbolo en BYMA)}: los bonos en dólares se miden en dólar MEP (símbolo terminado en D)."""
    out = {}
    for t, b in B.universo_usd().items():
        out[t] = ("Tesoro USD" if b["familia"] == "Tesoro" else b["familia"], t + "D")
    for t, tipo, em, v, tem in P.FIJA:
        out[t] = (tipo, t)
    for t, tipo, em, v, fb in P.CERZ:
        out[t] = ("CER", t)
    for t, em, v, mg in P.TAMAR:
        out[t] = ("TAMAR", t)
    for d in P.DUALES:
        out[d["t"]] = ("Dual", d["t"])
    for t, em, v in P.DL:
        out[t] = ("Dólar linked", t)
    return out


def historias(simbolos, usar_api=True, dias=45):
    """Cierres diarios por símbolo. En vivo los baja de a uno de BYMA (las últimas semanas); lo que falle sale del respaldo."""
    H, fuente = {}, {}
    ses = None
    if usar_api:
        import requests
        ses = requests.Session(); ses.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://open.bymadata.com.ar/"})
    hasta = date.today(); desde = hasta - timedelta(days=dias)
    fallos = 0
    for s in simbolos:
        serie = pd.Series(dtype=float)
        if ses is not None and fallos < 6:
            try:
                serie = PF._historico(ses, s, desde, hasta)
            except Exception:
                fallos += 1
            time.sleep(0.08)
        if len(serie) >= 3:
            H[s], fuente[s] = serie, "BYMA"
        else:
            H[s], fuente[s] = PF._respaldo(s), "respaldo"
    return H, fuente


def _valor(serie, f):
    s = serie[serie.index <= pd.Timestamp(f)]
    return float(s.iloc[-1]) if len(s) else np.nan


def _tasa(t, grupo, px, f, cer, U):
    """Tasa del bono comprado a 'px' el día f (liquida en 24 hs): TIR en dólares, TIREA de la tasa fija o TIR real del CER."""
    if not (px > 0):
        return np.nan, np.nan
    s = P.siguiente_habil(f)
    try:
        if t in U:
            m = B.metricas_bono(U[t], s, px)
            return m["tir"], m["md"]
        x = next((x for x in P.FIJA if x[0] == t), None)
        if x:
            em, v, tem = date.fromisoformat(x[2]), date.fromisoformat(x[3]), x[4]
            d = (P.ajustar_pago(v) - s).days
            y = (100 * (1 + tem) ** (P.d360(em, v) / 30) / px) ** (365 / d) - 1 if d > 0 else np.nan
            return y, d / 365 / (1 + y)
        x = next((x for x in P.CERZ if x[0] == t), None)
        if x:
            v, fb = date.fromisoformat(x[3]), date.fromisoformat(x[4])
            d = (P.ajustar_pago(v) - s).days
            vt = 100 * _valor(cer, P.restar_habiles(s, 10)) / _valor(cer, fb)
            y = (vt / px) ** (365 / d) - 1 if d > 0 else np.nan
            return y, d / 365 / (1 + y)
    except Exception:
        pass
    return np.nan, np.nan


def ventana(H, ref=("AL30D", "GD30D", "S29E7", "TZX27")):
    """Última rueda con datos y la rueda de una semana antes (la última con fecha ≤ fin − 7 días)."""
    fin = max((H[r].index.max() for r in ref if r in H and len(H[r])), default=None)
    if fin is None:
        raise RuntimeError("No hay precios para armar la semana")
    base = next(H[r] for r in ref if r in H and len(H[r]))
    previas = base[base.index <= fin - pd.Timedelta(days=7)]
    ini = previas.index.max() if len(previas) else base.index.min()
    return ini.date(), fin.date()


def tabla_bonos(H, inst, ini, fin, cer):
    """Una fila por bono con precio y tasa al inicio y al fin de la semana. Se descartan los bonos sin precio reciente."""
    U = B.universo_usd()
    filas = []
    for t, (grupo, sym) in inst.items():
        s = H.get(sym)
        if s is None or not len(s) or s.index.max() < pd.Timestamp(fin) - pd.Timedelta(days=3):
            continue
        p0, p1 = _valor(s, ini), _valor(s, fin)
        if not (p0 > 0 and p1 > 0):
            continue
        y0, _ = _tasa(t, grupo, p0, ini, cer, U)
        y1, md = _tasa(t, grupo, p1, fin, cer, U)
        filas.append({"ticker": t, "grupo": grupo, "precio_ini": p0, "precio_fin": p1, "var": p1 / p0 - 1, "tasa_ini": y0, "tasa_fin": y1,
                      "var_pb": (y1 - y0) * 1e4 if np.isfinite(y0) and np.isfinite(y1) else np.nan, "md": md})
    return pd.DataFrame(filas).set_index("ticker") if filas else pd.DataFrame()


def tabla_ons(H, ons, ini, fin):
    """Precio y TIR al inicio y al fin de la semana de las ONs más operadas.
    ons: [(ticker, emisor, vencimiento ISO, cupón anual, pagos por año, cuotas, símbolo, volumen US$)]."""
    import externos as X
    filas = []
    for t, emisor, venc, cupon, frec, cuotas, sym, vol in ons:
        s = H.get(sym)
        if s is None or not len(s) or s.index.max() < pd.Timestamp(fin) - pd.Timedelta(days=3):
            continue
        p0, p1 = _valor(s, ini), _valor(s, fin)
        if not (p0 > 0 and p1 > 0):
            continue
        v = date.fromisoformat(venc)
        tir = lambda px, f: X.tir_on(px, X.flujos_on(v, cupon, frec, cuotas, P.siguiente_habil(f)), P.siguiente_habil(f))[0]
        y0, y1 = tir(p0, ini), tir(p1, fin)
        filas.append({"ticker": t, "emisor": emisor, "precio_ini": p0, "precio_fin": p1, "var": p1 / p0 - 1, "tir_ini": y0, "tir_fin": y1,
                      "var_pb": (y1 - y0) * 1e4 if np.isfinite(y0) and np.isfinite(y1) else np.nan, "volumen_usd": vol})
    return pd.DataFrame(filas).set_index("ticker") if filas else pd.DataFrame()


def semana(usar_api, cer, tamar, riesgo, dolares, licitaciones=None, extra=None, ons=()):
    """Todo lo que entra en el informe: tabla de bonos, indicadores al inicio y al fin, resúmenes por grupo y el contexto
    JSON para el LLM. dolares: DataFrame de dolares.series(); licitaciones: (subastas, colocaciones) de licitaciones.actualizar;
    ons: las ONs más operadas (ver tabla_ons)."""
    inst = instrumentos()
    H, fuente = historias(sorted({s for _, s in inst.values()} | {o[6] for o in ons}), usar_api)
    ini, fin = ventana(H)
    T = tabla_bonos(H, inst, ini, fin, cer)
    TO = tabla_ons(H, ons, ini, fin)
    en_vivo = sum(v == "BYMA" for v in fuente.values())
    # indicadores
    ind = {}
    agrega = lambda k, nombre, a, b, unidad, dec=2: ind.__setitem__(k, {"nombre": nombre, "inicio": a, "fin": b, "unidad": unidad, "dec": dec})
    if riesgo is not None and len(riesgo):
        agrega("riesgo_pais", "Riesgo país (pb)", _valor(riesgo, ini), _valor(riesgo, fin), "pb", 0)
    if dolares is not None and len(dolares):
        for k, n in [("mep", "Dólar MEP ($)"), ("cable", "Dólar cable ($)"), ("a3500", "Dólar oficial A3500 ($)")]:
            if k in dolares:
                agrega(k, n, _valor(dolares[k].dropna(), ini), _valor(dolares[k].dropna(), fin), "$")
        if "mep" in dolares and "a3500" in dolares:
            b0 = _valor(dolares["mep"].dropna(), ini) / _valor(dolares["a3500"].dropna(), ini) - 1
            b1 = _valor(dolares["mep"].dropna(), fin) / _valor(dolares["a3500"].dropna(), fin) - 1
            agrega("brecha", "Brecha MEP vs oficial (%)", b0 * 100, b1 * 100, "pp")
    agrega("tamar", "TAMAR (TNA %)", _valor(tamar, ini), _valor(tamar, fin), "pp")
    grupos = {}
    if len(T):
        for g, x in T.groupby("grupo"):
            par = x.dropna(subset=["tasa_ini", "tasa_fin"])
            x = x.assign(tasa_ini=x["tasa_ini"].where(x.index.isin(par.index)), tasa_fin=x["tasa_fin"].where(x.index.isin(par.index)))
            grupos[g] = {"bonos": int(len(x)), "var_precio_mediana_%": round(float(x["var"].median()) * 100, 2),
                         "tasa_mediana_ini_%": round(float(x["tasa_ini"].median()) * 100, 2) if x["tasa_ini"].notna().any() else None,
                         "tasa_mediana_fin_%": round(float(x["tasa_fin"].median()) * 100, 2) if x["tasa_fin"].notna().any() else None,
                         "var_tasa_mediana_pb": round(float(x["var_pb"].median()), 0) if x["var_pb"].notna().any() else None}
        sob = T[T["grupo"].isin(["Bonares", "Globales"])]
        if len(sob):
            agrega("tir_soberanos", "TIR mediana Bonares y Globales (%)", sob["tasa_ini"].median() * 100, sob["tasa_fin"].median() * 100, "pp")
        lec = T[T["grupo"].isin(["LECAP", "BONCAP"])]
        if len(lec):
            agrega("tirea_fija", "TIREA mediana LECAP y BONCAP (%)", lec["tasa_ini"].median() * 100, lec["tasa_fin"].median() * 100, "pp")
        cz = T[T["grupo"] == "CER"]
        if len(cz) and cz["tasa_fin"].notna().any():
            agrega("tir_cer", "TIR real mediana CER (%)", cz["tasa_ini"].median() * 100, cz["tasa_fin"].median() * 100, "pp")
    # licitaciones de la semana
    lic = None
    if licitaciones is not None:
        sub, col = licitaciones
        s_ = sub[(sub["f"] > ini) & (sub["f"] <= fin)]
        if len(s_):
            u = s_.iloc[-1]
            c_ = col[col["fecha"] == u["f"]]
            lic = {"fecha": f"{u['f']:%d/%m/%Y}", "ofertado_billones": float(u["ofer"]), "adjudicado_billones": float(u["adj"]),
                   "instrumentos": [{"ticker": r["ticker"], "tipo": r["familia"], "adjudicado_millones": None if pd.isna(r["valor_efectivo"]) else float(r["valor_efectivo"]),
                                     "tirea_corte_%": None if pd.isna(r["tirea_corte"]) else round(float(r["tirea_corte"]) * 100, 2),
                                     "premio_pb": None if pd.isna(r["premio_pb"]) else round(float(r["premio_pb"]))} for _, r in c_.iterrows()]}
    var = lambda k: round(ind[k]["fin"] - ind[k]["inicio"], 2) if k in ind and np.isfinite(ind[k]["fin"]) and np.isfinite(ind[k]["inicio"]) else None
    ctx = {"semana": {"desde": f"{ini:%d/%m/%Y}", "hasta": f"{fin:%d/%m/%Y}", "precios": "BYMA en vivo" if en_vivo > len(fuente) / 2 else "respaldo del 09/10/2026"},
           "indicadores": {k: {"inicio": round(float(v["inicio"]), v["dec"]) if np.isfinite(v["inicio"]) else None,
                               "fin": round(float(v["fin"]), v["dec"]) if np.isfinite(v["fin"]) else None, "variacion": var(k), "nombre": v["nombre"]}
                           for k, v in ind.items()},
           "por_grupo": grupos}
    if len(T):
        orden = T.sort_values("var")
        ctx["mayores_subas_%"] = {t: round(float(v) * 100, 2) for t, v in orden["var"].iloc[::-1].head(5).items()}
        ctx["mayores_bajas_%"] = {t: round(float(v) * 100, 2) for t, v in orden["var"].head(5).items()}
        usd = T[T["grupo"].isin(["Bonares", "Globales", "Tesoro USD"])].dropna(subset=["var_pb"])
        if len(usd):
            ctx["usd_mas_comprimio_pb"] = {t: round(float(v)) for t, v in usd["var_pb"].sort_values().head(3).items()}
            ctx["usd_mas_se_amplio_pb"] = {t: round(float(v)) for t, v in usd["var_pb"].sort_values().tail(3)[::-1].items()}
    if lic:
        ctx["licitacion_de_la_semana"] = lic
    if len(TO):
        ctx["obligaciones_negociables"] = {
            "ons_medidas": int(len(TO)), "var_precio_mediana_%": round(float(TO["var"].median()) * 100, 2),
            "tir_mediana_ini_%": round(float(TO["tir_ini"].median()) * 100, 2), "tir_mediana_fin_%": round(float(TO["tir_fin"].median()) * 100, 2),
            "var_tir_mediana_pb": round(float(TO["var_pb"].median())) if TO["var_pb"].notna().any() else None,
            "mas_operadas": {t: {"emisor": r["emisor"], "precio": round(float(r["precio_fin"]), 2), "var_%": round(float(r["var"]) * 100, 2),
                                 "tir_%": round(float(r["tir_fin"]) * 100, 2) if np.isfinite(r["tir_fin"]) else None,
                                 "var_tir_pb": round(float(r["var_pb"])) if np.isfinite(r["var_pb"]) else None}
                             for t, r in TO.sort_values("volumen_usd", ascending=False).head(8).iterrows()},
            "mejor_semana": [TO["var"].idxmax(), round(float(TO["var"].max()) * 100, 2)],
            "peor_semana": [TO["var"].idxmin(), round(float(TO["var"].min()) * 100, 2)]}
        sob = ctx["por_grupo"]
        tir_sob = [sob[g]["tasa_mediana_fin_%"] for g in ("Globales", "Bonares") if g in sob and sob[g]["tasa_mediana_fin_%"] is not None]
        if tir_sob:
            ctx["obligaciones_negociables"]["spread_vs_soberanos_pb"] = round((float(TO["tir_fin"].median()) * 100 - float(np.median(tir_sob))) * 100)
    ctx.update(extra or {})
    return {"ini": ini, "fin": fin, "tabla": T, "ons": TO, "indicadores": ind, "grupos": grupos, "licitacion": lic, "contexto": ctx, "historias": H,
            "riesgo": riesgo, "fuente_precios": ctx["semana"]["precios"]}


# ───────────────────────────── lo que se sumó: ONs de los grandes emisores, agenda y tu portafolio ─────────────────────────────
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_r = lambda v, d=2: None if v is None or not np.isfinite(v) else round(float(v), d)


def contexto_emisores(ONS_S, emisores):
    """TIR mediana y spread contra la curva de Globales de las ONs de cada gran emisor (herramientas.spread_ons)."""
    if ONS_S is None or ONS_S.empty:
        return None
    out = {}
    for e in emisores:
        x = ONS_S[ONS_S["emisor"] == e].dropna(subset=["tir"])
        if len(x):
            out[e] = {"ons": int(len(x)), "tir_mediana_%": _r(x["tir"].median() * 100), "spread_vs_globales_pb": _r(x["spread_pb"].median(), 0),
                      "la_mas_operada": x["volumen_usd"].idxmax()}
    return out or None


def agenda(desde, hasta, ons=()):
    """Lo que viene entre 'desde' y 'hasta': licitaciones, feriados, pagos y vencimientos del Tesoro (pesos y dólares) y, aparte,
    los pagos de las ONs. ons: como en licitaciones.eventos [(ticker, emisor, clase, flujos, vencimiento, frecuencia supuesta)].
    Las fechas de las ONs se estiman desde el vencimiento con la frecuencia de la ficha de BYMA y pasan al día hábil siguiente."""
    import licitaciones as L
    EV = L.eventos(desde, hasta)
    EV = EV[EV["tipo"].isin(["lla", "lic", "liq", "fer", "ren", "amo", "ven"])]
    G = L.agrupar(EV)
    tesoro = [{"fecha": f"{r.fecha:%d/%m/%Y}", "dia": DIAS[r.fecha.weekday()], "que": L.TIPOS_EV[r.tipo], "instrumentos": r.ticker,
               "detalle": r.detalle} for r in G.itertuples()]
    pagos = []
    for t, emisor, clase, flujos, venc, supuesta in ons:
        for f, cup, am in flujos:
            pago = L.siguiente_habil(f)
            if desde <= pago <= hasta:
                pagos.append({"fecha_": pago, "fecha": f"{pago:%d/%m/%Y}", "dia": DIAS[pago.weekday()], "ticker": t, "emisor": emisor,
                              "clase": clase or "", "que": "vencimiento" if f >= venc else "amortización y renta" if am > 0 else "renta",
                              "renta_usd_por_100": round(float(cup), 3), "capital_usd_por_100": round(float(am), 2),
                              "frecuencia_supuesta": bool(supuesta)})
    pagos.sort(key=lambda x: (x["fecha_"], x["emisor"], x["ticker"]))
    for x in pagos:
        del x["fecha_"]
    return {"desde": f"{desde:%d/%m/%Y}", "hasta": f"{hasta:%d/%m/%Y}", "tesoro_y_soberanos": tesoro, "pagos_de_ons": pagos,
            "emisores_que_pagan": sorted({x["emisor"] for x in pagos}),
            "nota_ons": "fechas de las ONs estimadas desde el vencimiento con la frecuencia de la ficha de BYMA (semestral si no la informa), "
                        "corridas al día hábil siguiente; montos en US$ por 100 VN"}


def historia_cartera(H_largo, simbolos, usar_api=True):
    """Historia de los bonos de la cartera para el informe: la larga de la pestaña «Mi portafolio» (portafolio.historicos, que
    puede ser el respaldo del 09/10/2026) completada con los cierres de las últimas semanas (historias), así cubre la semana."""
    rec, _ = historias(simbolos, usar_api)
    out = {}
    for s in simbolos:
        x = pd.concat([(H_largo or {}).get(s, pd.Series(dtype=float)), rec.get(s, pd.Series(dtype=float))])
        out[s] = x[~x.index.duplicated(keep="last")].sort_index()
    return out


def contexto_portafolio(C, R, Hh, U, ini, fin, moneda="ARS", tabla=None, ons=None):
    """Cómo le fue a tu cartera (pestaña «Mi portafolio») en la semana del informe y qué dicen sus indicadores.
    C: portafolio.cartera; R: portafolio.retornos (retornos diarios de cada bono con los cupones sumados, en la moneda de
    valuación); Hh: portafolio.historicos (cierres por símbolo); U: el universo de bonos; tabla y ons: las tablas de semana(),
    de donde sale la tasa de cada bono al inicio y al final de la semana. Los pesos son los de hoy.
    Las claves que empiezan con «_» son para el PDF (no van al modelo de lenguaje)."""
    if C is None or C.empty:
        return None
    RS, TC = PF.resumen(C), PF.tipo_cartera(C)
    mon = "US$" if moneda == "USD" else "$"
    i0, f0 = pd.Timestamp(ini), pd.Timestamp(fin)
    R = R if R is not None else pd.DataFrame()
    cols = [t for t in C.index if t in R.columns]
    rp = (R[cols].fillna(0) * C.loc[cols, "peso"]).sum(axis=1) if cols else pd.Series(dtype=float)   # retorno diario de la cartera
    Rw = R[(R.index > i0) & (R.index <= f0)] if len(R) else R
    rpw = rp[(rp.index > i0) & (rp.index <= f0)]
    ret = float((1 + rpw).prod() - 1) if len(rpw) else np.nan
    bonos = []
    for t, r in C.sort_values("valor", ascending=False).iterrows():
        px = (Hh or {}).get(U[t]["simbolo"], pd.Series(dtype=float))
        p0, p1 = (_valor(px.dropna(), ini), _valor(px.dropna(), fin)) if len(px) else (np.nan, np.nan)
        rt = float((1 + Rw[t].dropna()).prod() - 1) if t in Rw and Rw[t].notna().any() else np.nan
        y0 = y1 = np.nan
        if tabla is not None and len(tabla) and t in tabla.index:
            y0, y1 = tabla.loc[t, "tasa_ini"], tabla.loc[t, "tasa_fin"]
        elif ons is not None and len(ons) and t in ons.index:
            y0, y1 = ons.loc[t, "tir_ini"], ons.loc[t, "tir_fin"]
        dy = (y1 - y0) * 1e4 if np.isfinite(y0) and np.isfinite(y1) else np.nan
        bonos.append({"ticker": t, "grupo": PF.GRUPOS.get(r["grupo"], r["grupo"]), "peso_%": _r(r["peso"] * 100, 1), "md": _r(r["md"]),
                      "precio_ini": _r(p0), "precio_fin": _r(p1), "var_precio_%": _r((p1 / p0 - 1) * 100) if p0 and np.isfinite(p0) else None,
                      "retorno_semana_%": _r(rt * 100), "aporte_pp": _r(r["peso"] * rt * 100), "cambio_tasa_pb": _r(dy, 0)})
    con_tasa = [b for b in bonos if b["cambio_tasa_pb"] is not None]
    peso_tasa = sum(b["peso_%"] for b in con_tasa) / 100
    efecto = sum(-b["md"] * b["cambio_tasa_pb"] / 100 * b["peso_%"] / 100 for b in con_tasa) if con_tasa else None
    dy_med = sum(b["cambio_tasa_pb"] * b["peso_%"] for b in con_tasa) / sum(b["peso_%"] for b in con_tasa) if con_tasa else None
    por_grupo = {}
    for b in bonos:
        g = por_grupo.setdefault(b["grupo"], {"peso_%": 0.0, "aporte_pp": 0.0})
        g["peso_%"] += b["peso_%"] or 0; g["aporte_pp"] += b["aporte_pp"] or 0
    for g in por_grupo.values():
        g["retorno_%"] = _r(g["aporte_pp"] / g["peso_%"] * 100) if g["peso_%"] else None
        g["peso_%"], g["aporte_pp"] = _r(g["peso_%"], 1), _r(g["aporte_pp"])
    ordenados = sorted([b for b in bonos if b["aporte_pp"] is not None], key=lambda b: b["aporte_pp"])
    var = None
    if len(cols) and len(R) >= 60:
        try:
            v = PF.var_historico(R[cols].fillna(0), C.loc[cols, "valor"], 0.95, 1, 500)
            var = {"var_95_1_dia": _r(v["var"], 0), "var_%": _r(v["var_pct"] * 100), "perdida_promedio_peor_5%": _r(v["es"], 0),
                   "volatilidad_anual_%": _r(v["vol_anual"] * 100, 1), "ruedas": int(v["ruedas"])}
        except Exception:
            var = None
    dias_sem = [{"fecha": f"{d:%d/%m}", "retorno_%": _r(x * 100)} for d, x in rpw.items()]
    serie = (1 + rp[(rp.index > f0 - pd.Timedelta(days=92)) & (rp.index <= f0)]).cumprod() * 100 if len(rp) else pd.Series(dtype=float)
    out = {"moneda": mon, "valor_hoy": _r(RS["valor"], 0), "bonos": len(C),
           "retorno_semana_%": _r(ret * 100), "resultado_aprox": _r(RS["valor"] * ret / (1 + ret), 0) if np.isfinite(ret) else None,
           "dias_de_la_semana": dias_sem,
           "mejor_bono": [ordenados[-1]["ticker"], ordenados[-1]["retorno_semana_%"]] if ordenados else None,
           "peor_bono": [ordenados[0]["ticker"], ordenados[0]["retorno_semana_%"]] if ordenados else None,
           "por_grupo": por_grupo, "bonos_detalle": bonos,
           "indicadores": {"duracion_modificada": _r(RS["md"]), "dv01": _r(RS["dv01"], 0), "convexidad": _r(RS["convexidad"], 1),
                           "plazo_promedio_anios": _r(RS["plazo_dias"] / 365), "tasa_por_grupo_%": {PF.GRUPOS.get(k, k): _r(v * 100) for k, v in RS["tir_por_grupo"].items()},
                           "mayor_exposicion": [PF.GRUPOS.get(RS["mayor_grupo"], RS["mayor_grupo"]), _r(RS["mayor_peso"] * 100, 0)],
                           "bonos_efectivos": _r(RS["efectivos"], 1), "tipo_de_cartera": [TC["tipo"], TC["explicacion"]],
                           "riesgo": var},
           "tasas_en_la_semana": {"cambio_promedio_pb": _r(dy_med, 0), "peso_con_dato_%": _r(peso_tasa * 100, 0),
                                  "efecto_de_la_duracion_pp": _r(efecto), "resto_pp": _r(ret * 100 - efecto) if efecto is not None and np.isfinite(ret) else None},
           "como_leerlo": ("retornos con los pesos de hoy y los cupones cobrados; efecto_de_la_duracion_pp ≈ −MD × cambio de tasa de cada bono "
                           "(cuánto del resultado explican las tasas); resto_pp = devengamiento, cupones, tipo de cambio y lo que la duración no capta; "
                           "dv01: cuánto pierde la cartera si todas las tasas suben 1 pb; VaR 95% a 1 día: la pérdida diaria que solo se superó "
                           "en 1 de cada 20 ruedas del período"),
           "_serie": [[f"{d:%Y-%m-%d}", round(float(v), 3)] for d, v in serie.items()], "_ini": f"{ini:%Y-%m-%d}", "_fin": f"{fin:%Y-%m-%d}"}
    return out


def _para_llm(ctx):
    """El contexto sin las claves internas del PDF (las que empiezan con «_»)."""
    limpio = lambda d: {k: (limpio(v) if isinstance(v, dict) else v) for k, v in d.items() if not str(k).startswith("_")}
    return limpio(ctx)


# ───────────────────────────── texto ─────────────────────────────
# Para quien no sabe de finanzas: frases cortas, pocos números por párrafo y cada término explicado la primera vez.
PROMPT = ("Sos analista de renta fija en Argentina y escribís un informe semanal para personas que NO saben de finanzas: tiene que leerse "
          "fácil y de corrido. Español rioplatense, claro y cálido, con frases cortas. Usás SOLO los números del JSON (no inventes datos ni "
          "noticias, ni expliques causas que no surjan de los números) y redondeás: porcentajes con un decimal, montos sin decimales. Cada "
          "párrafo empieza con un título corto seguido de dos puntos, en este orden: 'La semana en una línea', 'El dólar y el riesgo país', "
          "'Bonos del Estado en dólares', 'Bonos de empresas (ONs)', 'Bonos en pesos', 'Tu portafolio' y 'Qué dicen los indicadores de tu "
          "portafolio' (estos dos solo si el JSON trae portafolio), 'Lo que viene' y 'Para tener en cuenta'. Reglas: como máximo tres números "
          "por párrafo (en los dos del portafolio, hasta cinco); la primera vez que aparece un término técnico, explicalo en pocas palabras "
          "(por ejemplo: riesgo país = cuánto más que Estados Unidos le cobra el mercado a la Argentina para prestarle; TIR = la ganancia por "
          "año si tenés el bono hasta el final; cuando el precio de un bono sube, su tasa baja); contá primero qué pasó y qué significa, y "
          "recién después el número; no nombres un bono por su código sin decir qué es. 'La semana en una línea' resume en una o dos frases si "
          "fue una buena o mala semana para los bonos. En 'Tu portafolio' contá cuánto ganó o perdió la cartera en la semana (en porcentaje y "
          "en plata) y qué grupo de bonos lo explicó, con el bono que más ayudó y el que menos. En 'Qué dicen los indicadores de tu portafolio' "
          "explicá con ejemplos simples la duración (qué pasa si las tasas suben 1 punto), cuánto cambiaron esta semana las tasas de sus bonos "
          "y cuánto del resultado explica eso (efecto_de_la_duracion_pp), el VaR (cuánto podría perder en un día malo) y cómo está armada la "
          "cartera (tipo_de_cartera). En 'Lo que viene' mencioná la licitación del Tesoro (la de la semana si hubo y la próxima), los "
          "vencimientos del Tesoro más importantes de agenda_proximos_dias y cuántas ONs pagan intereses, nombrando dos o tres empresas y "
          "aclarando que esas fechas son estimadas. 'Para tener en cuenta' cierra en dos frases con los riesgos y aclara que no es una "
          "recomendación de inversión. Sin markdown, sin viñetas, sin tablas.")


def texto_de(msj):
    """Junta los bloques de texto de la respuesta. Los modelos nuevos pueden devolver primero un bloque de razonamiento
    ('thinking'), que no tiene .text: por eso no alcanza con msj.content[0].text."""
    return "\n\n".join(b.text for b in msj.content if getattr(b, "type", "") == "text" and getattr(b, "text", "")).strip()


def redactar(ctx, modelo="claude-sonnet-5-5"):
    """Texto del informe: Claude si hay ANTHROPIC_API_KEY; si no (o si falla), la plantilla. Devuelve (texto, origen)."""
    clave = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if clave:
        try:
            import anthropic
            msj = anthropic.Anthropic(api_key=clave).messages.create(
                model=modelo, max_tokens=4000, system=PROMPT,
                messages=[{"role": "user", "content": "Datos de la semana (JSON):\n" + json.dumps(_para_llm(ctx), ensure_ascii=False)}])
            texto = texto_de(msj)
            if not texto:
                raise RuntimeError(f"respuesta sin texto (stop_reason={getattr(msj, 'stop_reason', '?')})")
            return texto, f"redactado por Claude ({modelo})"
        except Exception as e:
            return plantilla(ctx), f"plantilla (Claude no respondió: {type(e).__name__}: {str(e)[:120]})"
    return plantilla(ctx), "plantilla automática (sin clave de Anthropic)"


def _n(v, d=2):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _s(v, d=2):
    """Con signo y coma decimal: +0,49."""
    return ("+" if v >= 0 else "−") + _n(abs(v), d)


MES_LARGO = {"ene": "enero", "feb": "febrero", "mar": "marzo", "abr": "abril", "may": "mayo", "jun": "junio", "jul": "julio", "ago": "agosto",
             "sep": "septiembre", "oct": "octubre", "nov": "noviembre", "dic": "diciembre"}
GRUPO_LLANO = {"Hard dollar": "los bonos del Estado en dólares", "Obligaciones negociables": "las ONs (bonos de empresas)",
               "Tasa fija": "las letras a tasa fija", "CER": "los bonos que ajustan por inflación (CER)", "TAMAR": "los bonos TAMAR",
               "Duales": "los bonos duales", "Dólar linked": "los bonos atados al dólar oficial"}
TIPO_LLANO = {"Bullet": "concentrada: casi todo vence cerca de la misma fecha",
              "Barbell": "tiene bonos que vencen pronto y bonos muy largos, y casi nada en el medio",
              "Ladder": "escalonada: los vencimientos están repartidos a lo largo del tiempo"}


def plantilla(c):
    """Resumen sin LLM, con los mismos números y las mismas reglas que el prompt: frases cortas, pocos números y cada término
    técnico explicado la primera vez."""
    I, G = c["indicadores"], c["por_grupo"]
    val = lambda k, campo="fin": (I.get(k) or {}).get(campo)
    var = lambda k: (I.get(k) or {}).get("variacion")
    p = []
    rp_v = var("riesgo_pais")
    usd = [G[g]["var_precio_mediana_%"] for g in ("Bonares", "Globales") if g in G]
    usd_v = float(np.mean(usd)) if usd else None
    mep_ini, mep_fin = val("mep", "inicio"), val("mep")
    mep_v = (mep_fin / mep_ini - 1) * 100 if mep_ini and mep_fin else None
    # 1. la semana en una línea
    if rp_v is not None and usd_v is not None:
        tono = "una buena semana" if (rp_v < 0 and usd_v > 0) else "una mala semana" if (rp_v > 0 and usd_v < 0) else "una semana mixta"
        t = (f"La semana en una línea: fue {tono} para los bonos argentinos. El riesgo país {'bajó' if rp_v < 0 else 'subió'} y los "
             f"bonos del Estado en dólares {'subieron' if usd_v >= 0 else 'bajaron'} cerca de {_n(abs(usd_v), 1)}%")
        if mep_v is not None:
            t += ", mientras el dólar casi no se movió" if abs(mep_v) < 1 else f", y el dólar {'subió' if mep_v > 0 else 'bajó'} {_n(abs(mep_v), 1)}%"
        p.append(t + ".")
    # 2. el dólar y el riesgo país
    t = "El dólar y el riesgo país: "
    if rp_v is not None:
        t += (f"el riesgo país, que mide cuánto más que Estados Unidos le cobra el mercado a la Argentina para prestarle plata, "
              f"{'bajó' if rp_v < 0 else 'subió'} {abs(rp_v):.0f} puntos y quedó en {val('riesgo_pais'):.0f}. "
              + ("Es una señal de más confianza. " if rp_v < 0 else "Es una señal de más desconfianza. " if rp_v > 0 else ""))
    if mep_fin:
        t += f"El dólar MEP, el que se compra con bonos en la bolsa, terminó en $ {_n(mep_fin, 0)}"
        br = val("brecha")
        if br is not None:
            t += f", apenas {_n(br, 1)}% más caro que el oficial" if br < 5 else f", {_n(br, 1)}% más caro que el oficial"
        t += ". "
    fu = c.get("dolar_futuro")
    if fu:
        mes = MES_LARGO.get(str(fu.get("contrato", ""))[-5:-2].lower(), "")
        cuando = "fin de año" if mes == "diciembre" else f"fin de {mes}" if mes else "los próximos meses"
        t += f"Para {cuando}, el mercado de futuros espera un dólar oficial de $ {_n(fu['precio'], 0)}."
    p.append(t.strip())
    # 3. bonos del Estado en dólares
    bon, glo = G.get("Bonares"), G.get("Globales")
    if bon or glo:
        partes = ([f"los Bonares (ley argentina) {_n(abs(bon['var_precio_mediana_%']), 1)}%"] if bon else []) + \
                 ([f"los Globales (ley de Nueva York) {_n(abs(glo['var_precio_mediana_%']), 1)}%"] if glo else [])
        t = f"Bonos del Estado en dólares: en la semana {'subieron' if (usd_v or 0) >= 0 else 'bajaron'} " + " y ".join(partes) + ". "
        ts = I.get("tir_soberanos")
        if ts and ts.get("inicio") is not None and ts.get("fin") is not None:
            sube = (usd_v or 0) >= 0
            t += (f"Cuando el precio de un bono {'sube' if sube else 'baja'}, su rendimiento {'baja' if sube else 'sube'}: la TIR, que es la ganancia por "
                  f"año si tenés el bono hasta que vence, pasó de {_n(ts['inicio'], 1)}% a {_n(ts['fin'], 1)}%.")
        p.append(t.strip())
    # 4. ONs
    on = c.get("obligaciones_negociables")
    if on:
        v = on["var_precio_mediana_%"]
        t = ("Bonos de empresas (ONs): las obligaciones negociables son los bonos que emiten las empresas para financiarse. Las más operadas "
             + ("casi no se movieron" if abs(v) < 0.5 else f"{'subieron' if v > 0 else 'bajaron'} {_n(abs(v), 1)}%")
             + f" y rinden en promedio {_n(on['tir_mediana_fin_%'], 1)}% por año en dólares")
        sp = on.get("spread_vs_soberanos_pb")
        if sp is not None:
            t += (", menos que los bonos del Estado: al mercado le parece más seguro prestarle a las grandes empresas que al país."
                  if sp < 0 else ", más que los bonos del Estado.")
        else:
            t += "."
        ge = {k: v_ for k, v_ in (c.get("ons_grandes_emisores") or {}).items() if v_.get("tir_mediana_%") is not None}
        if len(ge) >= 2:
            a = max(ge, key=lambda k: ge[k]["tir_mediana_%"]); b = min(ge, key=lambda k: ge[k]["tir_mediana_%"])
            t += f" Entre las grandes, la que más rinde es {a} ({_n(ge[a]['tir_mediana_%'], 1)}%) y la que menos, {b} ({_n(ge[b]['tir_mediana_%'], 1)}%)."
        p.append(t)
    # 5. bonos en pesos
    fija = [G[g]["var_precio_mediana_%"] for g in ("LECAP", "BONCAP") if g in G]
    tf = I.get("tirea_fija")
    if fija or tf:
        t = "Bonos en pesos: las letras a tasa fija (LECAP y BONCAP), que pagan un monto en pesos conocido de antemano, "
        if fija:
            m_ = float(np.mean(fija))
            t += f"{'subieron' if m_ >= 0 else 'bajaron'} cerca de {_n(abs(m_), 1)}%"
        if tf and tf.get("fin") is not None:
            t += (" y hoy rinden" if fija else "rinden") + f" alrededor de {_n(tf['fin'], 1)}% por año"
        t += ". "
        be, rem = c.get("breakeven_proximos_meses_%") or {}, c.get("rem_mismos_meses_%") or {}
        if be:
            m, v = next(iter(be.items()))
            mes = MES_LARGO.get(m[:3], m)
            t += f"Comparándolas con los bonos que se ajustan por inflación (CER), el mercado espera una inflación de {_n(v, 1)}% por mes para {mes}"
            r = rem.get(m)
            if r is not None:
                t += (f", parecida a la que esperan los economistas del REM ({_n(r, 1)}%)" if abs(v - r) < 0.15 else
                      f", {'más' if v > r else 'menos'} que la que esperan los economistas del REM ({_n(r, 1)}%)")
            t += "."
        p.append(t.strip())
    # 6 y 7. tu portafolio
    pf = c.get("portafolio")
    if pf and pf.get("retorno_semana_%") is not None:
        m_ = pf["moneda"]; r_ = pf["retorno_semana_%"]
        t = f"Tu portafolio: tu cartera {'ganó' if r_ >= 0 else 'perdió'} {_n(abs(r_), 1)}% en la semana"
        if pf.get("resultado_aprox") is not None:
            t += f", unos {m_} {_n(abs(pf['resultado_aprox']), 0)}"
        t += ". "
        grupos = {k: v for k, v in pf["por_grupo"].items() if v.get("retorno_%") is not None}
        if grupos:
            g = max(grupos, key=lambda k: abs(grupos[k]["aporte_pp"] or 0))
            x = grupos[g]
            t += (f"La mayor parte vino de {GRUPO_LLANO.get(g, g)}, que son el {_n(x['peso_%'], 0)}% de la cartera y "
                  f"{'subieron' if x['retorno_%'] >= 0 else 'bajaron'} {_n(abs(x['retorno_%']), 1)}%. ")
        if pf.get("mejor_bono") and pf.get("peor_bono") and pf["mejor_bono"][0] != pf["peor_bono"][0]:
            t += f"El bono que más ayudó fue el {pf['mejor_bono'][0]} ({_s(pf['mejor_bono'][1], 1)}%) y el que menos, el {pf['peor_bono'][0]} ({_s(pf['peor_bono'][1], 1)}%)."
        p.append(t.strip())
        ix, ts = pf["indicadores"], pf["tasas_en_la_semana"]
        md = ix["duracion_modificada"]
        t = (f"Qué dicen los indicadores de tu portafolio: la duración es {_n(md, 1)}. Quiere decir que si las tasas del mercado suben 1 punto, "
             f"la cartera pierde cerca de {_n(md, 1)}%, y si bajan, gana lo mismo. ")
        if ts.get("cambio_promedio_pb") is not None and ts.get("efecto_de_la_duracion_pp") is not None:
            d = ts["cambio_promedio_pb"] / 100; ef = ts["efecto_de_la_duracion_pp"]
            t += (f"Esta semana las tasas de tus bonos {'bajaron' if d < 0 else 'subieron'} {_n(abs(d), 2)} puntos en promedio, y eso solo explica "
                  f"{_n(abs(ef), 1)} puntos de lo que {'ganaste' if ef >= 0 else 'perdiste'}. ")
        rg = ix.get("riesgo")
        if rg:
            t += (f"El VaR mide cuánto podrías perder en un día malo: solo 1 de cada 20 días la cartera perdió más de {m_} {_n(rg['var_95_1_dia'], 0)} "
                  f"({_n(rg['var_%'], 1)}% de su valor). ")
        tipo, expl = ix["tipo_de_cartera"]
        t += f"Por cómo están repartidos los vencimientos, es una cartera «{tipo}»: {TIPO_LLANO.get(tipo, expl.rstrip('.'))}."
        p.append(t.strip())
    # 8. lo que viene
    lic, ag = c.get("licitacion_de_la_semana"), c.get("agenda_proximos_dias")
    t = "Lo que viene: "
    if lic:
        t += (f"esta semana el Tesoro hizo una licitación (vendió bonos nuevos para financiarse) y consiguió $ {_n(lic['adjudicado_billones'], 1)} "
              "billones")
    else:
        t += "esta semana no hubo licitación del Tesoro (cuando el Estado vende bonos nuevos para financiarse)"
    t += f"; la próxima es el {c['proxima_licitacion'].split(' ')[0][:5]}. " if c.get("proxima_licitacion") else ". "
    if ag:
        venc = [x for x in ag["tesoro_y_soberanos"] if x["que"] == "Vencimiento" and x["instrumentos"]]
        tk = [z.strip() for x in venc for z in x["instrumentos"].split(",")]
        po = ag.get("pagos_de_ons") or []
        frases = []
        if tk:
            frases.append(f"vencen {len(tk)} bonos del Tesoro" + (f" ({', '.join(tk[:3])})" if len(tk) <= 3 else ""))
        if po:
            em = list(dict.fromkeys(x["emisor"] for x in po))[:3]
            frases.append(f"{len(po)} ONs pagan intereses, entre ellas las de {', '.join(em[:-1]) + ' y ' + em[-1] if len(em) > 1 else em[0]} "
                          "(fechas estimadas)")
        if frases:
            t += "En las próximas dos semanas " + " y ".join(frases) + "."
    p.append(t.strip())
    p.append("Para tener en cuenta: una semana es poco tiempo, y los precios pueden cambiar rápido con el dólar, la inflación o las noticias. Las "
             "tasas de los bonos incluyen un premio por el riesgo, así que no son pronósticos. Es una herramienta académica, no una recomendación "
             "de inversión.")
    return "\n\n".join(p)


# ───────────────────────────── gráficos ─────────────────────────────
def _png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def graficos(S):
    """Figuras del informe (como PNG en memoria): movimientos de precio, curva en dólares, tasa fija y riesgo país."""
    T, ini, fin = S["tabla"], S["ini"], S["fin"]
    out = {}
    if len(T):
        x = T.sort_values("var")
        x = pd.concat([x.head(8), x.tail(8)]).drop_duplicates()
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.barh([f"{t} · {g}" for t, g in zip(x.index, x["grupo"])], x["var"] * 100, color=[VERDE if v >= 0 else ROJO for v in x["var"]])
        ax.axvline(0, color=GRIS, lw=.8); ax.set_xlabel("Cambio de precio en la semana (%)"); ax.grid(alpha=.25, axis="x")
        ax.tick_params(axis="y", labelsize=7)
        out["movimientos"] = _png(fig)
        u = T[T["grupo"].isin(["Bonares", "Globales", "Tesoro USD"])].dropna(subset=["tasa_fin", "md"])
        if len(u):
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            colores = {"Bonares": AZUL, "Globales": VIOLETA, "Tesoro USD": VERDE}
            for g, z in u.groupby("grupo"):
                ax.scatter(z["md"], z["tasa_ini"] * 100, facecolors="none", edgecolors=colores.get(g, GRIS), s=30)
                ax.scatter(z["md"], z["tasa_fin"] * 100, color=colores.get(g, GRIS), s=30, label=g)
                for t, r in z.iterrows():
                    ax.annotate("", (r["md"], r["tasa_fin"] * 100), (r["md"], r["tasa_ini"] * 100), arrowprops={"arrowstyle": "->", "color": colores.get(g, GRIS), "lw": .8})
                    ax.annotate(t, (r["md"], r["tasa_fin"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
            ax.set_xlabel("Duración (cuanto más a la derecha, más largo el bono)"); ax.set_ylabel("Rendimiento por año (TIR) %"); ax.grid(alpha=.25); ax.legend(fontsize=7)
            ax.set_title(f"Círculo vacío: hace una semana ({ini:%d/%m}) · lleno: ahora ({fin:%d/%m})", fontsize=8.5, loc="left", color="#4A5063")
            out["curva_usd"] = _png(fig)
        f_ = T[T["grupo"].isin(["LECAP", "BONCAP"])].dropna(subset=["tasa_fin", "md"])
        f_ = f_[f_["md"] * (1 + f_["tasa_fin"]) * 365 >= 15]                     # sin las que vencen en días: su tasa es puro ruido
        if len(f_) >= 3:
            f_ = f_.sort_values("md").assign(dias=lambda z: z["md"] * (1 + z["tasa_fin"]) * 365)
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.plot(f_["dias"], f_["tasa_ini"] * 100, color=GRIS, lw=1, ls="--", marker="o", ms=3, mfc="white", label=f"Curva del {ini:%d/%m}")
            ax.plot(f_["dias"], f_["tasa_fin"] * 100, color=AZUL, lw=1.4, alpha=.6, label=f"Curva del {fin:%d/%m}")
            for g, mk, col in (("LECAP", "o", AZUL), ("BONCAP", "s", VIOLETA)):
                z = f_[f_["grupo"] == g]
                if len(z):
                    ax.scatter(z["dias"], z["tasa_fin"] * 100, marker=mk, s=34, color=col, zorder=3,
                               label="LECAP (letras, hasta 1 año)" if g == "LECAP" else "BONCAP (bonos, más de 1 año)")
            for i, (t, r) in enumerate(f_.iterrows()):          # etiqueta de cada instrumento, alternando arriba y abajo para que no se pisen
                arriba = r["tasa_fin"] >= r["tasa_ini"] if i % 2 == 0 else r["tasa_fin"] < r["tasa_ini"]
                ax.annotate(f"{t}\n{r['tasa_fin'] * 100:.1f}%".replace(".", ","), (r["dias"], r["tasa_fin"] * 100),
                            xytext=(0, 9 if arriba else -18), textcoords="offset points", ha="center", fontsize=6.3, color="#141A2E",
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": .85})
            lo, hi = np.nanmin(f_[["tasa_ini", "tasa_fin"]].values) * 100, np.nanmax(f_[["tasa_ini", "tasa_fin"]].values) * 100
            ax.set_ylim(lo - 1.2, hi + 1.2)
            ax.set_xlabel("Días que faltan para el vencimiento"); ax.set_ylabel("Rendimiento por año (TIREA) %"); ax.grid(alpha=.25); ax.legend(fontsize=6.5, loc="lower right")
            out["tasa_fija"] = _png(fig)
            out["tasa_fija_tabla"] = f_
    r = S.get("riesgo")
    if r is not None and len(r):
        z = r[r.index >= pd.Timestamp(fin) - pd.Timedelta(days=95)]
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(z.index, z.values, color=AZUL, lw=1.6)
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=AZUL, alpha=.1, label="semana del informe")
        ax.set_ylabel("Puntos básicos"); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper left")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        out["riesgo"] = _png(fig)
    return out


# ───────────────────────────── PDF ─────────────────────────────
def _limpio(s):
    """Texto seguro para reportlab (escapa < > & y reemplaza símbolos que la fuente Helvetica no tiene)."""
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for a, b in {"→": "->", "≈": "~", "−": "-", "≥": ">=", "≤": "<=", "·": "-", "«": '"', "»": '"'}.items():
        s = s.replace(a, b)
    return s


# nombre simple y qué mide cada indicador de la semana (tarjetas, tabla del anexo y ayudas de la app)
NOMBRE_SIMPLE = {"riesgo_pais": "Riesgo país (pb)", "mep": "Dólar MEP ($)", "cable": "Dólar cable ($)", "a3500": "Dólar oficial A3500 ($)",
                 "brecha": "Brecha MEP contra oficial (%)", "tamar": "TAMAR (% por año)", "tir_soberanos": "Rendimiento de los bonos en dólares (TIR, %)",
                 "tirea_fija": "Rendimiento de las letras a tasa fija (TIREA, %)", "tir_cer": "Rendimiento de los CER sobre la inflación (TIR real, %)"}
QUE_MIDE = {"riesgo_pais": "Cuánto más que Estados Unidos le cobra el mercado a la Argentina para prestarle. Si baja, hay más confianza.",
            "mep": "El dólar que se compra con bonos en la bolsa.", "cable": "El dólar que se usa para girar plata al exterior.",
            "a3500": "El dólar oficial mayorista que publica el BCRA.", "brecha": "Cuánto más caro está el dólar MEP que el oficial.",
            "tamar": "La tasa que pagan los bancos grandes por depósitos grandes.",
            "tir_soberanos": "La ganancia por año del bono típico del Estado en dólares (mediana de Bonares y Globales).",
            "tirea_fija": "La ganancia por año de la letra típica a tasa fija en pesos (mediana de LECAP y BONCAP).",
            "tir_cer": "Cuánto rinde por año, por encima de la inflación, el bono CER típico."}
def pdf(S, texto, origen="", ctx=None):
    """Arma el PDF y devuelve sus bytes. Pensado para quien no sabe de finanzas: tarjetas con los números de la semana, el resumen
    con títulos, gráficos con «cómo leerlo», tu portafolio y un anexo con las tablas.
    ctx: el contexto completo que recibió el LLM (con la agenda y el portafolio que arma la app); si falta, se usa S['contexto']."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ini, fin = S["ini"], S["fin"]
    ctx = ctx or S.get("contexto", {})
    TINTA, VIOL, GRISC, FONDO = colors.HexColor("#141A2E"), colors.HexColor("#4B3AA6"), colors.HexColor("#4A5063"), colors.HexColor("#F2F1EC")
    BUENO, MALO = colors.HexColor("#14724F"), colors.HexColor("#B3261E")
    est = getSampleStyleSheet()
    E = lambda nombre, **kw: ParagraphStyle(nombre, parent=est["Normal"], **kw)
    tit = E("t", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=TINTA, spaceAfter=2)
    sub = E("s", fontSize=10.5, leading=14, textColor=GRISC, spaceAfter=2)
    meta = E("m", fontSize=7.8, leading=10, textColor=colors.HexColor("#7A8094"), spaceAfter=10)
    h2 = E("h2", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=VIOL, spaceBefore=14, spaceAfter=6)
    h3 = E("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=TINTA, spaceBefore=9, spaceAfter=2)
    cuerpo = E("b", fontSize=10.2, leading=15.2, textColor=colors.HexColor("#22283A"), alignment=TA_LEFT, spaceAfter=2)
    leer = E("l", fontName="Helvetica-Oblique", fontSize=8.6, leading=11.5, textColor=GRISC, spaceBefore=2, spaceAfter=6)
    chico = E("c", fontSize=7.8, leading=10, textColor=colors.HexColor("#6B7085"))
    celda = E("cel", fontSize=7.6, leading=9.4)
    celda_b = E("celb", fontName="Helvetica-Bold", fontSize=7.6, leading=9.4)
    k_lab = E("kl", fontSize=7.6, leading=9.5, textColor=GRISC)
    k_val = E("kv", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=TINTA)
    k_que = E("kq", fontSize=7, leading=8.8, textColor=colors.HexColor("#6B7085"))
    estilo_t = lambda f=7.8: TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", f), ("FONT", (0, 1), (-1, -1), "Helvetica", f),
                                         ("TEXTCOLOR", (0, 0), (-1, 0), GRISC), ("BACKGROUND", (0, 0), (-1, 0), FONDO),
                                         ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#C9CDDA")), ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#E2E4EA")),
                                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])
    pct = lambda v, d=1: "" if v is None else f"{_s(v, d)}%"

    def imagen(png, ancho=15.5 * cm):
        img = Image(png); img.drawHeight = ancho * img.imageHeight / img.imageWidth; img.drawWidth = ancho
        return img

    hist = [Paragraph("Bond Analytics · Informe semanal", tit),
            Paragraph(_limpio(f"Cómo se movieron los bonos argentinos entre el {ini.day} de {MESES[ini.month - 1]} y el {fin.day} de "
                              f"{MESES[fin.month - 1]} de {fin.year}"), sub),
            Paragraph(_limpio(f"Precios: {S['fuente_precios']} · generado el {datetime.now():%d/%m/%Y %H:%M} · texto: {origen}"), meta)]

    # ── tarjetas: la semana en cinco números ──
    I = S["indicadores"]
    fin_ = lambda k: I[k]["fin"] if k in I and np.isfinite(I[k]["fin"]) else None
    ini_ = lambda k: I[k]["inicio"] if k in I and np.isfinite(I[k]["inicio"]) else None
    tarjetas = []

    def tarjeta(etiqueta, valor, cambio, que, color=GRISC):
        tarjetas.append([Paragraph(_limpio(etiqueta), k_lab), Paragraph(_limpio(valor), k_val),
                         Paragraph(f'<font color="{color.hexval()}">{_limpio(cambio)}</font>', E("kc", fontSize=8.2, leading=10, fontName="Helvetica-Bold")),
                         Spacer(1, 2), Paragraph(_limpio(que), k_que)])
    if fin_("riesgo_pais") is not None and ini_("riesgo_pais") is not None:
        d = fin_("riesgo_pais") - ini_("riesgo_pais")
        tarjeta("Riesgo país", f"{fin_('riesgo_pais'):.0f} pb", f"{'bajó' if d < 0 else 'subió'} {abs(d):.0f} pb", "Si baja, el mercado confía más en el país.",
                BUENO if d < 0 else MALO)
    if fin_("mep") is not None and ini_("mep"):
        d = (fin_("mep") / ini_("mep") - 1) * 100
        tarjeta("Dólar MEP", f"$ {_n(fin_('mep'), 0)}", f"{'subió' if d > 0 else 'bajó'} {_n(abs(d), 1)}%", "El dólar que se compra con bonos en la bolsa.")
    G = S["grupos"]
    usd = [G[g]["var_precio_mediana_%"] for g in ("Bonares", "Globales") if g in G]
    if usd:
        d = float(np.mean(usd))
        tarjeta("Bonos en dólares", pct(d), "precio en la semana", "Bonos del Estado: Bonares y Globales.", BUENO if d >= 0 else MALO)
    if fin_("tirea_fija") is not None:
        a = ini_("tirea_fija")
        tarjeta("Tasa fija en pesos", f"{_n(fin_('tirea_fija'), 1)}%", f"por año (antes {_n(a, 1)}%)" if a is not None else "por año",
                "Lo que rinden las letras LECAP y BONCAP.")
    pf = ctx.get("portafolio")
    if pf and pf.get("retorno_semana_%") is not None:
        r_ = pf["retorno_semana_%"]
        tarjeta("Tu portafolio", pct(r_), f"{pf['moneda']} {_n(pf['resultado_aprox'], 0)}" if pf.get("resultado_aprox") is not None else "en la semana",
                "Lo que ganó o perdió tu cartera.", BUENO if r_ >= 0 else MALO)
    if tarjetas:
        ancho = 17 * cm / len(tarjetas)
        t = Table([tarjetas], colWidths=[ancho] * len(tarjetas))
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), FONDO), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
        hist.append(t)

    # ── qué pasó en la semana ──
    hist.append(Paragraph("Qué pasó en la semana", h2))
    for par in [x.strip() for x in re.split(r"\n\s*\n", texto) if x.strip()]:
        par = par.replace("**", "").replace("\n", " ")
        m = re.match(r"^([^:]{3,60}):\s*(.*)$", par)
        if m:
            cuerpo_ = m.group(2)[:1].upper() + m.group(2)[1:]
            hist.append(KeepTogether([Paragraph(_limpio(m.group(1)), h3), Paragraph(_limpio(cuerpo_), cuerpo)]))
        else:
            hist.append(Paragraph(_limpio(par), cuerpo))

    # ── la semana en gráficos ──
    Gf = graficos(S)
    graf = [("riesgo", "El riesgo país en los últimos tres meses",
             "La línea muestra cuánto más que Estados Unidos le cobra el mercado a la Argentina. Si baja, el país se ve más confiable; la franja "
             "sombreada es la semana del informe."),
            ("movimientos", "Los bonos que más subieron y más bajaron",
             "Cada barra es un bono: en verde los que subieron de precio en la semana y en rojo los que bajaron."),
            ("curva_usd", "Bonos del Estado en dólares: cuánto rinden según su plazo",
             "Cada punto es un bono: más a la derecha, más largo; más arriba, más rinde. El círculo vacío es hace una semana y el lleno, ahora: "
             "si el punto bajó, su precio subió."),
            ("tasa_fija", "Letras a tasa fija en pesos: cuánto rinden según su plazo",
             "Cada punto es una letra: cuánto rinde por año según cuántos días faltan para que venza. La línea gris es la curva de hace una semana.")]
    hist.append(Paragraph("La semana en gráficos", h2))
    for k, titulo, como in graf:
        if k in Gf:
            hist.append(KeepTogether([Paragraph(_limpio(titulo), h3), imagen(Gf[k]), Paragraph(_limpio("Cómo leerlo: " + como), leer)]))

    # ── tu portafolio ──
    if pf:
        m_ = pf["moneda"]
        ix, ts = pf["indicadores"], pf["tasas_en_la_semana"]
        filas = [["Indicador", "Tu cartera", "Qué quiere decir"]]
        agregar = lambda nom, val, txt: filas.append([Paragraph(_limpio(nom), celda_b), Paragraph(_limpio(val), celda), Paragraph(_limpio(txt), celda)])
        agregar("Resultado de la semana", "" if pf["retorno_semana_%"] is None else pct(pf["retorno_semana_%"], 2)
                + (f" ({m_} {_n(pf['resultado_aprox'], 0)})" if pf.get("resultado_aprox") is not None else ""),
                "Cuánto ganó o perdió la cartera, contando los intereses cobrados.")
        agregar("Valor de la cartera", f"{m_} {_n(pf['valor_hoy'], 0)}", f"Lo que valen hoy tus {pf['bonos']} bonos.")
        agregar("Duración", _n(ix["duracion_modificada"]), f"Si las tasas suben 1 punto, la cartera pierde cerca de {_n(ix['duracion_modificada'], 1)}%; si bajan, gana lo mismo.")
        agregar("DV01", f"{m_} {_n(ix['dv01'], 0)}", "Lo que perdés si las tasas suben apenas 1 punto básico (0,01%).")
        agregar("Convexidad", _n(ix["convexidad"], 1), "Un extra a favor: con tasas en baja ganás un poco más, y con tasas en suba perdés un poco menos.")
        agregar("Plazo promedio", f"{_n(ix['plazo_promedio_anios'], 1)} años", "Cuánto falta, en promedio, para que venzan tus bonos.")
        tipo, expl = ix["tipo_de_cartera"]
        agregar("Cómo está armada", tipo, TIPO_LLANO.get(tipo, expl).capitalize() + ".")
        agregar("Mayor exposición", f"{ix['mayor_exposicion'][0]} ({_n(ix['mayor_exposicion'][1], 0)}%)",
                f"El grupo con más peso. Diversificación: equivale a {_n(ix['bonos_efectivos'], 1)} bonos de igual peso.")
        if ix.get("riesgo"):
            rg = ix["riesgo"]
            agregar("VaR (día malo)", f"{m_} {_n(rg['var_95_1_dia'], 0)} ({_n(rg['var_%'], 1)}%)",
                    f"Solo 1 de cada 20 días se perdió más que esto (con {rg['ruedas']} días de historia).")
        if ts.get("cambio_promedio_pb") is not None:
            agregar("Las tasas en la semana", f"{_s(ts['cambio_promedio_pb'] / 100, 2)} puntos",
                    f"Cuánto cambiaron, en promedio, las tasas de tus bonos. Eso explica {_s(ts['efecto_de_la_duracion_pp'], 1)} puntos del resultado"
                    + ("" if ts.get("resto_pp") is None else f"; el resto ({_s(ts['resto_pp'], 1)}) son intereses, cupones y dólar."))
        t = Table(filas, colWidths=[3.6 * cm, 3.6 * cm, 9.8 * cm], repeatRows=1)
        t.setStyle(estilo_t())
        bloque = [Paragraph("Tu portafolio en la semana", h2), t]
        if len(pf.get("_serie", [])) > 5:
            sx = pd.Series({pd.Timestamp(d): v for d, v in pf["_serie"]})
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(sx.index, sx.values, color=AZUL, lw=1.8)
            ax.axvspan(pd.Timestamp(pf["_ini"]), pd.Timestamp(pf["_fin"]), color=VIOLETA, alpha=.12, label="Semana del informe")
            ax.axhline(100, color=GRIS, lw=.8, ls="--")
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.set_ylabel("Valor (empieza en 100)"); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper left")
            fig.tight_layout()
            bloque += [Paragraph("Cómo evolucionó el valor de tu cartera", h3), imagen(_png(fig)),
                       Paragraph(_limpio("Cómo leerlo: la línea arranca en 100 hace tres meses; si está en 101, la cartera vale 1% más que entonces. Se "
                                         "calcula con los bonos que tenés hoy."), leer)]
        hist.append(KeepTogether(bloque))
        filas = [["Bono", "Tipo", "Peso", f"Precio {ini:%d/%m}", f"Precio {fin:%d/%m}", "Cambio", "Ganancia", "Aporte"]]
        for b_ in pf["bonos_detalle"][:18]:
            filas.append([b_["ticker"], Paragraph(_limpio(b_["grupo"]), celda), f"{_n(b_['peso_%'], 0)}%",
                          "" if b_["precio_ini"] is None else _n(b_["precio_ini"]), "" if b_["precio_fin"] is None else _n(b_["precio_fin"]),
                          pct(b_["var_precio_%"]), pct(b_["retorno_semana_%"]), "" if b_["aporte_pp"] is None else f"{_s(b_['aporte_pp'])} pp"])
        t2 = Table(filas, colWidths=[1.6 * cm, 3.6 * cm, 1.4 * cm, 2 * cm, 2 * cm, 2.3 * cm, 2 * cm, 2.1 * cm], repeatRows=1)
        t2.setStyle(estilo_t()); t2.setStyle(TableStyle([("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        hist.append(KeepTogether([Paragraph("Bono por bono", h3), t2,
                                  Paragraph(_limpio("Cómo leerlo: el precio está en la moneda en que cotiza cada bono. La ganancia está en la moneda de tu "
                                                    "cartera y cuenta los intereses cobrados. El aporte es cuánto del resultado total explica cada bono "
                                                    "(peso x ganancia): la suma de los aportes da el resultado de la cartera."), leer)]))

    # ── anexo: los números en detalle ──
    hist += [Spacer(1, 8), Paragraph("Anexo: los números en detalle", h2),
             Paragraph(_limpio("Para quien quiera ver todas las cifras con las que se escribió el resumen."), cuerpo), Spacer(1, 4)]
    filas = [["Indicador", f"{ini:%d/%m}", f"{fin:%d/%m}", "Cambio", "Qué mide"]]
    for k, v in I.items():
        a, b = v["inicio"], v["fin"]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = v["dec"]
        cambio = f"{(b / a - 1) * 100:+.1f}%" if v["unidad"] == "$" else f"{b - a:+.0f} pb" if v["unidad"] == "pb" else f"{b - a:+.2f} pts"
        filas.append([Paragraph(_limpio(NOMBRE_SIMPLE.get(k, v["nombre"])), celda_b), _n(a, d), _n(b, d), cambio.replace(".", ","),
                      Paragraph(_limpio(QUE_MIDE.get(k, "")), celda)])
    t = Table(filas, colWidths=[4.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 7 * cm], repeatRows=1)
    t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (1, 0), (3, -1), "RIGHT")]))
    hist += [KeepTogether([Paragraph("Los indicadores de la semana", h3), t])]
    if S["grupos"]:
        filas = [["Tipo de bono", "Bonos", "Cambio de precio", f"Tasa {ini:%d/%m}", f"Tasa {fin:%d/%m}", "Cambio de tasa"]]
        for g, x in S["grupos"].items():
            filas.append([g, x["bonos"], pct(x["var_precio_mediana_%"], 2), "" if x["tasa_mediana_ini_%"] is None else f"{_n(x['tasa_mediana_ini_%'])}%",
                          "" if x["tasa_mediana_fin_%"] is None else f"{_n(x['tasa_mediana_fin_%'])}%",
                          "" if x["var_tasa_mediana_pb"] is None else f"{x['var_tasa_mediana_pb']:+.0f} pb"])
        t = Table(filas, colWidths=[3.6 * cm, 1.5 * cm, 3 * cm, 2.6 * cm, 2.6 * cm, 3 * cm], repeatRows=1)
        t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        hist += [KeepTogether([Paragraph("Por tipo de bono (valores del bono del medio de cada grupo)", h3), t])]
    if "tasa_fija_tabla" in Gf:
        filas = [["Letra o bono", "Tipo", "Vence en", f"Precio {fin:%d/%m}", f"Tasa {ini:%d/%m}", f"Tasa {fin:%d/%m}", "Cambio"]]
        for t_, r in Gf["tasa_fija_tabla"].iterrows():
            filas.append([t_, "Letra (LECAP)" if r["grupo"] == "LECAP" else "Bono (BONCAP)", f"{r['dias']:.0f} días", _n(r["precio_fin"]),
                          f"{_n(r['tasa_ini'] * 100)}%" if np.isfinite(r["tasa_ini"]) else "", f"{_n(r['tasa_fin'] * 100)}%",
                          f"{_s(r['var_pb'], 0)} pb" if np.isfinite(r["var_pb"]) else ""])
        t = Table(filas, colWidths=[2.2 * cm, 2.8 * cm, 2.1 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2 * cm], repeatRows=1)
        t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        hist += [KeepTogether([Paragraph("Letras y bonos a tasa fija (tasa = TIREA, la ganancia por año)", h3), t])]
    TO = S.get("ons")
    if TO is not None and len(TO):
        filas = [["ON", "Empresa", f"Precio {ini:%d/%m}", f"Precio {fin:%d/%m}", "Cambio", f"TIR {fin:%d/%m}", "Cambio de TIR"]]
        for t_, r in TO.sort_values("volumen_usd", ascending=False).head(12).iterrows():
            filas.append([t_, Paragraph(_limpio(r["emisor"]), celda), _n(r["precio_ini"]), _n(r["precio_fin"]), pct(r["var"] * 100, 2),
                          f"{_n(r['tir_fin'] * 100)}%" if np.isfinite(r["tir_fin"]) else "", f"{_s(r['var_pb'], 0)} pb" if np.isfinite(r["var_pb"]) else ""])
        t = Table(filas, colWidths=[1.6 * cm, 4.2 * cm, 2 * cm, 2 * cm, 1.8 * cm, 2 * cm, 2.2 * cm], repeatRows=1)
        t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        hist += [KeepTogether([Paragraph("Las ONs más operadas (precio en dólar MEP)", h3), t])]
    lic = S.get("licitacion")
    if lic:
        filas = [["Instrumento", "Tipo", "Colocado (millones)", "Tasa de corte", "Premio contra el mercado"]]
        for x in lic["instrumentos"]:
            filas.append([x["ticker"], x["tipo"], "" if x["adjudicado_millones"] is None else _n(x["adjudicado_millones"], 0),
                          "" if x["tirea_corte_%"] is None else f"{_n(x['tirea_corte_%'])}%", "" if x["premio_pb"] is None else f"{x['premio_pb']:+d} pb"])
        t = Table(filas, colWidths=[2.6 * cm, 2.6 * cm, 3.6 * cm, 3 * cm, 4.6 * cm], repeatRows=1)
        t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        hist += [KeepTogether([Paragraph(_limpio(f"Licitación del Tesoro del {lic['fecha']}: colocó $ {_n(lic['adjudicado_billones'])} billones"), h3), t])]
    ag = ctx.get("agenda_proximos_dias")
    if ag:
        if ag["tesoro_y_soberanos"]:
            filas = [["Fecha", "Día", "Qué pasa", "Bonos", "Detalle"]]
            for x in ag["tesoro_y_soberanos"]:
                filas.append([x["fecha"][:5], x["dia"], x["que"], Paragraph(_limpio(x["instrumentos"]), celda), Paragraph(_limpio(x["detalle"]), celda)])
            t = Table(filas, colWidths=[1.3 * cm, 1.7 * cm, 2.2 * cm, 2.6 * cm, 9.2 * cm], repeatRows=1)
            t.setStyle(estilo_t())
            hist += [KeepTogether([Paragraph(_limpio(f"Agenda del {ag['desde'][:5]} al {ag['hasta'][:5]}: Tesoro y bonos del Estado"), h3), t])]
        po = ag.get("pagos_de_ons") or []
        if po:
            filas = [["Fecha", "Día", "ON", "Empresa", "Qué paga", "Intereses", "Capital"]]
            for x in po:
                filas.append([x["fecha"][:5], x["dia"], x["ticker"], Paragraph(_limpio(x["emisor"]), celda), x["que"],
                              _n(x["renta_usd_por_100"], 2), _n(x["capital_usd_por_100"]) if x["capital_usd_por_100"] else ""])
            t = Table(filas, colWidths=[1.3 * cm, 1.8 * cm, 1.5 * cm, 4.6 * cm, 3.3 * cm, 2.2 * cm, 2.3 * cm], repeatRows=1)
            t.setStyle(estilo_t()); t.setStyle(TableStyle([("ALIGN", (5, 0), (-1, -1), "RIGHT")]))
            hist += [Paragraph(_limpio(f"Pagos de ONs del {ag['desde'][:5]} al {ag['hasta'][:5]} ({len(po)} pagos de {len(ag['emisores_que_pagan'])} empresas)"), h3),
                     t, Paragraph(_limpio("Montos en dólares por cada 100 de valor nominal. Las fechas son estimadas: confirmalas en el aviso de pago de "
                                          "cada empresa. Para cobrar hay que tener la ON el día hábil anterior."), leer)]

    # ── cómo se hizo ──
    hist += [Spacer(1, 10), Paragraph(_limpio(
        "Cómo se hizo: precios de cierre de BYMA (24 hs); los bonos en dólares, en dólar MEP. Cada tasa se calcula con el mismo método de la app: "
        "TIR de los flujos para los bonos en dólares y las ONs, TIREA del pago final para LECAP y BONCAP y TIR real para los CER. Se compara el "
        "último cierre con el de una semana antes. Fuentes: BYMA Open Data, BCRA, ArgentinaDatos, A3 Mercados, data912 y Secretaría de Finanzas. "
        "El texto lo redacta un modelo de lenguaje (Claude, de Anthropic) solo con esos números. Herramienta académica: no es "
        "recomendación de inversión."), chico)]

    def pie(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#6B7085"))
        canvas.drawString(2 * cm, 1.2 * cm, "Bond Analytics · informe semanal · no es recomendación de inversión")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                      title=f"Bond Analytics - Informe semanal {fin:%d-%m-%Y}", author="Bond Analytics").build(hist, onFirstPage=pie, onLaterPages=pie)
    return buf.getvalue()
