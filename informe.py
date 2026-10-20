"""
informe.py — informe semanal de renta fija argentina en PDF, para la pestaña «Informe semanal con IA» de BondAI.

Qué hace:
  1. Baja los cierres diarios de BYMA (24 hs) de las últimas semanas de cada bono (o usa el respaldo del 09/10/2026).
  2. Toma la última rueda disponible y la de una semana antes, y calcula para cada bono la variación del precio y de su tasa
     (TIR en dólares para Bonares, Globales y Tesoro USD; TIREA para LECAP y BONCAP; TIR real para los CER cupón cero).
  3. Suma el riesgo país, los dólares (MEP, cable, A3500 y brecha), la TAMAR, la licitación de la semana, el dólar futuro y la
     inflación breakeven de hoy.
  4. Un LLM (Claude) redacta el resumen ejecutivo usando SOLO esos números; sin clave se usa una plantilla determinística.
  5. Arma un PDF con el texto, una tabla de indicadores, gráficos y las mayores subas y bajas.

Uso desde la app:
    import informe as I
    S = I.semana(usar_api, cer, tamar, riesgo, dolares, licitaciones, extra)   # números de la semana
    texto, origen = I.redactar(S["contexto"])                                 # Claude o plantilla
    pdf = I.pdf(S, texto, origen)                                              # bytes del PDF
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


# ───────────────────────────── texto ─────────────────────────────
PROMPT = ("Sos analista de renta fija en Argentina y escribís el informe semanal de un equipo de research para estudiantes de economía. "
          "Español rioplatense neutro, frases claras. Usás SOLO los números del JSON (no inventes datos, no agregues noticias ni "
          "explicaciones causales que no surjan de los números) y respetás sus redondeos. Escribí 7 párrafos cortos separados por una "
          "línea en blanco; cada uno empieza con un título corto seguido de dos puntos: 'La semana en una línea', 'Dólar y riesgo país', "
          "'Bonos en dólares', 'Obligaciones negociables', 'Bonos en pesos', 'Licitación y agenda' y 'Riesgos'. En 'Obligaciones "
          "negociables' resumí cómo se movieron las ONs más operadas (precio, TIR y su diferencia contra los soberanos) nombrando las empresas. Si un dato falta, omitilo. Sin markdown, sin viñetas, "
          "sin tablas. El último párrafo aclara que no es recomendación de inversión.")


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
                messages=[{"role": "user", "content": "Datos de la semana (JSON):\n" + json.dumps(ctx, ensure_ascii=False)}])
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


def plantilla(c):
    """Resumen sin LLM, con los mismos números y la misma estructura."""
    I, G = c["indicadores"], c["por_grupo"]
    s = c["semana"]
    sub = lambda k: (I.get(k) or {}).get("variacion")
    p = []
    rp, mep, tir = I.get("riesgo_pais"), I.get("mep"), I.get("tir_soberanos")
    frase = f"La semana en una línea: entre el {s['desde']} y el {s['hasta']}"
    partes = []
    if rp and rp["fin"] is not None:
        partes.append(f"el riesgo país {'bajó' if (sub('riesgo_pais') or 0) < 0 else 'subió'} a {rp['fin']:.0f} pb ({_s(sub('riesgo_pais'), 0)} pb)")
    if tir and tir["fin"] is not None:
        partes.append(f"la TIR mediana de los soberanos en dólares pasó de {_n(tir['inicio'])}% a {_n(tir['fin'])}%")
    if mep and mep["fin"] is not None:
        partes.append(f"el dólar MEP cerró en $ {_n(mep['fin'])} ({_s((mep['fin'] / mep['inicio'] - 1) * 100)}%)")
    p.append(frase + (", " + ", ".join(partes[:-1]) + (" y " if len(partes) > 1 else "") + partes[-1] if partes else "") + ".")
    t = "Dólar y riesgo país: "
    if mep:
        t += f"el MEP pasó de $ {_n(mep['inicio'])} a $ {_n(mep['fin'])}"
        if I.get("cable"):
            t += f" y el cable de $ {_n(I['cable']['inicio'])} a $ {_n(I['cable']['fin'])}"
        t += ". "
    if I.get("a3500"):
        t += f"El oficial (A3500) terminó en $ {_n(I['a3500']['fin'])}"
        if I.get("brecha"):
            t += f", con una brecha del MEP de {_n(I['brecha']['fin'], 1)}% ({_s(sub('brecha'), 1)} pp)"
        t += ". "
    fu = c.get("dolar_futuro")
    if fu:
        t += f"En A3, el contrato {fu['contrato']} cotiza $ {_n(fu['precio'], 1)}, un {_n(fu['devaluacion_%'])}% de devaluación ({_n(fu['mensual_%'])}% mensual). "
    if rp:
        t += f"El riesgo país terminó en {rp['fin']:.0f} pb."
    p.append(t.strip())
    t = "Bonos en dólares: "
    for g in ("Globales", "Bonares", "Tesoro USD"):
        if g in G:
            x = G[g]
            t += f"los {g} {'subieron' if x['var_precio_mediana_%'] >= 0 else 'bajaron'} {_n(abs(x['var_precio_mediana_%']))}% (mediana)" + (f" y su TIR se movió {_s(x['var_tasa_mediana_pb'], 0)} pb" if x["var_tasa_mediana_pb"] is not None else "") + "; "
    if c.get("usd_mas_comprimio_pb"):
        k, v = next(iter(c["usd_mas_comprimio_pb"].items()))
        k2, v2 = next(iter(c["usd_mas_se_amplio_pb"].items()))
        t += (f"el que más comprimió fue {k} ({_s(v, 0)} pb) y " + (f"el que más se amplió, {k2} ({_s(v2, 0)} pb)." if v2 > 0 else f"el que menos se movió, {k2} ({_s(v2, 0)} pb)."))
    p.append(t.rstrip("; ") + ("" if t.endswith(".") else "."))
    on = c.get("obligaciones_negociables")
    if on:
        t = (f"Obligaciones negociables: entre las {on['ons_medidas']} ONs en dólares más operadas, el precio mediano "
             f"{'subió' if on['var_precio_mediana_%'] >= 0 else 'bajó'} {_n(abs(on['var_precio_mediana_%']))}% y la TIR mediana pasó de "
             f"{_n(on['tir_mediana_ini_%'])}% a {_n(on['tir_mediana_fin_%'])}%")
        if on.get("spread_vs_soberanos_pb") is not None:
            d_ = on["spread_vs_soberanos_pb"]
            t += f" ({_n(abs(d_), 0)} pb {'por debajo' if d_ < 0 else 'por encima'} de la TIR mediana de Bonares y Globales)"
        t += ". "
        top = list(on["mas_operadas"].items())[:4]
        if top:
            t += "Las más operadas: " + "; ".join(f"{k} ({v['emisor']}) {_s(v['var_%'])}%" + (f", TIR {_n(v['tir_%'])}%" if v["tir_%"] is not None else "")
                                                for k, v in top) + ". "
        b_, m_ = on["mejor_semana"], on["peor_semana"]
        t += f"La mejor semana fue la de {b_[0]} ({_s(b_[1])}%) y la peor, {m_[0]} ({_s(m_[1])}%)."
        p.append(t)
    t = "Bonos en pesos: en precio, "
    for g, nom in (("LECAP", "las LECAP"), ("BONCAP", "las BONCAP"), ("CER", "los CER"), ("TAMAR", "los TAMAR"), ("Dual", "los duales"), ("Dólar linked", "los dólar linked")):
        if g in G:
            x = G[g]
            t += f"{nom} {_s(x['var_precio_mediana_%'])}%" + (f" (tasa {_s(x['var_tasa_mediana_pb'], 0)} pb)" if x["var_tasa_mediana_pb"] is not None else "") + ", "
    t = t.rstrip(", ") + "."
    be = c.get("breakeven_proximos_meses_%")
    if be:
        t += " La inflación breakeven que descuenta el mercado para los próximos meses es " + ", ".join(f"{_n(v)}% en {k}" for k, v in list(be.items())[:3]) + "."
    if I.get("tamar"):
        t += f" La TAMAR quedó en {_n(I['tamar']['fin'])}% TNA."
    p.append(t)
    lic = c.get("licitacion_de_la_semana")
    if lic:
        ins = ", ".join(f"{x['ticker']}" + (f" a {_n(x['tirea_corte_%'])}%" if x["tirea_corte_%"] is not None else "") for x in lic["instrumentos"])
        prem = [x["premio_pb"] for x in lic["instrumentos"] if x["premio_pb"] is not None]
        t = (f"Licitación y agenda: en la licitación del {lic['fecha']} el Tesoro recibió ofertas por $ {_n(lic['ofertado_billones'])} billones y adjudicó "
             f"$ {_n(lic['adjudicado_billones'])} billones ({ins})." + (f" El premio mediano contra el secundario fue de {_s(float(np.median(prem)), 0)} pb." if prem else ""))
    else:
        t = "Licitación y agenda: no hubo licitación del Tesoro en la semana."
    if c.get("proxima_licitacion"):
        t += f" La próxima licitación es el {c['proxima_licitacion']}."
    p.append(t)
    p.append("Riesgos: los precios de una semana pueden cambiar rápido con el dólar, la inflación o noticias fiscales; las tasas implícitas incluyen "
             "primas por plazo y por riesgo, así que no son pronósticos. Herramienta académica: no constituye recomendación de inversión.")
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
        ax.axvline(0, color=GRIS, lw=.8); ax.set_xlabel("Variación del precio en la semana (%)"); ax.grid(alpha=.25, axis="x")
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
            ax.set_xlabel("Duración modificada (años)"); ax.set_ylabel("TIR %"); ax.grid(alpha=.25); ax.legend(fontsize=7)
            ax.set_title(f"Círculo vacío: {ini:%d/%m} · lleno: {fin:%d/%m}", fontsize=8.5, loc="left", color="#4A5063")
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
            ax.set_xlabel("Días al vencimiento"); ax.set_ylabel("TIREA %"); ax.grid(alpha=.25); ax.legend(fontsize=6.5, loc="lower right")
            out["tasa_fija"] = _png(fig)
            out["tasa_fija_tabla"] = f_
    r = S.get("riesgo")
    if r is not None and len(r):
        z = r[r.index >= pd.Timestamp(fin) - pd.Timedelta(days=95)]
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(z.index, z.values, color=AZUL, lw=1.6)
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=AZUL, alpha=.1, label="semana del informe")
        ax.set_ylabel("pb"); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper left")
        fig.autofmt_xdate()
        out["riesgo"] = _png(fig)
    return out


# ───────────────────────────── PDF ─────────────────────────────
def _limpio(s):
    """Texto seguro para reportlab (escapa < > & y reemplaza símbolos que la fuente Helvetica no tiene)."""
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for a, b in {"→": "->", "≈": "~", "−": "-", "≥": ">=", "≤": "<=", "·": "-"}.items():
        s = s.replace(a, b)
    return s


def pdf(S, texto, origen=""):
    """Arma el PDF y devuelve sus bytes."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ini, fin = S["ini"], S["fin"]
    est = getSampleStyleSheet()
    tit = ParagraphStyle("t", parent=est["Title"], fontSize=18, leading=22, spaceAfter=4, alignment=0, textColor=colors.HexColor("#141A2E"))
    sub = ParagraphStyle("s", parent=est["Normal"], fontSize=9.5, textColor=colors.HexColor("#4A5063"), spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=est["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#4B3AA6"))
    cuerpo = ParagraphStyle("b", parent=est["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=7)
    chico = ParagraphStyle("c", parent=est["Normal"], fontSize=7.8, leading=10, textColor=colors.HexColor("#6B7085"))
    hoy = datetime.now()
    hist = []
    hist.append(Paragraph("BondAI · Informe semanal de renta fija argentina", tit))
    hist.append(Paragraph(_limpio(f"Semana del {ini.day} de {MESES[ini.month - 1]} al {fin.day} de {MESES[fin.month - 1]} de {fin.year} · "
                                  f"precios: {S['fuente_precios']} · generado el {hoy:%d/%m/%Y %H:%M} · texto: {origen}"), sub))
    # indicadores
    filas = [["Indicador", f"{ini:%d/%m}", f"{fin:%d/%m}", "Variación"]]
    for k, v in S["indicadores"].items():
        a, b = v["inicio"], v["fin"]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = v["dec"]
        if v["unidad"] == "$":
            var = f"{(b / a - 1) * 100:+.2f}%"
        elif v["unidad"] == "pb":
            var = f"{b - a:+.0f} pb"
        else:
            var = f"{b - a:+.2f} pp"
        filas.append([v["nombre"], _n(a, d), _n(b, d), var])
    t = Table(filas, colWidths=[7.4 * cm, 2.7 * cm, 2.7 * cm, 2.9 * cm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141A2E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F1EC")]),
                           ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#C9CDDA")), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    hist += [Paragraph("Indicadores de la semana", h2), t]
    # resumen ejecutivo
    hist.append(Paragraph("Resumen ejecutivo", h2))
    for par in [x.strip() for x in re.split(r"\n\s*\n", texto) if x.strip()]:
        par = _limpio(par.replace("**", "").replace("\n", " "))
        m = re.match(r"^([^:]{3,40}):\s*(.*)$", par)
        hist.append(Paragraph(f"<b>{m.group(1)}:</b> {m.group(2)}" if m else par, cuerpo))
    # gráficos
    G = graficos(S)
    titulos = {"riesgo": "Riesgo país, últimos tres meses", "curva_usd": "Curva en dólares: TIR contra duración", "tasa_fija": "Tasa fija en pesos (LECAP y BONCAP)",
               "movimientos": "Precios: mayores subas y bajas de la semana"}
    for k in ("riesgo", "curva_usd", "tasa_fija", "movimientos"):
        if k in G:
            img = Image(G[k]); w = 16.5 * cm; img.drawHeight = w * img.imageHeight / img.imageWidth; img.drawWidth = w
            bloque = [Paragraph(titulos[k], h2), img]
            if k == "tasa_fija" and "tasa_fija_tabla" in G:
                filas = [["Instrumento", "Tipo", "Vence en", f"Precio {fin:%d/%m}", f"TIREA {ini:%d/%m}", f"TIREA {fin:%d/%m}", "Cambio"]]
                for t_, r in G["tasa_fija_tabla"].iterrows():
                    filas.append([t_, "Letra (LECAP)" if r["grupo"] == "LECAP" else "Bono (BONCAP)", f"{r['dias']:.0f} días", _n(r["precio_fin"]),
                                  f"{_n(r['tasa_ini'] * 100)}%" if np.isfinite(r["tasa_ini"]) else "", f"{_n(r['tasa_fin'] * 100)}%",
                                  f"{_s(r['var_pb'], 0)} pb" if np.isfinite(r["var_pb"]) else ""])
                tt = Table(filas, colWidths=[2.2 * cm, 2.8 * cm, 2.1 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2 * cm])
                tt.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
                                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141A2E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                        ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F1EC")])]))
                bloque += [Spacer(1, 4), tt, Paragraph(_limpio("TIREA = (pago final / precio)^(365 / días) - 1, con el pago final de cada instrumento: "
                                                               "VNO x (1 + TEM de emisión)^(meses 30/360). Se omiten las que vencen en menos de 15 días."), chico)]
            hist.append(KeepTogether(bloque))
    # obligaciones negociables
    TO = S.get("ons")
    if TO is not None and len(TO):
        filas = [["ON", "Emisor", f"Precio {ini:%d/%m}", f"Precio {fin:%d/%m}", "Var.", f"TIR {fin:%d/%m}", "Cambio TIR"]]
        for t_, r in TO.sort_values("volumen_usd", ascending=False).head(12).iterrows():
            filas.append([t_, _limpio(r["emisor"])[:24], _n(r["precio_ini"]), _n(r["precio_fin"]), f"{_s(r['var'] * 100)}%",
                          f"{_n(r['tir_fin'] * 100)}%" if np.isfinite(r["tir_fin"]) else "", f"{_s(r['var_pb'], 0)} pb" if np.isfinite(r["var_pb"]) else ""])
        t = Table(filas, colWidths=[1.6 * cm, 4.2 * cm, 2 * cm, 2 * cm, 1.8 * cm, 2 * cm, 2.2 * cm])
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.2),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141A2E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F1EC")])]))
        hist += [KeepTogether([Paragraph("Obligaciones negociables más operadas (dólar MEP)", h2), t,
                               Paragraph(_limpio("TIR con las condiciones de la ficha técnica de BYMA (pagos semestrales si la ficha no informa la "
                                                 "frecuencia). Ordenadas por volumen operado."), chico)])]
    # tabla por grupo
    if S["grupos"]:
        filas = [["Grupo", "Bonos", "Precio (mediana)", f"Tasa {ini:%d/%m}", f"Tasa {fin:%d/%m}", "Cambio de tasa"]]
        for g, x in S["grupos"].items():
            filas.append([g, x["bonos"], f"{x['var_precio_mediana_%']:+.2f}%", "" if x["tasa_mediana_ini_%"] is None else f"{_n(x['tasa_mediana_ini_%'])}%",
                          "" if x["tasa_mediana_fin_%"] is None else f"{_n(x['tasa_mediana_fin_%'])}%",
                          "" if x["var_tasa_mediana_pb"] is None else f"{x['var_tasa_mediana_pb']:+.0f} pb"])
        t = Table(filas, colWidths=[3.6 * cm, 1.5 * cm, 3 * cm, 2.6 * cm, 2.6 * cm, 3 * cm])
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141A2E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F1EC")])]))
        hist += [KeepTogether([Paragraph("Por tipo de bono", h2), t])]
    lic = S.get("licitacion")
    if lic:
        filas = [["Instrumento", "Tipo", "Adjudicado (millones)", "TIREA de corte", "Premio vs. secundario"]]
        for x in lic["instrumentos"]:
            filas.append([x["ticker"], x["tipo"], "" if x["adjudicado_millones"] is None else _n(x["adjudicado_millones"], 0),
                          "" if x["tirea_corte_%"] is None else f"{_n(x['tirea_corte_%'])}%", "" if x["premio_pb"] is None else f"{x['premio_pb']:+d} pb"])
        t = Table(filas, colWidths=[2.6 * cm, 2.6 * cm, 4 * cm, 3 * cm, 4 * cm])
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141A2E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        hist += [KeepTogether([Paragraph(_limpio(f"Licitación del {lic['fecha']}: adjudicó $ {_n(lic['adjudicado_billones'])} billones de "
                                                 f"$ {_n(lic['ofertado_billones'])} ofertados"), h2), t])]
    hist += [Spacer(1, 10), Paragraph(_limpio(
        "Metodología: precios de cierre de BYMA (24 hs); bonos en dólares en dólar MEP (tickers D). La tasa de cada bono se calcula con el "
        "mismo método de la app: TIR efectiva anual de los flujos para Bonares, Globales y Tesoro USD; TIREA del pago final para LECAP y BONCAP; "
        "TIR real con el CER de 10 hábiles antes para los CER cupón cero; ONs con la TIR de sus flujos según la ficha de BYMA. Variación = último cierre contra el de una semana antes. Fuentes: BYMA "
        "Open Data, BCRA, ArgentinaDatos (riesgo país), A3 Mercados (futuros) y Secretaría de Finanzas (licitaciones). El texto lo redacta un "
        "modelo de lenguaje a partir de esos números, sin datos propios. Herramienta académica (Programación para Economistas, UCA); no es "
        "recomendación de inversión."), chico)]

    def pie(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#6B7085"))
        canvas.drawString(2 * cm, 1.2 * cm, "BondAI · informe semanal · no es recomendación de inversión")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                      title=f"BondAI - Informe semanal {fin:%d-%m-%Y}", author="BondAI").build(hist, onFirstPage=pie, onLaterPages=pie)
    return buf.getvalue()
