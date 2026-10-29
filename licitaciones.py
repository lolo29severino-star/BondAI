"""
licitaciones.py — licitaciones del Tesoro (premio contra el mercado secundario) y calendario de licitaciones, pagos y
vencimientos. Los usa el informe semanal: la licitación de la semana y la agenda de las dos semanas siguientes.

Uso desde la app:
    import licitaciones as L
    sub, col, prox, info = L.actualizar(en_vivo, cer, tamar, a3500)   # respaldo + licitaciones nuevas leídas solas de Finanzas
    ev = L.eventos(date(2026, 10, 1), date(2027, 3, 31))                # calendario: licitaciones, cortes, rentas, amortizaciones, feriados

Actualización automática: en vivo, actualizar() lee la página de noticias de la Secretaría de Finanzas
(argentina.gob.ar/economia/finanzas/noticias), abre cada «Resultado de la licitación» posterior al respaldo, lee sus tablas
(instrumento, VNO ofertado y adjudicado, valor efectivo, precio de corte, TIREA) y calcula el premio contra el cierre de BYMA del día
de la licitación; también lee el último «Llamado a licitación» para mostrar el menú de la próxima. Si algo falla, queda el respaldo.

El premio es la TIREA de corte menos la TIREA del secundario (BYMA, cierre de 24 hs del día de la licitación), las dos con el mismo
método (el respaldo se armó con data/build_licitaciones.js y se validó contra este código en Python). Fuentes: comunicados
«Resultado de la licitación» de la Secretaría de Finanzas y planilla «Colocaciones de deuda»; cronograma 2026 y feriados bancarios
del BCRA. Respaldo con las licitaciones de 2026 hasta la del 14/10/2026.
"""
import base64
import html as _html
import json
import re
import zlib
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

import numpy as np
import pandas as pd

import bondai as B
import pesos as P

RESPALDO = json.loads(zlib.decompress(base64.b64decode("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")))
FAMILIAS = {"fija": "Tasa fija", "cer": "CER", "tamar": "TAMAR", "dual": "Dual", "dl": "Dólar linked", "usd": "BONAR USD"}
COLORES = {"fija": "#2a78d6", "cer": "#1baf7a", "tamar": "#d99a00", "dual": "#e0569a", "dl": "#7a5cf0", "usd": "#eb6834"}
TIPOS_EV = {"lla": "Llamado", "lic": "Licitación", "liq": "Liquidación", "cor": "Corte de cupón", "ren": "Renta", "amo": "Amortización",
            "ven": "Vencimiento", "fer": "Feriado"}
FERIADOS = {date.fromisoformat(f) for f, _ in RESPALDO["calendario"]["feriados"]} | P.FERIADOS
habil = lambda f: f.weekday() < 5 and f not in FERIADOS
_n = lambda v, d=2: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")  # número con coma decimal


def siguiente_habil(f):
    while not habil(f):
        f += timedelta(days=1)
    return f


def anterior_habil(f):
    f -= timedelta(days=1)
    while not habil(f):
        f -= timedelta(days=1)
    return f


# ───────────────────────────── resultados de las licitaciones ─────────────────────────────
def resultados():
    """(subastas, colocaciones). Subastas: una fila por licitación (ofertado y adjudicado en billones de pesos).
    Colocaciones: una fila por instrumento con precio de corte, TIREA de corte, TIREA del secundario, premio (pb y % del precio)
    y lo que pasó el día de la liquidación."""
    L = RESPALDO["licitaciones"]
    sub = pd.DataFrame(L["subastas"])
    sub["f"], sub["liq"] = pd.to_datetime(sub["f"]).dt.date, pd.to_datetime(sub["liq"]).dt.date
    cols = ["fecha", "liquida", "ticker", "familia", "nueva", "en_usd", "vno_ofertado", "vno_adjudicado", "valor_efectivo", "precio", "tirea_corte",
            "tirea_propia", "dias", "tirea_secundario", "referencia", "premio_pb", "premio_precio", "tirea_al_liquidar", "fecha_post", "cambio_post_pb",
            "resultado_al_liquidar"]
    col = pd.DataFrame(L["filas"], columns=cols)
    col["fecha"], col["liquida"] = pd.to_datetime(col["fecha"]).dt.date, pd.to_datetime(col["liquida"]).dt.date
    return sub.sort_values("f"), col


def resumen(col, medida="pb"):
    """Premio mediano, cuántas colocaciones tuvieron premio y cuántas ganaban el día de la liquidación."""
    c = col.dropna(subset=["premio_pb"])
    v = c["premio_pb"] if medida == "pb" else c["premio_precio"] * 100
    post = col.dropna(subset=["resultado_al_liquidar"])
    return {"mediana": float(v.median()) if len(v) else np.nan, "con_premio": int((c["premio_pb"] > 0).sum()), "n": len(c),
            "ganan": int((post["resultado_al_liquidar"] > 0).sum()), "n_post": len(post),
            "mediana_post": float(post["resultado_al_liquidar"].median()) if len(post) else np.nan}


def proxima(hoy=None, prox=None):
    """Próxima licitación: la del último llamado si todavía no pasó; si no, la siguiente del cronograma (con el menú vacío
    hasta que la Secretaría de Finanzas publique el llamado)."""
    hoy = hoy or date.today()
    prox = prox or RESPALDO["calendario"]["proxima"]
    if date.fromisoformat(prox["lic"]) >= hoy:
        return prox
    for c in RESPALDO["calendario"]["cronograma"]:
        if date.fromisoformat(c["lic"]) >= hoy:
            ll = date.fromisoformat(c["llamado"])
            return {**c, "hora": "10:00 a 15:00", "menu": [],
                    "nota": f"El menú (qué instrumentos se licitan) se publica con el llamado del {ll:%d/%m/%Y}.",
                    "fuente": "Secretaría de Finanzas, cronograma de licitaciones 2026."}
    return prox


# ───────────────────────────── actualización automática (Secretaría de Finanzas + BYMA) ─────────────────────────────
GOB = "https://www.argentina.gob.ar"
LISTADO = GOB + "/economia/finanzas/noticias"
BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
MESES = {m: i for i, m in enumerate(["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE",
                                     "NOVIEMBRE", "DICIEMBRE"], 1)}
LETRA_MES = "EFMAYJLGSOND"   # letra del mes en los tickers del Tesoro (S29E7 = 29 de enero de 2027)


class _Tablas(HTMLParser):
    """Lee las tablas de un comunicado (cada fila como lista de textos) y el texto completo."""

    def __init__(self):
        super().__init__()
        self.tablas, self.texto, self._celda = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tablas.append([])
        elif tag == "tr" and self.tablas:
            self.tablas[-1].append([])
        elif tag in ("td", "th"):
            self._celda = []
        elif tag in ("br", "p", "li"):
            self.texto.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._celda is not None:
            if self.tablas and self.tablas[-1]:
                self.tablas[-1][-1].append(" ".join("".join(self._celda).split()))
            self._celda = None
            self.texto.append("\n")

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)
        self.texto.append(data)


def _leer_html(h):
    p = _Tablas()
    p.feed(h)
    return p.tablas, " ".join("".join(p.texto).split())


def _num(s):
    """'$ 4.448.291' → 4448291 · 'USD 946,46' → 946.46 · '27,47%' → 27.47."""
    z = re.sub(r"USD|\$|\(\*+\)|%|\s", "", str(s or ""))
    if re.match(r"^\d{1,3}(,\d{3})+\.\d+", z) or re.match(r"^\d{1,3}\.\d{1,2}$", z):   # formato inglés (USD 1,011.98)
        z = z.replace(",", "")
    else:
        z = z.replace(".", "").replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", z)
    return float(m.group()) if m else np.nan


def _fecha_txt(s):
    m = re.search(r"(\d{1,2}) DE ([A-ZÁÉÍÓÚ]+)(?: DE)? (\d{4})", s.upper())
    return date(int(m.group(3)), MESES[m.group(2)], int(m.group(1))) if m and m.group(2) in MESES else None


def _ticker_nuevo(fam, venc, nombre):
    """Ticker probable de un instrumento nuevo (el comunicado no lo trae): S/T + día + letra del mes + año para LECAP y BONCAP, etc."""
    if not venc:
        return None
    l, y, dd = LETRA_MES[venc.month - 1], venc.year % 10, f"{venc.day:02d}"
    letra = "LETRA" in nombre.upper()
    return {"fija": ("S" if letra else "T") + dd + l + str(y), "cer": ("X" + dd + l + str(y)) if letra else f"TZX{l}{y}",
            "tamar": ("M" + dd + l + str(y)) if letra else f"TM{l}{venc.year % 100}", "dl": ("D" + dd + l + str(y)) if letra else f"TZV{l}{y}",
            "dual": f"TXM{l}{y}", "dualdl": f"TMV{l}{y}"}.get(fam)


def _identificar(nombre, hd, liq, corte):
    """Familia, ticker y condiciones de un instrumento a partir de su nombre en el comunicado."""
    N = nombre.upper()
    venc = _fecha_txt(N.split("VENCIMIENTO", 1)[-1]) if "VENCIMIENTO" in N else None
    fam = ("dualdl" if re.search(r"DUAL TAMAR\s*/\s*D[ÓO]LAR|MONEDA DUAL", N) else "dual" if re.search(r"DUAL CER", N) else
           "tamar" if "TAMAR" in N else "dl" if re.search(r"VINCULAD|D[ÓO]LAR LINKED", N) else
           "usd" if re.search(r"D[ÓO]LARES ESTADOUNIDENSES|HARD DOLLAR|BONAR", N + " " + hd.upper()) and "CERO CUP" not in N else
           "cer" if "CER" in N else "fija")
    m = re.search(r"\(((?=[A-Z0-9]*\d)[A-Z0-9]{4,6})\s*[-–]", N) or re.search(r"[-–]\s*((?=[A-Z0-9]*\d)[A-Z0-9]{4,6})\s*\)", N)
    t = m.group(1) if m else None
    nueva = bool(re.search(r"\bNUEV[OA]\b", N))
    meta = {"fam": fam, "venc": venc, "nueva": nueva}
    if fam == "fija":
        x = next((x for x in P.FIJA if x[0] == t or (t is None and date.fromisoformat(x[3]) == venc)), None)
        if x:
            t, meta["em"], meta["tem"] = x[0], date.fromisoformat(x[2]), x[4]
        elif "%" in corte:
            meta["em"], meta["tem"] = liq, _num(corte) / 100
    elif fam == "cer":
        x = next((x for x in P.CERZ if x[0] == t or (t is None and date.fromisoformat(x[3]) == venc)), None)
        meta["fb"] = date.fromisoformat(x[4]) if x else P.restar_habiles(liq, 10)
        t = x[0] if x else t
    elif fam == "tamar":
        x = next((x for x in P.TAMAR if x[0] == t or (t is None and date.fromisoformat(x[2]) == venc)), None)
        if x:
            t, meta["em"], meta["m"] = x[0], date.fromisoformat(x[1]), x[3] or 0.0
        else:
            meta["em"], meta["m"] = liq, (_num(corte) if "%" in corte else 0.0)
    elif fam in ("dual", "dualdl"):
        x = next((d for d in P.DUALES if d["t"] == t or (t is None and date.fromisoformat(d["v"]) == venc)), None)
        em = date.fromisoformat(x["em"]) if x else liq
        t = x["t"] if x else t
        meta["fb"] = P.restar_habiles(em, 10)
    elif fam == "usd":
        meta["b"] = B.universo_usd().get(t)
    meta["t"] = t or _ticker_nuevo(fam, venc, nombre)
    meta["ticker_estimado"] = t is None
    return meta


def _cer_en(cer, f):
    c = cer[cer.index <= pd.Timestamp(f)]
    return float(c.iloc[-1]) if len(c) else np.nan


def _pago_tamar(em, venc, m, hoy, tamar):
    """Pago final por 100 VN de un bono TAMAR: promedio de la TAMAR de 10 hábiles antes de la emisión a 10 hábiles antes del
    vencimiento; lo que todavía no se publicó se completa con el promedio de las últimas 5 ruedas (convención del mercado)."""
    a, z = P.restar_habiles(em, 10), P.restar_habiles(venc, 10)
    obs = tamar[tamar.index < pd.Timestamp(hoy)].sort_index()
    if obs.empty:
        return np.nan
    sup = float(obs.iloc[-5:].mean())
    dentro = obs[(obs.index >= pd.Timestamp(a)) & (obs.index <= pd.Timestamp(z))]
    f, nr = max(a, obs.index[-1].date() + timedelta(days=1)), 0
    while f <= z:
        nr += P.habil(f); f += timedelta(days=1)
    prom = (float(dentro.sum()) + sup * nr) / ((len(dentro) + nr) or 1)
    return 100 * (1 + P.tem_tamar(prom + m)) ** (P.d360(em, venc) / 30)


def _rend(meta, s, precio, hoy, S):
    """TIREA (y), días y MD de un instrumento comprado a 'precio' (por 100 VN; dólar linked y BONAR en dólares) liquidando en s."""
    fin = P.ajustar_pago(meta["venc"]) if meta.get("venc") else None
    if not fin or not (precio > 0):
        return np.nan, np.nan, np.nan
    d = (fin - s).days
    if d <= 0:
        return np.nan, d, np.nan
    fam = meta["fam"]
    if fam == "fija" and meta.get("tem") is not None and meta.get("em"):
        pago = 100 * (1 + meta["tem"]) ** (P.d360(meta["em"], meta["venc"]) / 30)
    elif fam in ("cer", "dual") and meta.get("fb"):
        pago = 100 * _cer_en(S["cer"], P.restar_habiles(s, 10)) / _cer_en(S["cer"], meta["fb"])
    elif fam == "tamar" and meta.get("em"):
        pago = _pago_tamar(meta["em"], meta["venc"], meta["m"], hoy, S["tamar"])
    elif fam in ("dl", "dualdl"):
        pago = 100.0
    elif fam == "usd" and meta.get("b"):
        try:
            m_ = B.metricas_bono(meta["b"], s, precio)
        except ValueError:
            return np.nan, d, np.nan
        return m_["tir"], d, m_["md"]
    else:
        return np.nan, d, np.nan
    return (pago / precio) ** (365 / d) - 1, d, d / 365


def _historia(sesion, t, desde, hasta):
    """Cierres diarios de BYMA (24 hs): {fecha: (cierre, máximo, mínimo, volumen)}."""
    ts = lambda f: int(datetime(f.year, f.month, f.day, tzinfo=timezone(timedelta(hours=-3))).timestamp())
    r = sesion.get(f"{BYMA_API}/chart/historical-series/history", params={"symbol": t + " 24HS", "resolution": "D", "from": ts(desde),
                   "to": ts(hasta) + 86400}, verify=False, timeout=(5, 12))
    j = r.json()
    if j.get("s") != "ok":
        return {}
    return {(datetime.fromtimestamp(x, tz=timezone.utc) - timedelta(hours=3)).date(): (c, h, l, v)
            for x, c, h, l, v in zip(j["t"], j["c"], j["h"], j["l"], j["v"])}


def _px(H, f):
    """Cierre válido de un día: hubo volumen y el cierre está entre el mínimo y el máximo (si no, el promedio de los dos)."""
    x = H.get(f)
    if not x:
        return None
    c, h, l, v = x
    if v and c and (not h or l - 1e-9 <= c <= h + 1e-9):
        return c
    if h and l and v:
        return (h + l) / 2
    return None


def serie_a3500(vivo=None):
    """A3500 diario: el del respaldo (todo 2026) completado con la serie en vivo que pasa la app."""
    s = pd.Series({pd.Timestamp(f): v for f, v in RESPALDO.get("a3500", [])}, dtype=float)
    if vivo is not None and len(vivo):
        s = pd.concat([s, vivo.astype(float)])
        s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _a3500_previo(a3500, f):
    s = a3500[a3500.index < pd.Timestamp(f)]
    return float(s.iloc[-1]) if len(s) else np.nan


def _precio_mercado(meta, H, f, S):
    p = _px(H, f)
    if p is None:
        return None
    return p / _a3500_previo(S["a3500"], f) if meta["fam"] in ("dl", "dualdl") else p


def _familia_meta(fam, liq):
    """Instrumentos de una familia todavía vivos (para interpolar la tasa de un instrumento nuevo)."""
    out = []
    if fam == "fija":
        out = [{"fam": fam, "t": x[0], "venc": date.fromisoformat(x[3]), "em": date.fromisoformat(x[2]), "tem": x[4]} for x in P.FIJA]
    elif fam == "cer":
        out = [{"fam": fam, "t": x[0], "venc": date.fromisoformat(x[3]), "fb": date.fromisoformat(x[4])} for x in P.CERZ]
    elif fam == "tamar":
        out = [{"fam": fam, "t": x[0], "venc": date.fromisoformat(x[2]), "em": date.fromisoformat(x[1]), "m": x[3] or 0.0} for x in P.TAMAR]
    elif fam == "dl":
        out = [{"fam": fam, "t": x[0], "venc": date.fromisoformat(x[2])} for x in P.DL]
    elif fam == "dual":
        out = [{"fam": fam, "t": d["t"], "venc": date.fromisoformat(d["v"]), "fb": P.restar_habiles(date.fromisoformat(d["em"]), 10)}
               for d in P.DUALES if d["clase"] == "cer"]
    return [m for m in out if m["venc"] > liq + timedelta(days=7)][:14]


def calcular_filas(f, filas_crudas, S, historia, hoy=None):
    """Una fila por instrumento licitado, con el premio contra el secundario (mismo método que el respaldo).
    filas_crudas: [(encabezado de la tabla, celdas)]; historia(ticker) → {fecha: (c, h, l, v)}."""
    hoy = hoy or date.today()
    liq = f
    for _ in range(2):
        liq = P.siguiente_habil(liq)
    s1 = P.siguiente_habil(f)
    out = []
    for hd, c in filas_crudas:
        if len(c) >= 2 and re.match(r"^DE \d{4}", c[1]):
            c = [c[0] + " " + c[1]] + c[2:]
        if len(c) < 6 or "VENCIMIENTO" not in c[0].upper() or re.search(r"CONVERSI|Detalle Nuevos", hd, re.I):
            continue
        meta = _identificar(c[0], hd, liq, c[4])
        if not meta.get("venc"):
            continue
        porcentajes = [_num(z) / 100 for z in c[5:] if "%" in z]
        tir_of = (max(porcentajes) if meta["fam"] == "usd" and len(porcentajes) > 1 else porcentajes[0]) if porcentajes else np.nan
        precio = 100.0 if "%" in c[4] else _num(c[4]) / 10
        yA, d, md = _rend(meta, liq, precio, f, S)
        H = historia(meta["t"] + ("D" if meta["fam"] == "usd" else "")) if meta["t"] else {}
        pre, ref = None, None
        for g, tag in ((f, "cierre"), (P.restar_habiles(f, 1), "previo")):
            p = _precio_mercado(meta, H, g, S)
            if p:
                y, _, _ = _rend(meta, P.siguiente_habil(g), p, f, S)
                if np.isfinite(y):
                    pre, ref = y, tag
                    break
        if pre is None and meta["fam"] != "usd":            # nuevo: interpolación por días con su familia
            pts = []
            for o in _familia_meta(meta["fam"], liq):
                if o["t"] == meta["t"]:
                    continue
                p = _precio_mercado(o, historia(o["t"]), f, S)
                if p:
                    y, dd, _ = _rend(o, s1, p, f, S)
                    if np.isfinite(y) and dd >= 7:
                        pts.append((dd, y))
            if len(pts) >= 2:
                pts.sort()
                dd = (P.ajustar_pago(meta["venc"]) - s1).days
                pre = float(np.interp(dd, [x for x, _ in pts], [y for _, y in pts]))
                ref = "curva" if pts[0][0] <= dd <= pts[-1][0] else "curva-extrap"
        post, f_post = None, None
        for g in sorted(k for k in H if k >= liq):
            p = _precio_mercado(meta, H, g, S)
            if p:
                y, _, _ = _rend(meta, P.siguiente_habil(g), p, g, S)
                if np.isfinite(y):
                    post, f_post = y, g
                    break
        gan = lambda y1: (np.nan if y1 is None or not np.isfinite(yA) else -(md or 0) * (y1 - yA) if meta["fam"] == "usd"
                          else ((1 + yA) / (1 + y1)) ** (d / 365) - 1)
        off = tir_of - yA if np.isfinite(tir_of) and np.isfinite(yA) else 0.0
        en_usd = meta["fam"] in ("dl", "dualdl", "usd")
        r4 = lambda v, k=5: round(float(v), k) if v is not None and np.isfinite(v) else None
        out.append([f.isoformat(), liq.isoformat(), meta["t"], "dual" if meta["fam"] == "dualdl" else meta["fam"], int(meta["nueva"]), int(en_usd),
                    r4(_num(c[1]), 0), r4(_num(c[2]), 0), r4(_num(c[3]), 0), r4(precio, 4), r4(tir_of if np.isfinite(tir_of) else yA),
                    r4(yA), int(d) if np.isfinite(d) else None, r4(pre + off if pre is not None else None), ref,
                    r4((yA - pre) * 1e4 if pre is not None and np.isfinite(yA) else None, 1), r4(gan(pre)),
                    r4(post + off if post is not None else None), f_post.isoformat() if f_post else None,
                    r4((post - yA) * 1e4 if post is not None and np.isfinite(yA) else None, 1), r4(gan(post))])
    return out


def leer_resultado(h, f):
    """Comunicado «Resultado de la licitación» → (subasta, filas crudas). Montos en millones; ofertado y adjudicado en billones."""
    tablas, texto = _leer_html(h)
    crudas = []
    for tb in tablas:
        if not tb or re.search(r"CONVERSI|Detalle Nuevos|CANJE", " ".join(tb[0]), re.I):
            continue
        hd = " | ".join(tb[0])
        for fila in tb[1:]:
            if len(fila) >= 2 and re.match(r"^DE \d{4}", fila[1]):       # nombre partido en dos celdas
                fila = [fila[0] + " " + fila[1]] + fila[2:]
            if fila and "VENCIMIENTO" in fila[0].upper():
                crudas.append((hd, fila))
    of = re.search(r"ofertas por un total de valor efectivo de \$\s*([\d.,]+)\s*billones", texto, re.I)
    ad = re.search(r"adjudic\w*\s+un total de valor efectivo (?:de )?\$\s*([\d.,]+)\s*billones", texto, re.I)
    tot = {fila[0].lower(): fila[-1] for tb in tablas for fila in tb if fila and fila[0].lower().startswith("total")}
    ofer = _num(of.group(1)) if of else _num(tot.get("total ve ofertado (*)", np.nan)) / 1e6
    adj = _num(ad.group(1)) if ad else _num(tot.get("total valor efectivo adjudicado (*)", np.nan)) / 1e6
    tc = re.search(r"Pesos\s*/\s*USD\s*([\d.,]+)", texto) or re.search(r"([\d.]+,\d+)\s*pesos por", texto, re.I)
    liq = f
    for _ in range(2):
        liq = P.siguiente_habil(liq)
    return {"f": f.isoformat(), "liq": liq.isoformat(), "tc": _num(tc.group(1)) if tc else None, "ofer": round(ofer, 2), "adj": round(adj, 2)}, crudas


TIPO_MENU = {"fija": "LECAP o BONCAP", "cer": "CER", "tamar": "Bono TAMAR", "dl": "Dólar linked", "dual": "Dual CER / TAMAR", "dualdl": "Dual TAMAR / dólar linked",
             "usd": "BONAR en dólares"}
PAGA_MENU = {"fija": "tasa fija", "cer": "CER + tasa", "tamar": "TAMAR + margen", "dl": "A3500", "dual": "CER o TAMAR", "dualdl": "TAMAR o A3500",
             "usd": "dólares"}


def leer_llamado(h, f):
    """Comunicado «Llamado a licitación» → próxima licitación con su menú."""
    tablas, texto = _leer_html(h)
    menu, vistos = [], set()
    for tb in tablas:
        for fila in tb[1:]:
            if not fila or "VENCIMIENTO" not in fila[0].upper():
                continue
            meta = _identificar(fila[0], " ".join(tb[0]), f, "")
            if meta["t"] in vistos:
                continue
            vistos.add(meta["t"])
            menu.append([meta["t"], TIPO_MENU.get(meta["fam"], ""), PAGA_MENU.get(meta["fam"], ""), "nuevo" if meta["nueva"] else "reapertura",
                         f"{meta['venc']:%d/%m/%Y}" if meta["venc"] else ""])
    lic = re.search(r"del día \w+ (\d{1,2} de \w+ de \d{4})\s*\(T\)", texto)
    liq = re.search(r"(\d{1,2} de \w+ de \d{4})\s*\(T\s*\+\s*\d\)", texto)
    lic_f, liq_f = (_fecha_txt(lic.group(1)) if lic else None), (_fecha_txt(liq.group(1)) if liq else None)
    if not lic_f:
        return None
    return {"llamado": f.isoformat(), "lic": lic_f.isoformat(), "liq": (liq_f or P.siguiente_habil(P.siguiente_habil(lic_f))).isoformat(),
            "hora": "10:00 a 15:00", "menu": menu, "nota": "Menú leído del comunicado del llamado.",
            "fuente": f"Secretaría de Finanzas, Llamado a licitación del {f:%d/%m/%Y}."}


def listar_noticias(h):
    """Página de noticias de Finanzas → [(fecha, url, título)]."""
    out = []
    for href, f, tit in re.findall(r'<a href="(/noticias/[^"]+)"[^>]*>.*?<time datetime="(\d{4}-\d{2}-\d{2})[^"]*">.*?<h3>(.*?)</h3>', h, re.S):
        out.append((date.fromisoformat(f), GOB + href, _html.unescape(re.sub(r"<[^>]+>", "", tit)).strip()))
    return out


es_resultado = lambda t: re.match(r"Resultado de (la )?licitaci", t, re.I) is not None and not re.search(r"conversi|segunda vuelta|canje", t, re.I)
es_llamado = lambda t: re.match(r"Llamado a licitaci", t, re.I) is not None and not re.search(r"conversi|canje", t, re.I)


def actualizar(usar_api, cer, tamar, a3500, hoy=None):
    """Resultados del respaldo + los comunicados nuevos de la Secretaría de Finanzas (con el premio calculado contra BYMA) y la
    próxima licitación. Devuelve (subastas, colocaciones, próxima, info). Si algo falla, se queda con el respaldo."""
    hoy = hoy or date.today()
    base = RESPALDO["licitaciones"]
    subs, filas, prox = list(base["subastas"]), list(base["filas"]), None
    info = {"fuente": "respaldo", "nuevas": [], "error": None}
    if usar_api:
        try:
            import requests
            import urllib3
            urllib3.disable_warnings()
            ses = requests.Session()
            ses.headers["User-Agent"] = "Mozilla/5.0 (Bond Analytics, trabajo práctico UCA)"
            lista = listar_noticias(ses.get(LISTADO, timeout=(5, 15)).text)
            ult = max(date.fromisoformat(x["f"]) for x in subs)
            S = {"cer": cer, "tamar": tamar, "a3500": serie_a3500(a3500)}
            cache = {}

            def historia(t):
                if t not in cache:
                    try:
                        cache[t] = _historia(ses, t, ult - timedelta(days=10), hoy)
                    except Exception:
                        cache[t] = {}
                return cache[t]

            for f, url, tit in sorted(x for x in lista if es_resultado(x[2]) and x[0] > ult):
                sub, crudas = leer_resultado(ses.get(url, timeout=(5, 15)).text, f)
                nuevas = calcular_filas(f, crudas, S, historia, hoy)
                if nuevas:
                    subs.append(sub); filas = nuevas + filas; info["nuevas"].append(f.isoformat())
            ult = max(date.fromisoformat(x["f"]) for x in subs)
            llam = sorted((x for x in lista if es_llamado(x[2]) and x[0] >= ult), reverse=True)
            if llam:
                prox = leer_llamado(ses.get(llam[0][1], timeout=(5, 15)).text, llam[0][0])
            info["fuente"] = "Secretaría de Finanzas (en vivo)"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:200]
    sub = pd.DataFrame(subs)
    sub["f"], sub["liq"] = pd.to_datetime(sub["f"]).dt.date, pd.to_datetime(sub["liq"]).dt.date
    cols = ["fecha", "liquida", "ticker", "familia", "nueva", "en_usd", "vno_ofertado", "vno_adjudicado", "valor_efectivo", "precio", "tirea_corte",
            "tirea_propia", "dias", "tirea_secundario", "referencia", "premio_pb", "premio_precio", "tirea_al_liquidar", "fecha_post", "cambio_post_pb",
            "resultado_al_liquidar"]
    col = pd.DataFrame(filas, columns=cols)
    col["fecha"], col["liquida"] = pd.to_datetime(col["fecha"]).dt.date, pd.to_datetime(col["liquida"]).dt.date
    col = col.drop_duplicates(subset=["fecha", "ticker"], keep="first")
    return sub.drop_duplicates(subset=["f"]).sort_values("f"), col, proxima(hoy, prox), info


# ───────────────────────────── calendario ─────────────────────────────
def _bopreales():
    """Flujos de los Bopreales (BCRA) por 100 VN: Serie 1 (A a D) 5% semestral, amortiza 50% en abr-27 y 50% en oct-27;
    Serie 4 (A y B) 3% semestral, amortiza al vencimiento (oct-28)."""
    out = {}
    s1 = [date(2026, 4, 30), date(2026, 10, 31), date(2027, 4, 30), date(2027, 10, 31)]
    for k in "ABCD":
        res, fl = 100.0, []
        for i, f in enumerate(s1[1:], 1):
            am = 50.0 if f >= date(2027, 4, 30) else 0.0
            fl.append((f, res * 0.05 / 2, am)); res -= am
        out[f"BPO{k}7"] = {"flujos": fl, "venc": date(2027, 10, 31)}
    s4 = [date(2026, 10, 31), date(2027, 4, 30), date(2027, 10, 31), date(2028, 4, 30), date(2028, 10, 31)]
    for k in "AB":
        out[f"BPO{k}8"] = {"flujos": [(f, 1.5, 100.0 if f == s4[-1] else 0.0) for f in s4], "venc": date(2028, 10, 31)}
    return out


def eventos(desde, hasta, ons=None):
    """Calendario como DataFrame: fecha, tipo (lla, lic, liq, cor, ren, amo, ven, fer), moneda ('usd', 'ars', 'on' o ''), ticker y detalle.
    Las fechas de pago que caen en feriado o fin de semana pasan al día hábil siguiente; el corte de cupón (registro) es el día hábil
    anterior al pago. ons: [(ticker, emisor, clase, flujos [(fecha, renta, amortización)], vencimiento, frecuencia supuesta)] de
    externos.ons_calculables: sus fechas se cuentan hacia atrás desde el vencimiento, así que son estimadas."""
    ev = []
    add = lambda f, tipo, mon, t, txt: ev.append({"fecha": f, "tipo": tipo, "moneda": mon, "ticker": t, "detalle": txt}) if desde <= f <= hasta else None
    C = RESPALDO["calendario"]
    prox = C["proxima"]
    for c in C["cronograma"]:
        ll, li, lq = (date.fromisoformat(c[k]) for k in ("llamado", "lic", "liq"))
        add(ll, "lla", "ars", "", "Llamado a licitación: la Secretaría de Finanzas publica el menú" + ("" if habil(ll) else " (ese día es feriado: es probable que se adelante)"))
        add(li, "lic", "ars", "", f"Licitación del Tesoro (ofertas de 10 a 15 hs), liquida el {lq:%d/%m/%Y}" + (" · menú: " + ", ".join(m[0] for m in prox["menu"]) if c["lic"] == prox["lic"] else ""))
        add(lq, "liq", "ars", "", f"Liquidación de la licitación del {li:%d/%m/%Y}")
    for f, n in C["feriados"]:
        add(date.fromisoformat(f), "fer", "", "", n)
    # bonos en dólares (Bonares, Globales, Tesoro USD y Bopreales)
    usd = {**B.universo_usd(), **_bopreales()}
    for t, b in usd.items():
        for f, cup, am in b["flujos"]:
            pago = siguiente_habil(f)
            if pago < desde or f > hasta + timedelta(days=5):
                continue
            fin = f >= b["venc"]
            tipo = "ven" if fin else "amo" if am > 0 else "ren"
            add(pago, tipo, "usd", t, (f"vence: US$ {_n(am)} de capital + US$ {_n(cup, 3)} de renta" if fin else f"amortiza US$ {_n(am)} + renta US$ {_n(cup, 3)}" if am > 0 else f"paga renta de US$ {_n(cup, 3)}") + " por 100 VN")
            add(anterior_habil(pago), "cor", "usd", t, f"corte de cupón del pago del {pago:%d/%m/%Y}: comprar hoy en CI o ayer en 24 hs para cobrarlo")
    # bonos en pesos
    for t, tipo, em, v, tem in P.FIJA:
        vf = 100 * (1 + tem) ** (P.d360(date.fromisoformat(em), date.fromisoformat(v)) / 30)
        add(P.ajustar_pago(date.fromisoformat(v)), "ven", "ars", t, f"vence ({tipo}): paga $ {_n(vf)} por 100 VN")
    add(P.ajustar_pago(date(2026, 10, 17)), "ven", "ars", "TO26", "vence (BONTE): paga $ 107,75 por 100 VN (capital + último cupón)")
    f = date(2025, 11, 30)
    while f <= date(2030, 5, 30):
        pago = P.ajustar_pago(f)
        if f == date(2030, 5, 30):
            add(pago, "ven", "ars", "TY30P", "vence (BONTE): paga $ 114,75 por 100 VN")
        else:
            add(pago, "ren", "ars", "TY30P", "paga cupón de $ 14,75 por 100 VN (tasa fija 29,5%)"); add(anterior_habil(pago), "cor", "ars", "TY30P", f"corte de cupón del pago del {pago:%d/%m/%Y}")
        f = P.sumar_meses(f, 6)
    for t, tipo, em, v, fb in P.CERZ:
        add(P.ajustar_pago(date.fromisoformat(v)), "ven", "ars", t, f"vence ({tipo}): paga el capital ajustado por CER")
    for t, em, v, fb, tasa, p1, am0, n, cond in P.BONCER:
        res, ini, f = 100.0, date.fromisoformat(em), date.fromisoformat(p1)
        while f <= date.fromisoformat(v):
            am = 100 / n if f >= date.fromisoformat(am0) else 0.0
            pago, fin = P.ajustar_pago(f), f == date.fromisoformat(v)
            add(pago, "ven" if fin else "amo" if am else "ren", "ars", t, f"{'vence: ' if fin else ''}paga {f'amortización de {_n(am)} + ' if am else ''}renta de {_n(res * tasa * P.d360(ini, f) / 360, 3)} por 100 VN, ajustadas por CER")
            add(anterior_habil(pago), "cor", "ars", t, f"corte de cupón del pago del {pago:%d/%m/%Y}")
            res -= am; ini = f; f = P.sumar_meses(f, 6)
    for t, tipo, v, (fd, fh, m1, m2), tasa_de, am0, capf in P.LARGOS:
        fechas = P._fechas_sem(fd, fh, m1, m2) + ([date.fromisoformat(v)] if tipo == "Par" else [])
        res = 100 * capf
        for ini, f in zip(fechas, fechas[1:]):
            am = 5 * capf if f >= date.fromisoformat(am0) else 0.0
            pago, fin = P.ajustar_pago(f), f == date.fromisoformat(v)
            add(pago, "ven" if fin else "amo" if am else "ren", "ars", t, f"{tipo}: paga {f'amortización de {_n(am)} + ' if am else ''}renta de {_n(res * tasa_de(ini) * P.d360(ini, f) / 360, 3)} por 100 VN originales, ajustadas por CER")
            add(anterior_habil(pago), "cor", "ars", t, f"corte de cupón del pago del {pago:%d/%m/%Y}")
            res -= am
    for t, em, v, mg in P.TAMAR:
        add(P.ajustar_pago(date.fromisoformat(v)), "ven", "ars", t, f"vence (TAMAR + {_n(mg)}%): paga capital más TAMAR capitalizada")
    for d in P.DUALES:
        add(P.ajustar_pago(date.fromisoformat(d["v"])), "ven", "ars", d["t"], f"vence (dual): {d['cond'].lower()}")
    for t, em, v in P.DL:
        add(P.ajustar_pago(date.fromisoformat(v)), "ven", "ars", t, "vence (dólar linked): paga USD 100 por lámina al A3500 de 3 hábiles antes")
    # obligaciones negociables (fechas estimadas desde el vencimiento con la frecuencia de la ficha de BYMA)
    for t, emisor, clase, flujos, venc, supuesta in (ons or []):
        for f, cup, am in flujos:
            pago = siguiente_habil(f)
            if not desde <= pago <= hasta:
                continue
            fin = f >= venc
            tipo = "ven" if fin else "amo" if am > 0 else "ren"
            txt = (f"vence: US$ {_n(am)} de capital + US$ {_n(cup, 3)} de renta" if fin else
                   f"amortiza US$ {_n(am)} + renta US$ {_n(cup, 3)}" if am > 0 else f"paga renta de US$ {_n(cup, 3)}")
            add(pago, tipo, "on", t, f"ON de {emisor} ({clase}): {txt} por 100 VN · fecha estimada" + (" (pagos semestrales supuestos)" if supuesta else ""))
    orden = {k: i for i, k in enumerate(["lic", "lla", "liq", "fer", "ven", "amo", "ren", "cor"])}
    df = pd.DataFrame(ev)
    df["o"] = df["tipo"].map(orden)
    return df.sort_values(["fecha", "o", "ticker"]).drop(columns="o").reset_index(drop=True)


def agrupar(ev):
    """Junta en una fila los eventos del mismo día, tipo y detalle (por ejemplo, los cortes de cupón de varios bonos)."""
    g = ev.groupby(["fecha", "tipo", "detalle"], sort=False)["ticker"].apply(lambda s: ", ".join(x for x in s if x)).reset_index()
    return g[["fecha", "tipo", "ticker", "detalle"]]
