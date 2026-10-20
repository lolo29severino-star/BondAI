"""
licitaciones.py — licitaciones del Tesoro (premio contra el mercado secundario) y calendario de licitaciones, cortes de cupón
y vencimientos para la pestaña «Licitaciones» de la app BondAI.

Uso desde la app:
    import licitaciones as L
    subastas, colocaciones = L.resultados()          # 18 licitaciones en efectivo de 2026 con el premio de cada colocación
    ev = L.eventos(date(2026, 10, 1), date(2027, 3, 31))   # calendario: licitaciones, cortes, rentas, amortizaciones, feriados
    html = L.mes_html(ev, 2026, 10, hoy=date(2026, 10, 10))  # grilla del mes para mostrar con st.markdown

El premio se calcula con data/build_licitaciones.js (mismo motor que la terminal): TIREA de corte menos la TIREA del secundario
(BYMA, cierre de 24 hs del día de la licitación), las dos con el mismo método. Fuentes: comunicados «Resultado de la licitación»
de la Secretaría de Finanzas y planilla «Colocaciones de deuda»; cronograma 2026 y feriados bancarios del BCRA.
"""
import base64
import calendar
import json
import zlib
from datetime import date, timedelta

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


def proxima():
    """Próxima licitación anunciada (llamado del 09/10/2026)."""
    return RESPALDO["calendario"]["proxima"]


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


def eventos(desde, hasta):
    """Calendario como DataFrame: fecha, tipo (lla, lic, liq, cor, ren, amo, ven, fer), moneda, ticker y detalle.
    Las fechas de pago que caen en feriado o fin de semana pasan al día hábil siguiente; el corte de cupón (registro) es el día hábil
    anterior al pago."""
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
    orden = {k: i for i, k in enumerate(["lic", "lla", "liq", "fer", "ven", "amo", "ren", "cor"])}
    df = pd.DataFrame(ev)
    df["o"] = df["tipo"].map(orden)
    return df.sort_values(["fecha", "o", "ticker"]).drop(columns="o").reset_index(drop=True)


def agrupar(ev):
    """Junta en una fila los eventos del mismo día, tipo y detalle (por ejemplo, los cortes de cupón de varios bonos)."""
    g = ev.groupby(["fecha", "tipo", "detalle"], sort=False)["ticker"].apply(lambda s: ", ".join(x for x in s if x)).reset_index()
    return g[["fecha", "tipo", "ticker", "detalle"]]


COLOR_EV = {"lic": "#2a78d6", "lla": "#8a96aa", "liq": "#8a96aa", "cor": "#b7791f", "ren": "#1baf7a", "amo": "#4a3aa7", "ven": "#eb6834", "fer": "#8a96aa"}


def mes_html(ev, anio, mes, hoy=None):
    """Grilla de un mes (lunes a domingo) con los eventos de cada día, como tabla HTML para st.markdown(unsafe_allow_html=True)."""
    cal = calendar.Calendar(firstweekday=0)
    dias_ev = {}
    for _, e in ev.iterrows():
        dias_ev.setdefault(e["fecha"], []).append(e)
    fer = {e["fecha"] for _, e in ev.iterrows() if e["tipo"] == "fer"}
    css = ("<style>.calbx{width:100%;border-collapse:collapse;table-layout:fixed;font-size:11.5px}.calbx th{background:rgba(128,128,128,.12);padding:4px;"
           "font-weight:600}.calbx td{border:1px solid rgba(128,128,128,.25);vertical-align:top;height:86px;padding:3px 4px;overflow:hidden}"
           ".calbx .n{font-weight:700}.calbx .fuera{opacity:.45}.calbx .hoy .n{background:#2a78d6;color:#fff;border-radius:9px;padding:0 6px}"
           ".calbx .fer{background:repeating-linear-gradient(135deg,transparent 0 6px,rgba(128,128,128,.12) 6px 12px)}"
           ".calbx .ev{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-left:3px solid;padding-left:4px;margin-top:2px}</style>")
    h = css + '<table class="calbx"><tr>' + "".join(f"<th>{d}</th>" for d in ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]) + "</tr>"
    for semana in cal.monthdatescalendar(anio, mes):
        h += "<tr>"
        for d in semana:
            es = dias_ev.get(d, [])
            grupos = {}
            for e in es:
                grupos.setdefault(e["tipo"], []).append(e["ticker"])
            clases = " ".join(c for c, ok in [("fuera", d.month != mes), ("hoy", d == hoy), ("fer", d in fer)] if ok)
            chips = ""
            for tipo in ["lic", "lla", "liq", "fer", "ven", "amo", "ren", "cor"]:
                if tipo in grupos:
                    tk = [t for t in grupos[tipo] if t]
                    txt = TIPOS_EV[tipo] + (" " + " ".join(tk[:3]) + (f" +{len(tk) - 3}" if len(tk) > 3 else "") if tk else "")
                    chips += f'<span class="ev" style="border-color:{COLOR_EV[tipo]}" title="{TIPOS_EV[tipo]}: {", ".join(tk)}">{txt}</span>'
            h += f'<td class="{clases}"><span class="n">{d.day}</span>{chips}</td>'
        h += "</tr>"
    return h + "</table>"
