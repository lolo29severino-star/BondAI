"""
herramientas.py — cinco herramientas de BondAI que se apoyan en los cálculos de los otros módulos.

  1. Fichas de las ONs de los grandes emisores y su curva contra la de los soberanos   → spread_ons, EMISORES_TOP
  2. Retorno de los bonos en dólares hasta una fecha según la TIR a la que se venda      → retorno_hasta, tabla_retornos
  3. Carry trade con bandas: rendimiento en dólares de una LECAP según el dólar final   → banda, tabla_carry, curva_carry
  4. Mercado de dinero: caución, plazo fijo, billeteras y fondos money market            → cargar_dinero, resumen_dinero
  5. Tu cartera contra un índice de bonos                                                → comparar_indice

La 1 se usa en la pestaña «ONs», la 2 en «Curva en dólares», la 5 en «Mi portafolio» y la 3 y la 4 en el informe semanal.

Fuentes en vivo: BYMA Open Data (cauciones), ArgentinaDatos (plazo fijo, cuentas remuneradas y fondos money market).
Si no responden, se usa el respaldo del 20/10/2026 que está en este archivo.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

import bondai as B
import pesos as P

BYMA_API = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
AD_API = "https://api.argentinadatos.com/v1/finanzas"

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


# ───────────────────────────── 3. carry trade con bandas ─────────────────────────────
# Banda cambiaria (BCRA): desde enero de 2026 el techo sube y el piso baja cada mes con la inflación de dos meses antes.
# Valores del 30/11/2026 publicados con el IPC de septiembre (1,93%): techo $1.989,12 y piso $698,44; techo del 30/10: $1.950,97.
BANDA_ANCLAS = [(date(2026, 10, 30), 1950.97), (date(2026, 11, 30), 1989.12)]
PISO_ANCLA = (date(2026, 11, 30), 698.44)


def _ultimo_dia(f):
    return date(f.year + (f.month == 12), f.month % 12 + 1, 1) - timedelta(days=1)


def techo(fecha, ipc):
    """Techo de la banda a una fecha. Hasta el 30/11/2026 se interpola entre los valores publicados; después crece cada mes con
    la inflación de dos meses antes (ipc: 'AAAA-MM' → decimal, con el IPC publicado y después el REM)."""
    (f0, v0), (f1, v1) = BANDA_ANCLAS
    if fecha <= f0:
        return v0
    if fecha <= f1:
        return v0 * (v1 / v0) ** ((fecha - f0).days / (f1 - f0).days)
    v, f = v1, f1
    while True:
        fin = _ultimo_dia(f + timedelta(days=1))
        crec = 1 + ipc(str(pd.Period(fin, "M") - 2))
        if fecha <= fin:
            return v * crec ** ((fecha - f).days / (fin - f).days)
        v, f = v * crec, fin


def banda(fecha, ipc):
    """(piso, techo) a una fecha: el piso baja al mismo ritmo al que sube el techo."""
    t = techo(fecha, ipc)
    return PISO_ANCLA[1] * techo(PISO_ANCLA[0], ipc) / t, t


def tabla_carry(RP, spot, ipc, futuro=None, brecha=1.0):
    """Carry trade con cada LECAP y BONCAP: se venden dólares hoy a 'spot', se compra la letra y al vencimiento se vuelven a comprar
    dólares. futuro(días) → dólar futuro de A3 interpolado (o None). brecha: dólar de referencia / A3500 (1 si se usa el A3500;
    MEP / A3500 si se mide con el MEP, suponiendo que la brecha de hoy se mantiene)."""
    filas = []
    for k in ("lecap", "boncap"):
        df = RP[k]
        if df.empty:
            continue
        for t, r in df[df["pago_final"].notna() & (df["dias"] >= 7)].iterrows():
            d, R = int(r["dias"]), float(r["pago_final"] / r["precio"])
            v = P.ajustar_pago(r["vence"])
            piso, tec = banda(v, ipc)
            esc = {"baja_5": spot * 0.95, "igual": spot, "futuro": futuro(d) * brecha if futuro and np.isfinite(futuro(d)) else np.nan,
                   "medio": (spot + tec * brecha) / 2, "techo": tec * brecha}
            fila = {"ticker": t, "tipo": "LECAP" if k == "lecap" else "BONCAP", "vence": v, "dias": d, "tirea": r["tirea"], "tem": r["tem"],
                    "dolar_be": spot * R, "piso": piso * brecha, "be_vs_techo": spot * R / (tec * brecha) - 1}
            for e, fx in esc.items():          # r_*: ganancia en dólares hasta el vencimiento en cada escenario; fx_*: dólar al vencimiento
                fila["r_" + e] = R * spot / fx - 1 if fx and np.isfinite(fx) else np.nan
                fila["fx_" + e] = fx
            fila["tea_igual"] = R ** (365 / d) - 1             # TEA en dólares si el dólar no se mueve (= la TIREA en pesos)
            filas.append(fila)
    return pd.DataFrame(filas).set_index("ticker").sort_values("dias") if filas else pd.DataFrame()


def curva_carry(R, spot, desde, hasta, n=80):
    """Ganancia en dólares hasta el vencimiento de la letra para cada dólar final entre 'desde' y 'hasta'."""
    fx = np.linspace(desde, hasta, n)
    return fx, R * spot / fx - 1


# ───────────────────────────── 4. mercado de dinero ─────────────────────────────
RESPALDO_DINERO = {
    "fecha": "2026-10-20",
    # caución en pesos (BYMA, contado inmediato): [días, TNA promedio operada, TNA cierre anterior, monto operado $]
    "caucion": [[1, 0.2084, 0.176, 3305834431811], [2, 0.2072, 0.189, 5462388188], [3, 0.2101, 0.187, 6398560848], [6, 0.208, 0.19, 2973901997],
                [7, 0.2171, 0.207, 106211156271], [8, 0.2109, 0.201, 727523220], [9, 0.2135, 0.205, 551272523], [10, 0.2173, 0.201, 610909186],
                [13, 0.2112, 0.206, 218539271], [14, 0.2242, 0.21, 4855527942], [15, 0.2271, 0.205, 1988563573], [16, 0.2206, 0.22, 194016698],
                [22, 0.2256, 0.215, 67514787], [23, 0.215, 0.217, 9221839], [24, 0.2255, 0.202, 28175049], [27, 0.2265, 0.22, 35011035],
                [28, 0.2076, 0.205, 25467095], [29, 0.2298, 0.211, 50437683], [30, 0.2308, 0.205, 827886444], [31, 0.2157, 0.22, 5025988],
                [35, 0.2254, 0.202, 6231169], [38, 0.23, 0.23, 3071836]],
    # plazo fijo a 30 días (BCRA vía ArgentinaDatos): [entidad, TNA clientes, TNA no clientes]
    "plazo_fijo": [["BANCO DE LA NACION ARGENTINA", 0.1925, 0], ["BANCO DE GALICIA Y BUENOS AIRES S.A.", 0.1825, 0], ["BANCO BBVA ARGENTINA S.A.", 0.19, 0],
                   ["BANCO SANTANDER ARGENTINA S.A.", 0.17, 0], ["BANCO DE LA PROVINCIA DE BUENOS AIRES", 0.21, 0.22], ["BANCO MACRO S.A.", 0.21, 0],
                   ["INDUSTRIAL AND COMMERCIAL BANK OF CHINA (ARGENTINA) S.A.U.", 0.181, 0], ["BANCO DE LA CIUDAD DE BUENOS AIRES", 0.17, 0],
                   ["BANCO PATAGONIA S.A.", 0.16, 0], ["BANCO CREDICOOP COOPERATIVO LIMITADO", 0.185, 0.185], ["BANCO BICA S.A.", 0.23, 0.23],
                   ["BANCO CMF S.A.", 0.235, 0.235], ["BANCO COMAFI SOCIEDAD ANONIMA", 0.19, 0.19], ["BANCO DE COMERCIO S.A.", 0.225, 0.225],
                   ["BANCO DE FORMOSA S.A.", 0.185, 0.185], ["BANCO DE LA PROVINCIA DE CORDOBA S.A.", 0.2075, 0.2075], ["BANCO DEL CHUBUT S.A.", 0.2, 0.2],
                   ["BANCO DEL SOL S.A.", 0.23, 0.23], ["BANCO DINO S.A.", 0.2, 0.2], ["BANCO HIPOTECARIO S.A.", 0.195, 0.22], ["BANCO JULIO SOCIEDAD ANONIMA", 0.19, 0.19],
                   ["BANCO MARIVA S.A.", 0.23, 0.225], ["BANCO MASVENTAS S.A.", 0.21, 0.19], ["BANCO MERIDIAN S.A.", 0.2375, 0.2375],
                   ["BANCO PROVINCIA DE TIERRA DEL FUEGO", 0.19, 0.15], ["BANCO VOII S.A.", 0.23, 0.23], ["BIBANK S.A.", 0.215, 0.215],
                   ["CRÉDITO REGIONAL COMPAÑÍA FINANCIERA S.A.U.", 0.235, 0.235], ["REBA COMPAÑIA FINANCIERA S.A.", 0.24, 0.24], ["Banco Piano", 0.175, 0.175],
                   ["Brubank", 0.17, 0.17], ["UALA", 0.22, 0.22]],
    # cuentas remuneradas y billeteras (ArgentinaDatos, fci/otros): [nombre, TNA, tope $, condiciones, fecha del dato]
    "billeteras": [["BELO", 0.19, 0, "", "2026-10-20"], ["BICA CUENTA POSITIVA 4", 0.22, 750000, "de $1 hasta $750.000", "2026-10-20"],
                   ["BNA", 0.14, 2000000, "Aplicable para clientes de cuenta sueldo.", "2026-10-20"], ["BRUBANK", 0.27, 750000, "", "2025-08-06"],
                   ["CARREFOUR BANCO", 0.2, 0, "Permite ingresar $ 4.694.400 / mes", "2026-10-20"], ["FIWIND", 0.2, 750000, "", "2026-10-20"],
                   ["MONTEMAR PAY", 0.19, 0, "", "2026-10-20"], ["NARANJA X", 0.2, 1000000, "", "2026-10-20"],
                   ["NARANJA X FRASCOS 28", 0.2, 30000000, "Plazo elegible entre 7 y 28 días.", "2026-10-20"],
                   ["SUPERVIELLE", 0.14, 1000000, "Solo Clientes Plan Sueldo.", "2026-10-20"], ["UALA", 0.19, 1000000, "Mantener entre $10.000 y $1.000.000.", "2026-10-20"],
                   ["UALA PLUS 1", 0.2, 1000000, "Acumular $50.000 en ciertas operaciones este mes.", "2026-10-20"],
                   ["UALA PLUS 2", 0.24, 1000000, "Acumular $500.000 en ciertas operaciones.", "2026-10-20"]],
    # fondos money market en pesos (CNV vía ArgentinaDatos): [fondo, fecha, fecha anterior, valor cuotaparte, valor anterior, patrimonio $ millones]
    "fci": [["Mercado Fondo - Clase A", "2026-10-16", "2026-10-15", 26031.053, 26017.393, 8714187], ["Fima Premium - Clase B", "2026-10-16", "2026-10-15", 103995.88, 103941.1, 4510619],
            ["Fima Premium - Clase A", "2026-10-16", "2026-10-15", 86021.448, 85979.602, 3624268],
            ["FBA Money Market Pesos - Clase B", "2026-10-16", "2026-10-15", 198881.452, 198777.393, 3518546],
            ["Pionero Pesos Plus - Clase B", "2026-10-16", "2026-10-15", 64182.818, 64149.495, 2657837], ["Super Ahorro $ - Clase A", "2026-10-16", "2026-10-15", 20265.809, 20255.752, 2281734],
            ["Super Ahorro $ - Clase B", "2026-10-16", "2026-10-15", 24620.757, 24607.661, 1991979],
            ["Pellegrini Renta Pesos - Clase B", "2026-10-16", "2026-10-15", 27467.088, 27451.21, 1916535], ["Alpha Pesos - Clase A", "2026-10-16", "2026-10-15", 105339.57, 105284.027, 1847715],
            ["Lombard Renta en Pesos", "2026-10-16", "2026-10-15", 253107.589, 252961.718, 1604089], ["1810 Ahorro", "2026-10-16", "2026-10-15", 242182.465, 242053.561, 1586430],
            ["1822 Raices Ahorro Pesos - Clase B", "2026-10-16", "2026-10-15", 38669.938, 38647.871, 1543702],
            ["Super Ahorro Plus - Clase A", "2026-10-16", "2026-10-15", 665276.789, 664840.07, 1438305], ["Fima Premium - Clase C", "2026-10-16", "2026-10-15", 105687.395, 105630.696, 970150],
            ["Delta Pesos - Clase B", "2026-10-16", "2026-10-15", 103156.168, 103104.088, 677294], ["IAM Ahorro Pesos - Clase B", "2026-10-16", "2026-10-15", 54139.804, 54110.445, 584778],
            ["Premier Renta CP en Pesos - Clase C", "2026-10-16", "2026-10-15", 26252.18, 26237.28, 581499],
            ["FBA Money Market Pesos - Clase A", "2026-10-16", "2026-10-15", 180654.641, 180567.052, 490212],
            ["Max Money Market - Clase B", "2026-10-16", "2026-10-15", 5679.574, 5676.488, 451904], ["MEGAQM Pesos - Clase B", "2026-10-16", "2026-10-15", 57799.863, 57767.924, 451037],
            ["Adcap Ahorro Pesos Fondo de Dinero - Clase B", "2026-10-16", "2026-10-15", 20635.948, 20624.147, 431504],
            ["Balanz Capital Money Market - Clase D", "2026-10-16", "2026-10-15", 13269.393, 13262.253, 404678],
            ["Toronto Trust Ahorro - Clase B", "2026-10-16", "2026-10-15", 60707.344, 60676.537, 388167], ["Allaria Ahorro - Clase C", "2026-10-16", "2026-10-15", 33943.726, 33926.222, 324610],
            ["Ualintec Ahorro Pesos - Clase A", "2026-10-16", "2026-10-15", 8695.032, 8690.114, 160380], ["Cocos Ahorro - Clase C", "2026-10-16", "2026-10-15", 2050.228, 2049.176, 48121]],
}
BILLETERA_FCI = {"Mercado Pago (Mercado Fondo)": "Mercado Fondo - Clase A", "Ualá (Ualintec Ahorro)": "Ualintec Ahorro Pesos - Clase A",
                 "Cocos (Cocos Ahorro)": "Cocos Ahorro - Clase C"}
BANCOS_GRANDES = {"BANCO DE LA NACION ARGENTINA": "Nación", "BANCO DE GALICIA Y BUENOS AIRES S.A.": "Galicia", "BANCO BBVA ARGENTINA S.A.": "BBVA",
                  "BANCO SANTANDER ARGENTINA S.A.": "Santander", "BANCO DE LA PROVINCIA DE BUENOS AIRES": "Provincia", "BANCO MACRO S.A.": "Macro"}
NOMBRE_BILLETERA = {"UALA": "Ualá", "NARANJA X": "Naranja X", "BELO": "Belo", "BRUBANK": "Brubank", "CARREFOUR BANCO": "Carrefour Banco"}


def _tea_simple(tna, dias):
    return (1 + tna * dias / 365) ** (365 / dias) - 1


def cargar_dinero(usar_api=True):
    """Cauciones, plazo fijo, billeteras y fondos money market. Devuelve (dict de DataFrames, {parte: fuente})."""
    R = RESPALDO_DINERO
    cau, pf, bil, fci = R["caucion"], R["plazo_fijo"], R["billeteras"], R["fci"]
    fuente = {k: f"respaldo del {pd.Timestamp(R['fecha']):%d/%m/%Y}" for k in ("caucion", "plazo_fijo", "billeteras", "fci")}
    if usar_api:
        import requests
        import urllib3
        urllib3.disable_warnings()
        ses = requests.Session(); ses.headers["User-Agent"] = "Mozilla/5.0 (BondAI)"
        try:
            r = ses.post(f"{BYMA_API}/cauciones", json={"excludeZeroPxAndQty": True, "page_size": 500, "T0": True, "T1": False, "T2": False},
                         headers={"Content-Type": "application/json"}, verify=False, timeout=(4, 10))
            j = r.json(); j = j if isinstance(j, list) else j.get("data", [])
            x = [[int(z["daysToMaturity"]), float(z.get("vwap") or z.get("trade") or 0), float(z.get("previousClosingPrice") or 0), float(z.get("volumeAmount") or 0)]
                 for z in j if z.get("denominationCcy") == "ARS"]
            x = [z for z in x if z[1] > 0 or z[2] > 0]
            if len(x) >= 3:
                cau, fuente["caucion"] = sorted(x), "BYMA (en vivo)"
        except Exception:
            pass
        try:
            j = ses.get(f"{AD_API}/tasas/plazoFijo", timeout=(4, 10)).json()
            if len(j) > 5:
                pf, fuente["plazo_fijo"] = [[z["entidad"], z.get("tnaClientes") or 0, z.get("tnaNoClientes") or 0] for z in j], "ArgentinaDatos (en vivo)"
        except Exception:
            pass
        try:
            j = ses.get(f"{AD_API}/fci/otros/ultimo", timeout=(4, 10)).json()
            if len(j) > 3:
                bil, fuente["billeteras"] = [[z["fondo"], z.get("tna") or 0, z.get("tope") or 0, z.get("condicionesCorto") or "", z.get("fecha")] for z in j], "ArgentinaDatos (en vivo)"
        except Exception:
            pass
        try:
            u = ses.get(f"{AD_API}/fci/mercadoDinero/ultimo", timeout=(4, 15)).json()
            p = {z["fondo"]: z for z in ses.get(f"{AD_API}/fci/mercadoDinero/penultimo", timeout=(4, 15)).json()}
            x = [[z["fondo"], z["fecha"], p[z["fondo"]]["fecha"], z["vcp"], p[z["fondo"]]["vcp"], (z.get("patrimonio") or 0) / 1e6] for z in u
                 if z["fondo"] in p and z.get("vcp") and p[z["fondo"]].get("vcp") and "lar" not in z["fondo"].lower() and "usd" not in z["fondo"].lower()]
            if len(x) > 10:
                fci, fuente["fci"] = x, "ArgentinaDatos / CNV (en vivo)"
        except Exception:
            pass
    C = pd.DataFrame(cau, columns=["dias", "tna_operada", "tna_cierre", "monto"])
    C["tna"] = C["tna_operada"].where(C["tna_operada"] > 0, C["tna_cierre"])
    C["tea"] = [_tea_simple(t, d) for t, d in zip(C["tna"], C["dias"])]
    PF_ = pd.DataFrame(pf, columns=["entidad", "tna_clientes", "tna_no_clientes"])
    PF_["tna"] = PF_["tna_clientes"].where(PF_["tna_clientes"] > 0, PF_["tna_no_clientes"])
    PF_ = PF_[PF_["tna"] > 0]
    PF_["tea"] = [_tea_simple(t, 30) for t in PF_["tna"]]
    BI = pd.DataFrame(bil, columns=["nombre", "tna", "tope", "condiciones", "fecha"])
    BI = BI[BI["tna"] > 0]
    BI["tea"] = (1 + BI["tna"] / 365) ** 365 - 1
    FC = pd.DataFrame(fci, columns=["fondo", "fecha", "fecha_ant", "vcp", "vcp_ant", "patrimonio"])
    FC["dias"] = [max((pd.Timestamp(a) - pd.Timestamp(b)).days, 1) for a, b in zip(FC["fecha"], FC["fecha_ant"])]
    FC = FC[FC["dias"] <= 5]
    FC["tna"] = (FC["vcp"] / FC["vcp_ant"] - 1) * 365 / FC["dias"]
    FC["tea"] = (FC["vcp"] / FC["vcp_ant"]) ** (365 / FC["dias"]) - 1
    FC = FC[(FC["tna"] > 0) & (FC["tna"] < 1)].sort_values("patrimonio", ascending=False)
    return {"caucion": C, "plazo_fijo": PF_.sort_values("tna", ascending=False), "billeteras": BI.sort_values("tna", ascending=False), "fci": FC}, fuente


def resumen_dinero(Dn, lecap):
    """Una fila por alternativa (TNA, TEA, plazo y diferencia contra la LECAP más corta). lecap: fila de la LECAP de referencia."""
    filas = []
    add = lambda nombre, tipo, tna, tea, plazo, nota="": filas.append({"alternativa": nombre, "tipo": tipo, "tna": tna, "tea": tea, "plazo": plazo, "nota": nota})
    C = Dn["caucion"]
    for d_obj, nom in ((1, "Caución 1 día"), (7, "Caución 7 días"), (30, "Caución 30 días")):
        if len(C):
            r = C.iloc[(C["dias"] - d_obj).abs().argsort().iloc[0]]
            add(f"{nom}" if r["dias"] == d_obj else f"{nom} ({int(r['dias'])} d)", "Caución bursátil", r["tna"], r["tea"],
                f"{int(r['dias'])} día" + ("s" if r["dias"] != 1 else ""),
                "colocadora: prestás pesos con garantía de BYMA")
    PF_ = Dn["plazo_fijo"]
    gr = PF_[PF_["entidad"].isin(BANCOS_GRANDES)]
    if len(gr):
        add("Plazo fijo, bancos grandes (mediana)", "Plazo fijo", gr["tna"].median(), _tea_simple(gr["tna"].median(), 30), "30 días",
            ", ".join(sorted(BANCOS_GRANDES[e] for e in gr["entidad"])))
    if len(PF_):
        m = PF_.iloc[0]
        add(f"Plazo fijo, mejor tasa ({m['entidad'].title()[:28]})", "Plazo fijo", m["tna"], m["tea"], "30 días", "tasa más alta informada al BCRA")
    FC = Dn["fci"]
    for nom, fondo in BILLETERA_FCI.items():
        x = FC[FC["fondo"] == fondo]
        if len(x):
            add(nom, "Billetera (fondo money market)", x["tna"].iloc[0], x["tea"].iloc[0], "a la vista", f"rendimiento del último día ({x['fecha'].iloc[0]})")
    BI = Dn["billeteras"]
    viejo = (pd.Timestamp(date.today()) - pd.Timedelta(days=60)).date().isoformat()
    for nom in NOMBRE_BILLETERA:
        x = BI[(BI["nombre"] == nom) & (BI["fecha"].astype(str) >= viejo)]          # sin datos viejos (Brubank tenía uno de 2025)
        if len(x):
            tope = f"hasta $ {x['tope'].iloc[0]:,.0f}".replace(",", ".") if x["tope"].iloc[0] else "sin tope"
            add(f"{NOMBRE_BILLETERA[nom]} (cuenta remunerada)", "Cuenta remunerada", x["tna"].iloc[0], x["tea"].iloc[0], "a la vista", tope)
    if len(FC):
        top = FC.head(10)
        add("Fondos money market (mediana de los 10 más grandes)", "Fondo money market", top["tna"].median(), top["tea"].median(), "a la vista", "rescate en el día")
    if lecap is not None:
        add(f"LECAP más corta ({lecap.name}, {int(lecap['dias'])} días)", "LECAP", lecap["tna"], lecap["tirea"], f"{int(lecap['dias'])} días",
            "tasa fija del Tesoro, se vende en el mercado (24 hs)")
    T = pd.DataFrame(filas).set_index("alternativa")
    if lecap is not None:
        T["vs_lecap_pb"] = (T["tea"] - lecap["tirea"]) * 1e4
    return T


# ───────────────────────────── 5. cartera contra un índice ─────────────────────────────
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
