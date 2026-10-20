# BondAI · Renta fija argentina con IA

Aplicación en Streamlit que baja precios de BYMA, series del BCRA, futuros de dólar de A3 (ex Matba Rofex), riesgo país (ArgentinaDatos), la curva del Tesoro de EE.UU. y las obligaciones negociables (data912). Calcula curvas, inflación breakeven, devaluación esperada, tasas forward, flujos, dólares y premios de licitación (las licitaciones nuevas se agregan solas desde la Secretaría de Finanzas); muestra emisor, vencimiento y cupón de todas las ONs (ficha técnica de BYMA); arma tu propio portafolio con su riesgo (duración, key rate durations, escenarios de tasas y VaR) y genera un informe semanal en PDF redactado con IA. También tiene carry trade con bandas, mercado de dinero (caución, plazo fijo, billeteras y fondos money market contra la LECAP), retornos de los bonos según la TIR de salida, la curva de ONs contra la de los soberanos, guardar y subir el portafolio y compararlo contra un índice, y alertas. El pronóstico de inflación con machine learning y el Monte Carlo están en el notebook.

Herramienta académica (Programación para Economistas, UCA). No es recomendación de inversión.

## Correrla en tu computadora

```
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501.

## Publicarla en internet (gratis, con GitHub y Streamlit Community Cloud)

### 1. Subir los archivos a GitHub

1. Entrá a https://github.com e iniciá sesión (si no tenés cuenta, creala).
2. Arriba a la derecha: **+** › **New repository**.
3. Nombre: `bondai`. Dejalo en **Public** (también funciona en Private). Tocá **Create repository**.
4. En la página del repositorio vacío, tocá **uploading an existing file**.
5. Arrastrá estos 12 archivos: `app.py`, `bondai.py`, `pesos.py`, `mercado.py`, `dolares.py`, `licitaciones.py`, `externos.py`, `portafolio.py`, `informe.py`, `herramientas.py`, `requirements.txt` y este `README.md`.

   **Si ya tenías la app publicada**: subí todos los archivos juntos (GitHub los reemplaza) y después, en la app, **Manage app** › ⋮ › **Reboot app**. Si subís solo `app.py`, va a fallar porque busca funciones que están en los otros archivos.
6. Abajo, **Commit changes**.

### 2. Publicar la app

1. Entrá a https://share.streamlit.io y elegí **Continue with GitHub**. Autorizá el acceso.
2. Arriba a la derecha: **Create app** › **Yup, I have an app**.
3. Completá:
   - Repository: `tu-usuario/bondai`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (opcional): por ejemplo `bondai-uca` → quedará `https://bondai-uca.streamlit.app`
4. **Advanced settings**:
   - Python version: 3.12.
   - Secrets (opcional, para que el informe semanal lo redacte Claude): pegá una línea así, con tu clave:
     ```
     ANTHROPIC_API_KEY = "sk-ant-..."
     ```
   - **Save**.
5. **Deploy**. La primera vez tarda unos minutos en instalar las librerías.

### 3. Cómo se actualizan los datos

- Cada vez que alguien abre la app o toca algo, si pasaron más de 15 minutos desde la última descarga, baja precios y tasas de nuevo.
- El botón **Actualizar datos ahora** de la barra lateral los baja en el momento.
- El interruptor **Actualizar sola cada 15 minutos** los renueva solo mientras la página está abierta.
- La barra lateral dice a qué hora se bajaron los datos y de qué fuente salió cada parte (API o respaldo).
- Si BYMA o el BCRA no responden, la app usa el respaldo del 09/10/2026 y lo avisa (las ONs, el del 19 y 20/10/2026; las licitaciones, hasta la del 14/10/2026).
- Licitaciones: cada 3 horas la app revisa la página de noticias de la Secretaría de Finanzas; si hay un «Resultado de la licitación» nuevo lo lee, calcula el premio contra BYMA y lo agrega solo. Del último «Llamado a licitación» saca el menú de la próxima.
- ONs: emisor, vencimiento y cupón salen de la ficha técnica de BYMA (guardada para las 293 ONs de data912; las nuevas se consultan solas). Las ONs en dólares a tasa fija también se pueden cargar en **Mi portafolio** y ver en **Flujos de fondos**, y el informe semanal resume cómo se movieron las más operadas.
- Informe semanal: en la última pestaña, **Generar el informe de la semana** compara el último cierre contra el de una semana antes y arma un PDF para descargar. En vivo tarda unos 20 segundos porque baja los cierres de cada bono. Además del dólar, el riesgo país, los bonos, las ONs y la licitación, incluye el carry trade con la banda, el mercado de dinero contra la LECAP más corta, la TIR y el spread de las ONs de los grandes emisores, las alertas activas y la agenda de las próximas dos semanas (licitación, pagos del Tesoro y qué ONs pagan, de qué empresa y qué día).
- Calendario (pestaña **Licitaciones**): además de licitaciones, cortes de cupón, pagos y vencimientos del Tesoro, muestra los pagos de renta, amortización y vencimiento de las ONs en dólares a tasa fija, con el día de la semana y el monto por 100 VN. Las fechas de las ONs son estimadas (se cuentan desde el vencimiento con la frecuencia de la ficha de BYMA y pasan al día hábil siguiente): confirmalas en el aviso de pago de cada emisor.
- Futuros de A3, riesgo país, Tesoro de EE.UU. y ONs se bajan de sus propias APIs públicas; las ONs se renuevan cada 5 minutos.

- Alertas: en la barra lateral, «Alertas». Se revisan cada vez que la app baja datos y quedan guardadas en el link de la página (guardalo en favoritos). Avisan solo mientras la app está abierta.
- Mercado de dinero: caución de BYMA y plazo fijo, cuentas remuneradas y fondos money market de ArgentinaDatos (respaldo del 20/10/2026).

### 4. Cosas a tener en cuenta

- Si nadie la visita durante 12 horas, la app se duerme. Al entrar aparece un botón para despertarla (tarda alrededor de un minuto). Antes de presentar, abrila unos minutos antes.
- Para cambiar algo, editá el archivo en GitHub (ícono del lápiz) y guardá: la app se actualiza sola.
- La clave de Anthropic nunca va dentro de los archivos: solo en Secrets.
