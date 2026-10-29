# BondAI · Renta fija argentina con IA

Aplicación en Streamlit con seis pestañas: **Curva en dólares**, **ONs**, **Tesoro en pesos**, **Mi portafolio**, **Flujos de fondos** e **Informe semanal con IA**. Baja precios de BYMA, series del BCRA y las obligaciones negociables de data912 (con su ficha técnica de BYMA); calcula TIR, duración, convexidad, paridad y curvas de los bonos en dólares, la TIR de las ONs contra la de los soberanos, los bonos en pesos con su breakeven, el riesgo de tu propia cartera (duración, DV01, key rate durations, escenarios de tasas y VaR) y el flujo de fondos de cada bono. El informe semanal en PDF lo redacta Claude con los números que calcula la app, e incluye cómo le fue a tu portafolio en la semana y qué dicen sus indicadores.

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
- ONs: emisor, vencimiento y cupón salen de la ficha técnica de BYMA (guardada para las 293 ONs de data912; las nuevas se consultan solas). Las ONs en dólares a tasa fija también se pueden cargar en **Mi portafolio** y ver en **Flujos de fondos**.
- Informe semanal: en la última pestaña, **Generar el informe de la semana** compara el último cierre contra el de una semana antes y arma un PDF para descargar (en vivo tarda unos 20 segundos). Cubre el dólar, el riesgo país, los bonos en dólares y en pesos, las ONs, la licitación del Tesoro, el carry trade, el mercado de dinero y la agenda de las próximas dos semanas. Con la casilla **Incluir mi portafolio** marcada, suma dos párrafos, una tabla y un gráfico sobre tu cartera: cómo se movieron los precios de sus bonos, cuánto aportó cada uno y qué dicen sus indicadores (duración, DV01, convexidad, tipo de cartera, exposición y VaR), con cuánto del resultado explican las tasas.
- La licitación de la semana se lee sola de la Secretaría de Finanzas (cada 3 horas revisa si hay un «Resultado de la licitación» nuevo).

### 4. Cosas a tener en cuenta

- Si nadie la visita durante 12 horas, la app se duerme. Al entrar aparece un botón para despertarla (tarda alrededor de un minuto). Antes de presentar, abrila unos minutos antes.
- Para cambiar algo, editá el archivo en GitHub (ícono del lápiz) y guardá: la app se actualiza sola.
- La clave de Anthropic nunca va dentro de los archivos: solo en Secrets.
