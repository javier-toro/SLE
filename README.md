# Simulación de Schramm–Loewner

El proyecto compara la proporción de poligonales simples en el modelo clásico y en el modelo con intensidad variable. El cuaderno conserva las implementaciones de Kennedy y zipper, explica el cómputo Monte Carlo e incluye las figuras de los experimentos 1–5. Los experimentos 6–10 están al final, en **Extras**.

## Archivos

- **[notebook/SLE.ipynb](notebook/SLE.ipynb)** contiene el código y las figuras intercaladas con las explicaciones.
- **[simulaciones.py](simulaciones.py)** ejecuta el mismo cálculo desde PowerShell con CuPy y CUDA.
- **[resultados/resumen.csv](resultados/resumen.csv)** contiene una copia del resumen exportado. Los experimentos principales llegaron a 100.000 trayectorias por combinación.
- **[resultados/graficos/Indice.html](resultados/graficos/Indice.html)** reúne las figuras exportadas. Las figuras de los extras pueden corresponder a un avance parcial.
- **[resultados/animaciones/](resultados/animaciones)** contiene los MP4, los HTML de reproducción y los datos de ejemplo.
- **[resultados/presentacion/](resultados/presentacion)** contiene el PDF Beamer y su ZIP de Overleaf. Las animaciones del PDF se reproducen en un visor compatible con el paquete animate.

## Instalar y ejecutar

Desde la raíz de SLE en Windows, usa **Python 3.12** y un controlador NVIDIA compatible con CUDA 13. Las versiones en requirements.txt corresponden al entorno con el que se preparó el proyecto.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
~~~

Abre notebook/SLE.ipynb desde JupyterLab. Las figuras guardadas en el cuaderno se pueden consultar directamente. Las llamadas de cálculo están comentadas para que abrir o ejecutar el cuaderno completo no lance el barrido largo.

### Ejecutar el cómputo desde PowerShell

~~~powershell
# Prueba con 10 trayectorias por combinación
.\.venv\Scripts\python.exe simulaciones.py --max-m 10 --output resultados\prueba

# Experimentos 1–5
.\.venv\Scripts\python.exe simulaciones.py

# Experimentos 1–10, reutilizando el mismo recálculo
.\.venv\Scripts\python.exe simulaciones.py --extras
~~~

Por defecto, el script guarda el recálculo en resultados/recalculo. La carpeta trayectorias contiene un CSV por configuración con una fila por semilla. El archivo resumen.csv agrega las métricas para los tamaños de muestra completos. Para reanudar una ejecución, repite el mismo comando y usa la misma carpeta de salida. Puedes crear un archivo vacío llamado STOP en esa carpeta para detener el cálculo al final del lote actual. Retíralo antes de reanudar.

Esta copia del proyecto incluye los resúmenes exportados. Un recálculo crea sus propios CSV de trayectorias.
