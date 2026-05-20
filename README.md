
# Guía de Evaluación: Tracker Rust vs Python (MOT17 / MOT20)

El objetivo de este proyecto es implementar el algoritmo TrackTrack en Rust. Este documento detalla las instrucciones para configurar el entorno, ejecutar el pipeline y evaluar el tracker desarrollado en Rust frente a la versión original de Python, garantizando la consistencia de los resultados en los datasets **MOT17** y **MOT20**.

## 1. Preparación del Entorno

Para la correcta ejecución del proyecto, es necesario establecer la siguiente estructura de directorios y configurar los repositorios externos.

### A. Descarga de Datasets (MOT17 / MOT20)s
Los datasets oficiales deben obtenerse desde [MOTChallenge](https://motchallenge.net/). La extracción en la máquina host debe apuntar a las siguientes rutas predeterminadas:
* `/home/usuario/Downloads/MOT17/train`
* `/home/usuario/Downloads/MOT20/train`

El contenedor de Docker (`mot_container`) requiere un volumen montado para acceder a estos datos en la ruta interna `/app/datasets/`.

### B. Comparación con TrackTrack (Referencia en Python)
El proyecto está basado en el algoritmo original de Python. Para establecer la línea base de comparación, se requieren los siguientes scripts personalizados:
* `run_python_mot.py`: Genera las trayectorias base usando el tracker de Python (FastReID desactivado temporalmente).
* `run_eval.py`: Compara los resultados exportados por Rust y Python contra el Ground Truth oficial.

**Pasos de configuración:**
1. Clonar el repositorio original en el host:
   ```bash
   cd /home/usuario/
   git clone [https://github.com/kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) TrackTrack   
   ```
2. Añadir los archivos `run_python_mot.py` y `run_eval.py` generados a la carpeta local `/home/usuario/TrackTrack/3. Tracker/`.

### C. Dependencia de Evaluación (TrackEval)

El script de evaluación (`run_eval.py`) requiere el repositorio oficial de métricas de MOTChallenge.

1. Clonar el repositorio dentro del directorio del tracker de Python:
```bash
cd "/home/usuario/TrackTrack/3. Tracker"
git clone [https://github.com/JonathonLuiten/TrackEval.git](https://github.com/JonathonLuiten/TrackEval.git) trackeval
```
## 2. Prerrequisitos de Ejecución

El flujo de trabajo requiere operar con **dos terminales** de forma simultánea en la máquina host:

* **Terminal 1 (Docker):** Destinada a la compilación y ejecución del código en Rust dentro del entorno contenedorizado.
* **Terminal 2 (Host):** Destinada a la gestión de archivos locales y ejecución de los scripts de evaluación en Python.

## 3. Selección del Dataset

Previo a cualquier ejecución, ambos lenguajes (Rust y Python) deben estar configurados de forma síncrona para apuntar al mismo dataset mediante sus respectivos parámetros globales.

**Configuración en Rust:**
Editar el archivo `src/bin/eval_mot.rs` (Opción A) o `src/bin/eval_yolo.rs` (Opción B) y ajustar la constante inicial:
```rust
const CONFIG_DATASET: DatasetMode = DatasetMode::Mot17; // Alternativa: DatasetMode::Mot20
```

**Configuración en Python:**
Editar los archivos `run_python_mot.py` y `run_eval.py` y ajustar la variable global inicial:

```python
CONFIG_DATASET = "MOT17" # Alternativa: "MOT20"
```


## 4. Flujos de Ejecución y Evaluación

### Opción A — Evaluación del Algoritmo Puro (Detecciones Precalculadas)

Modalidad diseñada para validar que la lógica matemática y de asociación del tracker en Rust es idéntica a la implementación en Python. Utiliza las detecciones estáticas del dataset (`det.txt`), por lo que los resultados de ambas implementaciones deben ser estadísticamente equivalentes.

**Paso 1 — Ejecutar el tracker Rust (Terminal 1 - Docker):**

```bash
cargo run --release --bin eval_mot
```

**Paso 2 — Ejecutar el tracker Python (Terminal 2 - Host):**

Generación de las trayectorias base de referencia.

```bash
cd "/home/usuario/TrackTrack/3. Tracker" 
python3 run_python_mot.py 
```

**Paso 3 — Extracción de resultados al host (Terminal 2 - Host):**

Ejecución del comando de copia para exportar las trayectorias generadas por Rust desde el contenedor.

```bash
docker cp mot_container:/app/rust_results/. /home/usuario/TrackTrack/outputs/rust_results_v2/
```

**Paso 4 — Ejecutar la evaluación (Terminal 2 - Host):**

Procesamiento de métricas (HOTA/ClearMOT) para la comparativa.

```bash
cd "/home/usuario/TrackTrack/3. Tracker"
python3 run_eval.py
```

### Opción B — Evaluación del Sistema Completo (YOLOv8 en Tiempo Real)

Modalidad orientada a medir el rendimiento real en producción, realizando inferencia directa sobre la secuencia de frames (`.jpg`) del dataset.

**Paso 1 — Ejecutar el tracker Rust con YOLO (Terminal 1 - Docker):**

```bash
cargo run --release --bin eval_yolo
```

**Paso 2 — Extracción de resultados al host (Terminal 2 - Host):**

Extracción de los datos exportados por el pipeline completo.

```bash
docker cp mot_container:/app/TrackTrack/outputs/rust_results/data/. /home/usuario/TrackTrack/outputs/rust_results_v2/
```

**Paso 3 — Ejecutar la evaluación (Terminal 2 - Host):**

```bash
cd "/home/usuario/TrackTrack/3. Tracker"
python3 run_eval.py
```

> **NOTA:** En esta modalidad, el análisis debe centrarse en los resultados directos de Rust. La comparativa estricta con Python resulta asimétrica, dado que ambos sistemas generan sus propias detecciones de forma independiente.

## 5. Mapa de Rutas Críticas

| Componente | Ruta de Acceso |
| --- | --- |
| **Código Fuente Rust** | `/app/src/` y `/app/src/bin/` *(Contenedor)* |
| **Datasets (Rust)** | `/app/datasets/MOT17/` y `/app/datasets/MOT20/` *(Contenedor)* |
| **Resultados Rust (Opción A)** | `/app/rust_results/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Rust (Opción B)** | `/app/TrackTrack/outputs/rust_results/data/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Python** | `/home/usuario/TrackTrack/outputs/python_results/` *(Host)* |
| **Scripts de Python** | `/home/usuario/TrackTrack/3. Tracker/` *(Host)* |

## 6. Métricas de Referencia

Los siguientes valores corresponden al *baseline* oficial de **MOT17 (FRCNN Train)** y **MOT20 (Train)** del tracker original en Python. El criterio de éxito de la Opción A es alcanzar estas métricas con exactitud.

| Métrica  | Definición de la Métrica                                                     | MOT17     |   MOT20   |
| ---      | ---                                                                          | ---       | ---       |
| **HOTA** | Balance global entre precisión de detección y asociación temporal.           | **45.41** | **17.44** |
| **MOTA** | Calidad general del sistema (integra FP, FN e IDSW).                         | **47.15** | **7.38**  |
| **IDF1** | Capacidad del tracker para mantener una misma identidad en el tiempo.        | **51.71** | **12.23** |
| **DetA** | Rendimiento aislado de la arquitectura de detección.                         | **43.09** | **6.97**  |
| **AssA** | Rendimiento aislado del algoritmo de asociación y enlace.                    | **48.02** | **43.64** |
| **IDSW** | Suma total de cambios de identidad *(menor valor indica mejor rendimiento)*. | **559**   | **175**   |