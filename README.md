# 🚀 Guía de Evaluación: Tracker Rust vs Python (MOT17 / MOT20)

Este documento contiene las instrucciones para configurar, ejecutar y evaluar el tracker desarrollado en Rust frente a la versión de Python, garantizando la consistencia de los resultados en los datasets **MOT17** y **MOT20**.

---

## 🛠️ 1. Preparación del Entorno

Antes de comenzar, asegúrate de tener la estructura de directorios y los repositorios externos correctamente configurados.

### A. Descarga de Datasets (MOT17 / MOT20)
1. Descarga los datasets oficiales desde [MOTChallenge](https://motchallenge.net/).
2. Descomprímelos en tu máquina host. Por defecto, los scripts apuntan a:
   * `/home/catec/Downloads/MOT17/train`
   * `/home/catec/Downloads/MOT20/train`
3. Asegúrate de que el contenedor de Docker (`mot_container`) tiene montado un volumen para leer estos datos en `/app/datasets/`.

### B. Comparación con TrackTrack
Este repositorio está basado en el algoritmo original de Python. Para compararlos, se han generado los siguientes archivos:
* `eval_mot.py`: *(Genera las trayectorias base usando el tracker de Python)*
* `run_eval.py`: *(Compara los resultados de Rust y Python contra el Ground Truth)*

1. Clona el repositorio original en tu host:
   ```bash
   cd /home/catec/
   git clone [https://github.com/kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) TrackTrack
   
```
2. Añade los archivos `eval_mot.py` y `run_eval.py` generados a la carpeta local `/home/catec/TrackTrack/3. Tracker/`.

### C. Dependencia de Evaluación (TrackEval)
El script de evaluación (`run_eval.py`) utiliza el repositorio oficial métrico de MOT.

1. Clona el repositorio dentro de la carpeta del tracker de Python:
   ```bash
   cd "/home/catec/TrackTrack/3. Tracker"
   git clone [https://github.com/JonathonLuiten/TrackEval.git](https://github.com/JonathonLuiten/TrackEval.git) trackeval
   
```

---

## 💻 2. Prerrequisitos de Ejecución

Para seguir esta guía necesitas **dos terminales** abiertas simultáneamente en tu máquina host:

* **Terminal 1 (Docker):** Para compilar y ejecutar el código de Rust dentro del entorno contenedorizado.
* **Terminal 2 (Host):** Para gestionar los archivos locales y lanzar los scripts de evaluación en Python.

---

## ⚙️ 3. Paso Previo: Seleccionar el Dataset

Tanto en Rust como en Python, hemos configurado un "interruptor central". Antes de lanzar cualquier ejecución, debes asegurarte de que ambos lenguajes apuntan al mismo dataset.

**En Rust (Terminal 1 - Docker):**
Abre `src/bin/eval_mot17.rs` (Opción A) o `src/main.rs` (Opción B) y ajusta la constante al inicio del archivo:
```rust
const CONFIG_DATASET: DatasetMode = DatasetMode::Mot17; // o DatasetMode::Mot20
```

**En Python (Terminal 2 - Host):**
Abre `eval_mot.py` y `run_eval.py` y ajusta la variable global al inicio:
```python
CONFIG_DATASET = "MOT17" # o "MOT20"
```

---

## 🚀 4. Opción A — Evaluar el algoritmo puro (con detecciones precalculadas)

> **Nota:** Esta opción valida que la lógica matemática y de asociación del tracker en Rust es idéntica a la de Python utilizando las mismas detecciones fijas del dataset (`det.txt`). En MOT17 los resultados deberían ser 100% equivalentes.

**Paso 1 — Ejecutar el tracker Rust (Terminal 1 - Docker):**  
Asegúrate de tener puesto el dataset correcto en Rust y ejecuta el binario:
```bash
cargo run --release --bin eval_mot17
```

**Paso 2 — Copiar resultados al host (Terminal 2 - Host):**  
Sin salir del contenedor de la Terminal 1, abre tu segunda terminal y ejecuta el comando estandarizado para extraer las trayectorias generadas:
```bash
docker cp mot_container:/app/rust_results/. /home/catec/TrackTrack/outputs/rust_results_v2/
```

**Paso 3 — Ejecutar la evaluación (Terminal 2 - Host):**  
Lanza el script de métricas (HOTA/ClearMOT) para comparar los resultados:
```bash
cd "/home/catec/TrackTrack/3. Tracker"
python3 run_eval.py
```

---

## 🎯 5. Opción B — Evaluar el sistema completo (con YOLOv8 en tiempo real)

> **Nota:** Sirve para medir el rendimiento real en producción haciendo inferencia directa sobre los frames (`.jpg`) del dataset.

**Paso 1 — Ejecutar el tracker Rust con YOLO (Terminal 1 - Docker):**  
Ejecuta el binario principal:
```bash
cargo run --release --bin mot
```

**Paso 2 — Copiar resultados al host (Terminal 2 - Host):**  
Extrae los datos generados por el pipeline completo:
```bash
docker cp mot_container:/app/TrackTrack/outputs/rust_results/data/. /home/catec/TrackTrack/outputs/rust_results_v2/
```

**Paso 3 — Evaluar (Terminal 2 - Host):**  
Lanza el script de evaluación:
```bash
cd "/home/catec/TrackTrack/3. Tracker"
python3 run_eval.py
```

---

## 🔄 6. Regenerar resultados de Python (Base)

Si es la primera vez que configuras el entorno, o si realizas modificaciones en el tracker original de Python, debes generar los resultados base para poder comparar. Ejecuta esto en la **Terminal 2 (Host)**:

```bash
cd "/home/catec/TrackTrack/3. Tracker" 
python3 eval_mot.py 
python3 run_eval.py
```

---

## 🔨 7. Recompilación en Rust

Si haces cambios en los archivos `.rs`, vuelve a compilar el proyecto desde la **Terminal 1 (Docker)** antes de lanzar cualquier evaluación para actualizar todos los binarios:

```bash
cargo build --release
```

---

## 📂 8. Rutas Importantes

| Qué | Ruta |
| --- | --- |
| **Código Fuente Rust** | `/app/src/` y `/app/src/bin/` *(Dentro del contenedor)* |
| **Datasets (Rust)** | `/app/datasets/MOT17/` y `/app/datasets/MOT20/` *(Contenedor)* |
| **Resultados Rust (Opción A)** | `/app/rust_results/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Rust (Opción B)** | `/app/TrackTrack/outputs/rust_results/data/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Python** | `/home/catec/TrackTrack/outputs/python_results/` *(Host)* |
| **Scripts de Python** | `/home/catec/TrackTrack/3. Tracker/` *(Host)* |

---

## 📊 9. Métricas de Referencia

> **Nota:** Los valores objetivo mostrados abajo corresponden al benchmark oficial de **MOT17 (FRCNN Train)** del tracker original en Python. El objetivo del port a Rust es alcanzar y calcar exactamente estos números en la Opción A.

| Métrica | Qué mide | Referencia Python (MOT17) |
| :--- | :--- | :---: |
| **HOTA** | Balance perfecto entre detección y asociación. | **45.41** |
| **MOTA** | Calidad global del sistema (TP, FP, IDSW). | **47.15** |
| **IDF1** | Capacidad para mantener la consistencia de identidades a lo largo del tiempo. | **51.71** |
| **DetA** | Calidad pura del rendimiento de detección. | **43.09** |
| **AssA** | Calidad pura de la asociación/enlace de cajas. | **48.02** |
| **IDSW** | Cantidad de cambios de identidad *(menor es mejor)*. | **559** |