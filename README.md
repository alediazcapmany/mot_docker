# Guía de Evaluación: Tracker Rust vs Python (MOT17 / MOT20)

Este documento contiene las instrucciones para configurar, ejecutar y evaluar el tracker desarrollado en Rust frente a la versión de Python, garantizando la consistencia de los resultados en los datasets **MOT17** y **MOT20**.



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
* `run_python_mot.py`: *(Genera las trayectorias base usando el tracker de Python)*
* `run_eval.py`: *(Compara los resultados de Rust y Python contra el Ground Truth)*

1. Clona el repositorio original en tu host:
   ```bash
   cd /home/catec/
   git clone [https://github.com/kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) TrackTrack   
   ```
2. Añade los archivos `run_python_mot.py` y `run_eval.py` generados a la carpeta local `/home/catec/TrackTrack/3. Tracker/`.

### C. Dependencia de Evaluación (TrackEval)
El script de evaluación (`run_eval.py`) utiliza el repositorio oficial métrico de MOT. 

1. Clona el repositorio dentro de la carpeta del tracker de Python:
   ```bash
   cd "/home/catec/TrackTrack/3. Tracker"
   git clone [https://github.com/JonathonLuiten/TrackEval.git](https://github.com/JonathonLuiten/TrackEval.git) trackeval
   ```

## 💻 2. Prerrequisitos de Ejecución

Para seguir esta guía necesitas **dos terminales** abiertas simultáneamente en tu máquina host:

* **Terminal 1 (Docker):** Para compilar y ejecutar el código de Rust dentro del entorno contenedorizado.
* **Terminal 2 (Host):** Para gestionar los archivos locales y lanzar los scripts de evaluación en Python.


## ⚙️ 3. Paso Previo: Seleccionar el Dataset

Tanto en Rust como en Python, hemos configurado un "interruptor central". Antes de lanzar cualquier ejecución, debes asegurarte de que ambos lenguajes apuntan al mismo dataset.

**En Rust:**
Abre `src/bin/eval_mot.rs` (Opción A) o `src/bin/eval_yolo.rs` (Opción B) y ajusta la constante al inicio del archivo:
```rust
const CONFIG_DATASET: DatasetMode = DatasetMode::Mot17; // o DatasetMode::Mot20
```

**En Python:**
Abre `run_python_mot.py` y `run_eval.py` y ajusta la variable global al inicio:
```python
CONFIG_DATASET = "MOT17" # o "MOT20"
```

## 🚀 4. Ejecutar y Evaluar MOTChallenge

### Opción A — Evaluar el algoritmo puro (con detecciones precalculadas)

Esta opción valida que la lógica matemática y de asociación del tracker en Rust es idéntica a la de Python utilizando las mismas detecciones fijas del dataset (`det.txt`). Los resultados de ambas implementaciones deberían ser 100% equivalentes.

**Paso 1 — Ejecutar el tracker Rust (Terminal 1 - Docker):**  
Asegúrate de tener puesto el dataset correcto en Rust y ejecuta el binario:
```bash
cargo run --release --bin eval_mot
```

**Paso 2 — Ejecutar el tracker Python (Terminal 2 - Host):**  
Genera las trayectorias base de Python para tener la referencia:
```bash
cd "/home/catec/TrackTrack/3. Tracker" 
python3 run_python_mot.py 
```

**Paso 3 — Copiar resultados al host (Terminal 2 - Host):**  
Sin salir del contenedor de la Terminal 1, abre tu segunda terminal y ejecuta el comando estandarizado para extraer las trayectorias generadas:
```bash
docker cp mot_container:/app/rust_results/. /home/catec/TrackTrack/outputs/rust_results_v2/
```

**Paso 4 — Ejecutar la evaluación (Terminal 2 - Host):**  
Lanza el script de métricas (HOTA/ClearMOT) para comparar los resultados de ambos lenguajes:
```bash
cd "/home/catec/TrackTrack/3. Tracker"
python3 run_eval.py
```


### Opción B — Evaluar el sistema completo (con YOLOv8 en tiempo real)

Sirve para medir el rendimiento real en producción haciendo inferencia directa sobre los frames (`.jpg`) del dataset.

**Paso 1 — Ejecutar el tracker Rust con YOLO (Terminal 1 - Docker):**  
Ejecuta el binario principal:
```bash
cargo run --release --bin eval_yolo
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

## 🔨 5. Tras cambiar código Rust — Recompilar primero

Si haces cambios en los archivos `.rs`, vuelve a compilar el proyecto desde la **Terminal 1 (Docker)** antes de lanzar cualquier evaluación para actualizar todos los binarios:

```bash
cargo build --release
```

---

## 📂 6. Rutas Importantes

| Qué | Ruta |
| --- | --- |
| **Código Fuente Rust** | `/app/src/` y `/app/src/bin/` *(Dentro del contenedor)* |
| **Datasets (Rust)** | `/app/datasets/MOT17/` y `/app/datasets/MOT20/` *(Contenedor)* |
| **Resultados Rust (Opción A)** | `/app/rust_results/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Rust (Opción B)** | `/app/TrackTrack/outputs/rust_results/data/` *(Contenedor)* → `outputs/rust_results_v2/` *(Host)* |
| **Resultados Python** | `/home/catec/TrackTrack/outputs/python_results/` *(Host)* |
| **Scripts de Python** | `/home/catec/TrackTrack/3. Tracker/` *(Host)* |

---

## 📊 7. Métricas de Referencia

Los valores objetivo mostrados abajo corresponden a los benchmarks oficiales de **MOT17 (FRCNN Train)** y **MOT20 (Train)** del tracker original en Python. El objetivo del port a Rust es alcanzar y calcar exactamente estos números en la Opción A.

| Métrica | Qué mide | Ref. Python (MOT17) | Ref. Python (MOT20) |
| :--- | :--- | :---: | :---: |
| **HOTA** | Balance perfecto entre detección y asociación. | **45.41** | **17.44** |
| **MOTA** | Calidad global del sistema (TP, FP, IDSW). | **47.15** | **7.38** |
| **IDF1** | Capacidad para mantener la consistencia de identidades a lo largo del tiempo. | **51.71** | **12.23** |
| **DetA** | Calidad pura del rendimiento de detección. | **43.09** | **6.97** |
| **AssA** | Calidad pura de la asociación/enlace de cajas. | **48.02** | **43.64** |
| **IDSW** | Cantidad de cambios de identidad *(menor es mejor)*. | **559** | **175** |