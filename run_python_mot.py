# ============================================================================
# runl_mot.py — en la carpeta "3. Tracker"
# Automatiza la evaluación del tracker Python para MOT17 y MOT20, cprriendo el algoritmo
# ============================================================================

import os
import numpy as np
import time
from trackers.tracker import Tracker

# ============================================================================
# CONTROL DE DATASET (SELECTOR CENTRAL)
# ============================================================================
# Cambia aquí entre "MOT17" o "MOT20"
CONFIG_DATASET = "MOT20"

class Args:
    def __init__(self):
        self.max_time_lost = 30
        self.det_thr       = 0.5
        self.match_thr     = 0.8
        self.penalty_p     = 0.1
        self.penalty_q     = 0.2
        self.reduce_step   = 0.1
        self.init_thr      = 0.6
        self.tai_thr       = 0.4
        self.data_path     = ""
        self.min_len       = 3
        self.min_box_area  = 100
        self.img_w         = 1920
        self.img_h         = 1080

def load_detections(det_path):
    """Lee det.txt y devuelve dict {frame_id: np.array([x1,y1,x2,y2,score])}"""
    dets = {}
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            score = float(parts[6])
            
            # Convertir formato de caja x1y1wh → x1y1x2y2
            det = np.array([x, y, x+w, y+h, score], dtype=np.float32)
            if frame not in dets:
                dets[frame] = []
            dets[frame].append(det)
            
    for frame in dets:
        dets[frame] = np.array(dets[frame], dtype=np.float32)
    return dets

def write_results(path, results):
    """Guarda los resultados formateados según el estándar oficial MOT"""
    with open(path, 'w') as f:
        for frame_id, track_ids, x1y1whs, scores in results:
            for tid, box, score in zip(track_ids, x1y1whs, scores):
                f.write(f"{frame_id},{tid},{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f},{score:.2f},-1,-1,-1\n")

def run_sequence(seq_path, seq_name, output_dir):
    """Procesa una secuencia completa leyendo detecciones locales fijas"""
    det_path = os.path.join(seq_path, 'det', 'det.txt')
    dets = load_detections(det_path)

    args = Args()
    args.data_path = seq_path
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
    
    # Mapeo de metadatos desde el archivo ini
    with open(seqinfo_path) as f:
        for line in f:
            if 'frameRate' in line:
                args.max_time_lost = int(line.split('=')[-1].strip()) * 2
            if 'imWidth' in line:
                args.img_w = int(line.split('=')[-1].strip())
            if 'imHeight' in line:
                args.img_h = int(line.split('=')[-1].strip())
            if 'seqLength' in line:
                seq_len = int(line.split('=')[-1].strip())

    tracker = Tracker(args, seq_name)
    results = []
    total_time = 0.0

    for frame_id in range(1, seq_len + 1):
        det_array = dets.get(frame_id, None)
        empty = np.empty((0, 5), dtype=np.float32)

        start = time.perf_counter()
        if det_array is not None:
            tracks = tracker.update(det_array, empty)
        else:
            tracks = tracker.update_without_detections()
        total_time += time.perf_counter() - start

        x1y1whs, track_ids, scores = [], [], []
        for t in tracks:
            # Filtro oficial de área mínima
            if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                x1y1whs.append(t.x1y1wh)
                track_ids.append(t.track_id)
                scores.append(t.score)
        results.append([frame_id, track_ids, x1y1whs, scores])

    out_path = os.path.join(output_dir, f"{seq_name}.txt")
    write_results(out_path, results)
    return total_time, seq_len

# ============================================================================
# CONFIGURACIÓN DINÁMICA DE EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    
    # Diccionario de mapeo automático para evitar tocar el código inferior
    dataset_configs = {
        "MOT17": {
            "data_dir": "/home/catec/Downloads/MOT17/train",
            "filter": "FRCNN"
        },
        "MOT20": {
            "data_dir": "/home/catec/Downloads/MOT20/train",
            "filter": "MOT20"
        }
    }

    # Extraemos la configuración seleccionada arriba
    current_config = dataset_configs.get(CONFIG_DATASET)
    if not current_config:
        raise ValueError(f"Dataset no soportado: {CONFIG_DATASET}")

    data_dir = current_config["data_dir"]
    filter_keyword = current_config["filter"]
    output_dir = "/home/catec/TrackTrack/outputs/python_results"
    
    os.makedirs(output_dir, exist_ok=True)

    # Filtrado inteligente de carpetas según el dataset activo (y aseguramos que sean carpetas reales)
    sequences = [
        s for s in os.listdir(data_dir) 
        if filter_keyword in s and os.path.isdir(os.path.join(data_dir, s)) # Filtrado de seguridad 
    ]
    sequences.sort()

    total_time, total_frames = 0.0, 0
    for seq in sequences:
        seq_path = os.path.join(data_dir, seq)
        print(f"Procesando {seq}...")
        t, n = run_sequence(seq_path, seq, output_dir)
        total_time += t
        total_frames += n
        print(f"  {seq}: {n/t:.1f} FPS")

    # Reporte adaptativo final
    print(f"\n=== RESULTADOS PYTHON {CONFIG_DATASET} ===")
    print(f"Tiempo total tracker: {total_time:.4f} s")
    print(f"Velocidad media: {total_frames/total_time:.2f} FPS")