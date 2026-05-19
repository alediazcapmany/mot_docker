# ============================================================================
# run_eval.py — en la carpeta "3. Tracker"
# Evaluador oficial usando TrackEval para MOT17 y MOT20
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import trackeval

# ============================================================================
# CONTROL DE DATASET (SELECTOR CENTRAL)
# ============================================================================
# Cambia aquí entre "MOT17" o "MOT20"
CONFIG_DATASET = "MOT20"

def create_seqmap(benchmark):
    """
    Genera el archivo seqmap físico dinámicamente usando rutas ABSOLUTAS
    para que TrackEval sepa exactamente dónde encontrarlo.
    """
    # Forzamos la ruta absoluta desde el directorio de este script
    base_dir = os.path.abspath(os.path.dirname(__file__))
    seqmap_folder = os.path.join(base_dir, "seqmaps_auto")
    os.makedirs(seqmap_folder, exist_ok=True)
    
    seqmap_filename = f"{benchmark}-train.txt"
    seqmap_file = os.path.join(seqmap_folder, seqmap_filename) # Ruta completa
    
    if benchmark == "MOT20":
        seqs = ["name", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
    elif benchmark == "MOT17":
        seqs = [
            "name", "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", 
            "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"
        ]
    else:
        seqs = ["name"]
        
    with open(seqmap_file, "w") as f:
        f.write("\n".join(seqs) + "\n")
        
    return seqmap_folder, seqmap_file # Devolvemos la ruta COMPLETA, no solo el nombre


def evaluate(tracker_name, results_dir, data_dir, benchmark, tracker_sub_folder=""):
    
    # 1. Creamos el archivo y obtenemos la RUTA COMPLETA
    seqmap_folder, seqmap_file_path = create_seqmap(benchmark)
    
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config['PLOT_CURVES'] = False

    # Configuración dinámica del dataset basada en los argumentos pasados
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['BENCHMARK']        = benchmark   
    dataset_config['GT_FOLDER']        = data_dir    
    dataset_config['TRACKERS_FOLDER']  = os.path.dirname(results_dir)
    dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
    dataset_config['SPLIT_TO_EVAL']    = 'train'
    
    # 2. Le pasamos a TrackEval la ruta absoluta del archivo generado
    dataset_config['SEQMAP_FOLDER']    = seqmap_folder
    dataset_config['SEQMAP_FILE']      = seqmap_file_path
    
    dataset_config['SKIP_SPLIT_FOL']   = True
    dataset_config['INPUT_AS_ZIP']     = False
    dataset_config['PRINT_CONFIG']     = False
    dataset_config['TRACKER_SUB_FOLDER'] = tracker_sub_folder

    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

    evaluator   = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]
    evaluator.evaluate(dataset_list, metrics_list)

# ============================================================================
# CONFIGURACIÓN DINÁMICA DE RUTAS
# ============================================================================
if __name__ == "__main__":
    
    # Diccionario de mapeo de Ground Truth según el interruptor superior
    dataset_configs = {
        "MOT17": {
            "data_dir": "/home/catec/Downloads/MOT17/train"
        },
        "MOT20": {
            "data_dir": "/home/catec/Downloads/MOT20/train"
        }
    }

    current_config = dataset_configs.get(CONFIG_DATASET)
    if not current_config:
        raise ValueError(f"Dataset no soportado: {CONFIG_DATASET}")

    data_dir = current_config["data_dir"]

    print(f"=== EVALUANDO PYTHON ({CONFIG_DATASET}) ===")
    evaluate(
        tracker_name  = "python_results",
        results_dir   = "/home/catec/TrackTrack/outputs/python_results",
        data_dir      = data_dir,
        benchmark     = CONFIG_DATASET,
    )

    print(f"\n=== EVALUANDO RUST ({CONFIG_DATASET}) ===")
    evaluate(
        tracker_name  = "rust_results_v2",
        results_dir   = "/home/catec/TrackTrack/outputs/rust_results_v2",
        data_dir      = data_dir,
        benchmark     = CONFIG_DATASET,
    )