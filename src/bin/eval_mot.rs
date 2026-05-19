// ============================================================================
// # Evaluador del Tracker en MOT17/MOT20
//
// Este binario automatiza la evaluación del rendimiento del algoritmo de seguimiento
// utilizando las detecciones precalculadas del dataset seleccionado (FRCNN / MOT20).
// Sirve para validar que la lógica matemática de asociación en Rust es equivalente 1:1 con Python.
// ============================================================================

use mot::tracktrack::track::{Detection, TrackState};
use mot::tracktrack::tracker::{Args, Tracker};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

// ============================================================================
// CONTROL DE DATASET (SELECTOR CENTRAL)
// ============================================================================

/// Enumeración con los datasets soportados
#[allow(dead_code)]
enum DatasetMode {
    Mot17,
    Mot20,
}

/// Cambia aquí entre 'DatasetMode::Mot17' o 'DatasetMode::Mot20'
const CONFIG_DATASET: DatasetMode = DatasetMode::Mot20;

// ============================================================================
// PARSERS DE CONFIGURACIÓN Y DETECCIONES
// ============================================================================

// Carga las detecciones de det.txt (Formato MOTChallenge: frame,-1,x,y,w,h,score,...)
fn load_detections(det_path: &str) -> HashMap<usize, Vec<Detection>> {
    let mut dets: HashMap<usize, Vec<Detection>> = HashMap::new();
    let file = std::fs::File::open(det_path).expect("No se pudo abrir det.txt");

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        // Parseamos la línea separada por comas a un vector de flotantes
        let parts: Vec<f32> = line
            .trim()
            .split(',')
            .map(|x| x.parse().unwrap_or(0.0))
            .collect();

        // Control de seguridad: saltar líneas incompletas o vacías
        if parts.len() < 7 {
            continue;
        }

        // Mapeo de índices según el estándar oficial MOT
        let frame = parts[0] as usize;
        let x = parts[2];
        let y = parts[3];
        let w = parts[4];
        let h = parts[5];
        let score = parts[6];

        // El motor interno trabaja con esquinas absolutas [x1, y1, x2, y2].
        // Pasamos de formato [x, y, w, h] -> [x1, y1, x2, y2] mediante sumas básicas.
        let det = Detection {
            bbox: [x as f64, y as f64, (x + w) as f64, (y + h) as f64],
            score: score as f64,
            feat: Vec::new(), // Opción A: No usamos embeddings visuales de ReID
        };

        // Agrupamos las detecciones indexadas por su correspondiente número de frame
        dets.entry(frame).or_default().push(det);
    }
    dets
}

// Lee seqinfo.ini para obtener los metadatos esenciales de la secuencia
fn read_seqinfo(seqinfo_path: &str) -> (usize, usize, usize) {
    let file = std::fs::File::open(seqinfo_path).expect("No se pudo abrir seqinfo.ini");
    let mut seq_len = 0usize;
    let mut frame_rate = 30usize;
    let mut img_w = 1920usize;

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        // Buscamos las variables clave dividiendo la cadena por el símbolo '='
        if line.contains("seqLength") {
            seq_len = line.split('=').nth(1).unwrap().trim().parse().unwrap_or(0);
        }
        if line.contains("frameRate") {
            frame_rate = line.split('=').nth(1).unwrap().trim().parse().unwrap_or(30);
        }
        if line.contains("imWidth") {
            img_w = line
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .parse()
                .unwrap_or(1920);
        }
    }
    (seq_len, frame_rate, img_w)
}

// ============================================================================
// EJECUCIÓN DE LA SECUENCIA
// ============================================================================

// Procesa una secuencia completa frame a frame y guarda los resultados en un .txt
fn run_sequence(seq_path: &str, seq_name: &str, output_dir: &str) -> (f64, usize) {
    let det_path = format!("{}/det/det.txt", seq_path);
    let seqinfo_path = format!("{}/seqinfo.ini", seq_path);

    let dets = load_detections(&det_path);
    let (seq_len, frame_rate, _) = read_seqinfo(&seqinfo_path);

    // Hiperparámetros del tracker: replicamos exactamente la configuración de Python para validar consistencia
    let args = Args {
        max_time_lost: frame_rate * 2, // Frames para recuperar un objeto antes de eliminarlo
        det_thr: 0.5,                  // Umbral de corte para detecciones de alta/baja confianza
        match_thr: 0.8,                // Límite de coste IoU para la asociación húngara
        penalty_p: 0.1,
        penalty_q: 0.2,
        reduce_step: 0.1,
        init_thr: 0.6,
        tai_thr: 0.4,
        min_len: 3, // Frames mínimos seguidos puntuando para confirmar un track nuevo
    };

    let mut tracker = Tracker::new(args, seq_name);
    let out_path = format!("{}/{}.txt", output_dir, seq_name);

    // Abrimos el archivo de logs limpiando (truncate) ejecuciones previas
    let mut out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&out_path)
        .unwrap();

    let mut total_time = 0.0f64;

    // Bucle temporal: recorremos la secuencia frame a frame de forma ordenada
    for frame_id in 1..=seq_len {
        let frame_dets = dets.get(&frame_id).cloned().unwrap_or_default();
        let empty: Vec<Detection> = Vec::new(); // Vector vacío auxiliar (obligado por firma sin ReID)

        let start = Instant::now();
        // Bifurcación lógica: si el frame tiene datos asociamos, si está vacío usar filtro de Kalman
        if !frame_dets.is_empty() {
            tracker.update(frame_dets, empty);
        } else {
            tracker.update_without_detections();
        }
        total_time += start.elapsed().as_secs_f64();

        // Guardamos las posiciones del frame actual en formato MOTChallenge
        for track in &tracker.tracks {
            if track.state == TrackState::Confirmed {
                let b = track.x1y1wh(); // Transformamos de vuelta al formato de salida [x, y, width, height]

                // Filtro oficial del Benchmark: descartamos cajas de ruido minúsculas (área <= 100 px)
                if b[2] * b[3] > 100.0 {
                    writeln!(
                        out_file,
                        "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},-1,-1,-1",
                        frame_id, track.track_id, b[0], b[1], b[2], b[3], track.score
                    )
                    .unwrap();
                }
            }
        }
    }
    (total_time, seq_len)
}

// ============================================================================
// FUNCIÓN PRINCIPAL / BENCHMARK
// ============================================================================

fn main() {
    // ASIGNACIÓN DINÁMICA: Configuramos las carpetas y filtros según el interruptor central
    let (data_dir, filter_keyword, dataset_name) = match CONFIG_DATASET {
        DatasetMode::Mot17 => (
            "/app/datasets/MOT17/train",
            "FRCNN", // Buscamos las carpetas base de Faster R-CNN
            "MOT17",
        ),
        DatasetMode::Mot20 => (
            "/app/datasets/MOT20/train",
            "MOT20", // Las carpetas de MOT20 no tienen sufijo de detector, se filtran por su propio nombre
            "MOT20",
        ),
    };

    let output_dir = "/app/rust_results";
    fs::create_dir_all(output_dir).unwrap();

    // Escaneo dinámico del directorio aplicando el filtro inteligente del dataset activo
    let mut sequences: Vec<String> = fs::read_dir(data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.contains(filter_keyword))
        .collect();
    sequences.sort(); // Orden alfabético estricto para garantizar determinismo

    let mut total_time = 0.0f64;
    let mut total_frames = 0usize;

    // Lanzamos el benchmark secuencial sobre todo el set mapeado
    for seq in &sequences {
        let seq_path = format!("{}/{}", data_dir, seq);
        println!("Procesando {}...", seq);

        let (t, n) = run_sequence(&seq_path, seq, output_dir);
        println!("  {}: {:.1} FPS", seq, n as f64 / t);

        total_time += t;
        total_frames += n;
    }

    // Reporte final adaptativo impreso en consola
    println!("\n=== RESULTADOS RUST {} ===", dataset_name);
    println!("Tiempo total tracker: {:.4} s", total_time);
    println!(
        "Velocidad media: {:.2} FPS",
        total_frames as f64 / total_time
    );
}