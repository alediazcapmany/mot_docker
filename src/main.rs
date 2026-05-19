// ============================================================================
// Tracker de Rust — Evaluación con detecciones FRCNN (det.txt)
// Equivalente 1:1 al eval_mot17.py de Python
// ============================================================================

use mot::tracktrack::track::{Detection, TrackState};
use mot::tracktrack::tracker::{Args, Tracker};

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

// ============================================================================
// LECTURA DE det.txt  (formato MOT: frame,-1,x,y,w,h,score,...)
// ============================================================================

fn load_detections(det_path: &str) -> HashMap<usize, Vec<Detection>> {
    let mut dets: HashMap<usize, Vec<Detection>> = HashMap::new();

    let file = std::fs::File::open(det_path)
        .unwrap_or_else(|_| panic!("No se pudo abrir: {}", det_path));

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 7 {
            continue;
        }

        let frame: usize = parts[0].trim().parse().unwrap();
        let x: f64      = parts[2].trim().parse().unwrap();
        let y: f64      = parts[3].trim().parse().unwrap();
        let w: f64      = parts[4].trim().parse().unwrap();
        let h: f64      = parts[5].trim().parse().unwrap();
        let score: f64  = parts[6].trim().parse().unwrap_or(1.0);

        // x1y1wh → x1y1x2y2
        let bbox = [x, y, x + w, y + h];

        dets.entry(frame).or_default().push(Detection {
            bbox,
            score,
            feat: Vec::new(),
        });
    }

    dets
}

// ============================================================================
// LECTURA DE seqinfo.ini
// ============================================================================

fn read_seq_info(seq_path: &str) -> (usize, usize, usize) {
    // Devuelve (seq_len, frame_rate, img_w, img_h) — usamos (seq_len, frame_rate)
    let ini_path = format!("{}/seqinfo.ini", seq_path);
    let file = std::fs::File::open(&ini_path)
        .unwrap_or_else(|_| panic!("No se pudo abrir: {}", ini_path));

    let mut seq_len    = 0usize;
    let mut frame_rate = 30usize;

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.starts_with("seqLength") {
            seq_len = line.split('=').nth(1).unwrap().trim().parse().unwrap();
        }
        if line.starts_with("frameRate") {
            frame_rate = line.split('=').nth(1).unwrap().trim().parse().unwrap();
        }
    }

    (seq_len, frame_rate, 0)
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let base_dir   = "/app/datasets/MOT17/train";
    let output_dir = "/app/TrackTrack/outputs/rust_results/data";

    std::fs::create_dir_all(output_dir)
        .expect("No se pudo crear el directorio de resultados");

    let sequences = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ];

    let mut total_tracker_time = 0.0f64;
    let mut total_frames       = 0usize;

    for seq_name in sequences.iter() {
        println!("\n=======================================================");
        println!("Procesando: {}", seq_name);

        let seq_path  = format!("{}/{}", base_dir, seq_name);
        let det_path  = format!("{}/det/det.txt", seq_path);
        let out_path  = format!("{}/{}.txt", output_dir, seq_name);

        // Leer detecciones y metadatos
        let dets = load_detections(&det_path);
        let (seq_len, frame_rate, _) = read_seq_info(&seq_path);

        // Configurar tracker — igual que Python eval_mot17.py
        let args = Args {
            max_time_lost: frame_rate * 2, // igual que Python: frameRate * 2
            det_thr:    0.5,
            match_thr:  0.8,
            penalty_p:  0.1,
            penalty_q:  0.2,
            reduce_step: 0.1,
            init_thr:   0.6,
            tai_thr:    0.4,
            min_len:    3,
        };
        let mut tracker = Tracker::new(args, seq_name);

        // Fichero de salida
        let mut result_file = OpenOptions::new()
            .create(true).write(true).truncate(true)
            .open(&out_path)
            .unwrap_or_else(|_| panic!("No se pudo crear: {}", out_path));

        let mut seq_time = 0.0f64;

        for frame_id in 1..=seq_len {
            let frame_dets = dets.get(&frame_id).cloned().unwrap_or_default();

            let start = Instant::now();
            let tracks = if frame_dets.is_empty() {
                tracker.update_without_detections()
            } else {
                tracker.update(frame_dets, Vec::new())
            };
            seq_time += start.elapsed().as_secs_f64();

            // Escribir tracks confirmados — formato MOTChallenge
            for track in &tracks {
                if track.state == TrackState::Confirmed {
                    let bbox = track.x1y1wh();
                    let w = bbox[2].max(1.0);
                    let h = bbox[3].max(1.0);

                    // Filtro de área mínima (igual que Python: min_box_area=100)
                    if w * h <= 100.0 {
                        continue;
                    }

                    let line = format!(
                        "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},-1,-1,-1\n",
                        frame_id, track.track_id,
                        bbox[0], bbox[1], w, h,
                        track.score
                    );
                    result_file.write_all(line.as_bytes()).unwrap();
                }
            }
        }

        let fps = seq_len as f64 / seq_time;
        println!("  {}: {} frames → {:.1} FPS", seq_name, seq_len, fps);
        total_tracker_time += seq_time;
        total_frames += seq_len;
    }

    println!("\n=======================================================");
    println!("=== RESUMEN GLOBAL RUST (FRCNN) ===");
    println!("Tiempo neto tracker: {:.4} s", total_tracker_time);
    println!("Velocidad media:     {:.2} FPS", total_frames as f64 / total_tracker_time);
}