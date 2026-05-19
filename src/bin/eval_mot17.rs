use mot::tracktrack::track::{Detection, TrackState};
use mot::tracktrack::tracker::{Args, Tracker};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

fn load_detections(det_path: &str) -> HashMap<usize, Vec<Detection>> {
    let mut dets: HashMap<usize, Vec<Detection>> = HashMap::new();
    let file = std::fs::File::open(det_path).expect("No se pudo abrir det.txt");
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let parts: Vec<f32> = line
            .trim()
            .split(',')
            .map(|x| x.parse().unwrap_or(0.0))
            .collect();
        if parts.len() < 7 {
            continue;
        }
        let frame = parts[0] as usize;
        let x = parts[2];
        let y = parts[3];
        let w = parts[4];
        let h = parts[5];
        let score = parts[6];
        let det = Detection {
            bbox: [x as f64, y as f64, (x + w) as f64, (y + h) as f64],
            score: score as f64,
            feat: Vec::new(),
        };
        dets.entry(frame).or_default().push(det);
    }
    dets
}

fn read_seqinfo(seqinfo_path: &str) -> (usize, usize, usize) {
    let file = std::fs::File::open(seqinfo_path).expect("No se pudo abrir seqinfo.ini");
    let mut seq_len = 0usize;
    let mut frame_rate = 30usize;
    let mut img_w = 1920usize;
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
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

fn run_sequence(seq_path: &str, seq_name: &str, output_dir: &str) -> (f64, usize) {
    let det_path = format!("{}/det/det.txt", seq_path);
    let seqinfo_path = format!("{}/seqinfo.ini", seq_path);
    let dets = load_detections(&det_path);
    let (seq_len, frame_rate, _) = read_seqinfo(&seqinfo_path);

    let args = Args {
        max_time_lost: frame_rate * 2,
        det_thr: 0.5,
        match_thr: 0.8,
        penalty_p: 0.1,
        penalty_q: 0.2,
        reduce_step: 0.1,
        init_thr: 0.6,
        tai_thr: 0.4,
        min_len: 3, // ← añadir
    };

    let mut tracker = Tracker::new(args, seq_name);
    let out_path = format!("{}/{}.txt", output_dir, seq_name);
    let mut out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&out_path)
        .unwrap();

    let mut total_time = 0.0f64;

    for frame_id in 1..=seq_len {
        let frame_dets = dets.get(&frame_id).cloned().unwrap_or_default();
        let empty: Vec<Detection> = Vec::new();

        let start = Instant::now();
        if !frame_dets.is_empty() {
            tracker.update(frame_dets, empty);
        } else {
            tracker.update_without_detections();
        }
        total_time += start.elapsed().as_secs_f64();

        for track in &tracker.tracks {
            if track.state == TrackState::Confirmed {
                let b = track.x1y1wh();
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

fn main() {
    let data_dir = "/app/datasets/MOT17/train";
    let output_dir = "/app/rust_results";
    fs::create_dir_all(output_dir).unwrap();

    let mut sequences: Vec<String> = fs::read_dir(data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.contains("FRCNN"))
        .collect();
    sequences.sort();

    let mut total_time = 0.0f64;
    let mut total_frames = 0usize;

    for seq in &sequences {
        let seq_path = format!("{}/{}", data_dir, seq);
        println!("Procesando {}...", seq);
        let (t, n) = run_sequence(&seq_path, seq, output_dir);
        println!("  {}: {:.1} FPS", seq, n as f64 / t);
        total_time += t;
        total_frames += n;
    }

    println!("\n=== RESULTADOS RUST MOT17 ===");
    println!("Tiempo total tracker: {:.4} s", total_time);
    println!(
        "Velocidad media: {:.2} FPS",
        total_frames as f64 / total_time
    );
}
