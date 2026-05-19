// ============================================================================
// Proyecto: Tracker de Alta Velocidad en Rust (Benchmark MOT17/MOT20)
// Descripción: Pipeline completo de tracking usando YOLOv8 (Detección),
// FastReID (Apariencia - Preparado) y Filtro de Kalman + Húngaro (Asociación)
// ============================================================================

use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
    videoio, Result,
};

// --- Dependencias del Proyecto ---
// use mot::fast_re_id::emb_computer::EmbeddingComputer;
use mot::tracktrack::track::{Detection, TrackState};
use mot::tracktrack::tracker::{Args, Tracker};

// --- Utilidades Estándar ---
use std::fs::OpenOptions;
use std::io::Write;
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
// HIPERPARÁMETROS GLOBALES
// ============================================================================

// Tamaño de entrada cuadrado que espera la red YOLOv8
const INPUT_SIZE: i32 = 640;
// Frecuencia de detección: YOLO procesará 1 de cada N frames
const DETECT_EVERY_N_FRAMES: usize = 1;
// Umbral mínimo de confianza para que YOLO considere válida una detección
const CONF_THRESHOLD: f32 = 0.3;
// Umbral de Non-Maximum Suppression para eliminar cajas duplicadas sobre un mismo objeto
const NMS_THRESHOLD: f32 = 0.4;

fn main() -> Result<()> {
    // ========================================================================
    // CONFIGURACIÓN DINÁMICA SEGÚN EL DATASET SELECCIONADO
    // ========================================================================

    // El match asigna automáticamente los valores correctos
    let (base_dir, sequences, scale): (&str, &[&str], f64) = match CONFIG_DATASET {
        DatasetMode::Mot17 => (
            "/app/datasets/MOT17/train",
            &[
                "MOT17-02-FRCNN",
                "MOT17-04-FRCNN",
                "MOT17-05-FRCNN",
                "MOT17-09-FRCNN",
                "MOT17-10-FRCNN",
                "MOT17-11-FRCNN",
                "MOT17-13-FRCNN",
            ],
            0.25, // MOT17 funciona genial y rápido al 25%
        ),
        DatasetMode::Mot20 => (
            "/app/datasets/MOT20/train",
            &["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"],
            0.50, // MOT20 es muy denso; necesita un 50% de escala mínimo para no perder objetivos
        ),
    };

    // Multiplicador para revertir el reescalado y devolver las cajas a sus dimensiones reales
    let inv_scale = (1.0 / scale) as f32;

    // ========================================================================
    // FASE 1: CONFIGURACIÓN DEL ENTORNO Y DATASET
    // ========================================================================

    // Rutas de salida unificadas
    let results_dir = "/app/TrackTrack/outputs/rust_results";
    let data_dir = format!("{}/data", results_dir);

    // Creamos la infraestructura de directorios para los resultados si no existe
    std::fs::create_dir_all(&data_dir).expect("No se pudo crear el directorio de resultados");

    // ========================================================================
    // FASE 2: INICIALIZACIÓN DE MODELOS DE IA Y TRACKING
    // ========================================================================

    println!("Cargando red neuronal YOLOv8...");
    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    // Configuración para ejecutar en CPU usando el backend optimizado de OpenCV
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    let mut total_tracker_time = 0.0f64;
    let mut tracker_frames = 0usize;

    for seq_name in sequences.iter() {
        println!("\n=======================================================");
        println!("Iniciando sistema. Cargando secuencia: {}", seq_name);

        // Patrón %06d para que OpenCV itere automáticamente sobre los nombres de imagen
        let sequence_path = format!("{}/{}/img1/%06d.jpg", base_dir, seq_name);
        let output_txt = format!("{}/{}.txt", data_dir, seq_name);

        // Inicializamos el lector de imágenes simulando un flujo de video continuo
        let mut cam = videoio::VideoCapture::from_file(&sequence_path, videoio::CAP_IMAGES)?;
        if !cam.is_opened()? {
            println!(
                "[AVISO] No se encontraron imágenes en: {}. Saltando...",
                sequence_path
            );
            continue;
        }

        // Creamos el archivo de texto de salida sobrescribiendo datos anteriores
        let mut result_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_txt)
            .unwrap_or_else(|_| panic!("No se pudo crear el archivo de salida: {}", output_txt));

        // Parámetros matemáticos del motor de asociación
        let args = Args {
            max_time_lost: 30, // Frames tolerados para mantener un objeto perdido en memoria
            det_thr: 0.5,
            match_thr: 0.8,
            penalty_p: 0.1,
            penalty_q: 0.2,
            reduce_step: 0.1,
            init_thr: 0.6,
            tai_thr: 0.4,
            min_len: 3,
        };
        let mut tracker = Tracker::new(args, seq_name);

        // ========================================================================
        // FASE 3: BUCLE PRINCIPAL DE PROCESAMIENTO
        // ========================================================================

        // Contador físico de archivos .jpg para establecer el límite del bucle
        let img_dir = format!("{}/{}/img1", base_dir, seq_name);
        let total_frames = std::fs::read_dir(&img_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "jpg"))
            .count();

        println!(
            "Secuencia {}: Encontrados {} frames. ¡Arrancamos!",
            seq_name, total_frames
        );

        let mut frame = Mat::default();

        for frame_num in 1..=total_frames {
            cam.read(&mut frame)?;

            // Reducimos las dimensiones del frame para acelerar drásticamente el procesado de YOLO
            let mut frame_small = Mat::default();
            imgproc::resize(
                &frame,
                &mut frame_small,
                Size::new(0, 0),
                scale,
                scale,
                imgproc::INTER_AREA,
            )?;

            // CONTROL DE INFERENCIA: Evaluamos si toca usar el detector o avanzar por predicción
            if frame_num % DETECT_EVERY_N_FRAMES == 0 || frame_num == 1 {
                // --- BLOQUE DE DETECCIÓN (YOLOv8) ---

                // Preparamos la imagen reescalada convirtiéndola en un blob tensorial de tipo FP32
                let blob = dnn::blob_from_image(
                    &frame_small,
                    1.0 / 255.0,
                    Size::new(INPUT_SIZE, INPUT_SIZE),
                    Scalar::default(),
                    true,
                    false,
                    core::CV_32F,
                )?;

                net.set_input(&blob, "", 1.0, Scalar::default())?;
                let mut output_blobs: Vector<Mat> = Vector::new();
                net.forward(&mut output_blobs, &net.get_unconnected_out_layers_names()?)?;

                let output = output_blobs.get(0)?;
                let size = output.mat_size();

                // Comprobamos la arquitectura (YOLOv8 transpone las dimensiones respecto a YOLOv5)
                let is_yolov8 = size[1] < size[2];
                let num_preds = if is_yolov8 {
                    size[2] as usize
                } else {
                    size[1] as usize
                };
                let num_attrs = if is_yolov8 {
                    size[1] as usize
                } else {
                    size[2] as usize
                };

                let x_factor = frame_small.cols() as f32 / INPUT_SIZE as f32;
                let y_factor = frame_small.rows() as f32 / INPUT_SIZE as f32;
                let data = output.data_typed::<f32>()?;

                let mut confidences = Vector::<f32>::new();
                let mut boxes = Vector::<Rect>::new();

                // Recorremos la matriz de predicciones filtrando por confianza
                for p in 0..num_preds {
                    let (cx, cy, w, h, confidence) = if is_yolov8 {
                        let cx = data[0 * num_preds + p];
                        let cy = data[1 * num_preds + p];
                        let w = data[2 * num_preds + p];
                        let h = data[3 * num_preds + p];
                        let conf = data[4 * num_preds + p]; // Clase 0 (Persona) en YOLOv8
                        (cx, cy, w, h, conf)
                    } else {
                        let b = p * num_attrs;
                        let conf = data[b + 4] * data[b + 5]; // Confianza de caja * Clase Persona en YOLOv5
                        (data[b], data[b + 1], data[b + 2], data[b + 3], conf)
                    };

                    let confidence = confidence as f32;
                    if confidence >= CONF_THRESHOLD {
                        // Pasamos de formato centro-ancho-alto [cx, cy, w, h] a esquinas [left, top, w, h] para OpenCV
                        boxes.push(Rect::new(
                            ((cx - w / 2.0) * x_factor) as i32,
                            ((cy - h / 2.0) * y_factor) as i32,
                            (w * x_factor) as i32,
                            (h * y_factor) as i32,
                        ));
                        confidences.push(confidence);
                    }
                }

                // Eliminamos las cajas solapadas de una misma persona mediante NMS
                let mut indices = Vector::<i32>::new();
                dnn::nms_boxes(
                    &boxes,
                    &confidences,
                    CONF_THRESHOLD,
                    NMS_THRESHOLD,
                    &mut indices,
                    1.0,
                    0,
                )?;

                // Empaquetamos las detecciones finales mapeándolas al tamaño original de la imagen
                let mut detections = Vec::new();
                for idx in indices {
                    let rect = boxes.get(idx as usize)?;
                    let conf = confidences.get(idx as usize)?;

                    // Aplicamos el escalado inverso y transformamos a f64 para el tracker
                    let bbox = [
                        (rect.x as f32 * inv_scale) as f64,
                        (rect.y as f32 * inv_scale) as f64,
                        ((rect.x + rect.width) as f32 * inv_scale) as f64,
                        ((rect.y + rect.height) as f32 * inv_scale) as f64,
                    ];

                    // Extracción de embeddings visuales -> con GPU
                    // let feat = embedder.compute_embedding(&frame, &bbox)?;

                    detections.push(Detection {
                        bbox,
                        score: conf as f64,
                        feat: Vec::new(),
                    });
                }

                // Cronometramos y actualizamos el motor asociando las nuevas detecciones de YOLO
                let start = Instant::now();
                tracker.update(detections, Vec::new());
                total_tracker_time += start.elapsed().as_secs_f64();
                tracker_frames += 1;

            } else {
                // --- BLOQUE PREDICTIVO (Solo Filtro de Kalman) ---

                // YOLO descansa en este frame. El Filtro de Kalman proyecta la trayectoria por inercia cinematográfica.
                let start = Instant::now();
                tracker.update_without_detections();
                total_tracker_time += start.elapsed().as_secs_f64();
                tracker_frames += 1;
            }

            // ========================================================================
            // FASE 4: EXPORTACIÓN DE DATOS (Formato MOTChallenge)
            // ========================================================================

            for track in &tracker.tracks {
                if track.state == TrackState::Confirmed {
                    let bbox = track.x1y1wh(); // Convertimos de vuelta al formato [x, y, width, height]

                    // Seguridad: evitamos que errores numéricos generen dimensiones negativas o nulas
                    let w = bbox[2].max(1.0);
                    let h = bbox[3].max(1.0);

                    // Formato oficial: El '1' quemado en el campo 7 indica la clase fija de Peatón (Pedestrian)
                    let line = format!(
                        "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},1,-1,-1\n",
                        frame_num, track.track_id, bbox[0], bbox[1], w, h, track.score
                    );
                    result_file.write_all(line.as_bytes()).unwrap();
                }
            }
        }
    }

    // ========================================================================
    // RESULTADOS FINALES Y BENCHMARKING
    // ========================================================================
    println!("\n=======================================================");
    println!("=== RESUMEN GLOBAL EN RUST ===");
    println!("Archivos generados en: {}", data_dir);
    println!(
        "Tiempo neto en cálculos de tracking: {:.4} segundos",
        total_tracker_time
    );

    if total_tracker_time > 0.0 {
        let fps = tracker_frames as f64 / total_tracker_time;
        println!("Velocidad media del motor puro: {:.2} FPS", fps);
    }

    Ok(())
}
