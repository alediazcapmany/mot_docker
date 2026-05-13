// Este ejemplo carga un modelo YOLO ONNX, ejecuta inferencia con OpenCV DNN,
// filtra detecciones de personas y envía los resultados a nuestro NUEVO TRACKER
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    videoio, Result,
};

// 1. IMPORTAMOS NUESTRO TRACKER PROPIO
// CÁMBIALO POR ESTO EN MAIN.RS:
use mot::tracktrack::track::{Detection, TrackState};
use mot::tracktrack::tracker::{Args, Tracker};
use std::time::Instant;

// ── Constantes de configuración ──────────────────────────────────────────────
const SCALE: f64 = 0.25;
const INPUT_SIZE: i32 = 640;
const DETECT_EVERY_N_FRAMES: usize = 1;
const CONF_THRESHOLD: f32 = 0.3;
const NMS_THRESHOLD: f32 = 0.4;
const PERSON_CLASS_ID: i32 = 0;

fn main() -> Result<()> {
    let inv_scale = (1.0 / SCALE) as f32;

    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    // 2. CONFIGURAMOS NUESTRO TRACKER
    let args = Args {
        max_time_lost: 30, // frames máximos perdido
        det_thr: 0.5,      // Límite para considerar det alta/baja
        match_thr: 0.8,    // Límite de similitud para asociar
        penalty_p: 0.1,    // Penalización por confianza baja
        penalty_q: 0.2,    // Penalización por det eliminada
        reduce_step: 0.1,  // Reducción del umbral iterativo
        init_thr: 0.6,     // Umbral inicio NMS
        tai_thr: 0.4,      // Track Aware NMS thresh
    };

    // OJO: "test.mp4" buscará "./trackers/cmc/GMC-test.mp4.txt".
    // Si no lo usas, crea el archivo vacío para que no explote.
    let mut tracker = Tracker::new(args, "test.mp4");
    let mut last_tracks = Vec::new();

    let window = "YOLO v8 + TrackTrack Pro";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_num: usize = 0;

    println!("Iniciando... Presiona 'q' o 'esc' para salir");

    let mut total_tracker_time = 0.0;
    let mut tracker_frames = 0;

    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video, han habido {} frames.", frame_num);
            break;
        }

        frame_num += 1;

        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,
            &mut frame_small,
            Size::new(0, 0),
            SCALE,
            SCALE,
            imgproc::INTER_AREA,
        )?;

        // 3. SEPARACIÓN DE LÓGICA: FRAME DE DETECCIÓN vs FRAME DE PREDICCIÓN
        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            // --- BLOQUE YOLO ---
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
            let out_names = net.get_unconnected_out_layers_names()?;
            net.forward(&mut output_blobs, &out_names)?;

            let output = output_blobs.get(0)?;
            let size = output.mat_size();

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
            let num_classes = if is_yolov8 {
                num_attrs - 4
            } else {
                num_attrs - 5
            };

            let mut class_ids = Vector::<i32>::new();
            let mut confidences = Vector::<f32>::new();
            let mut boxes = Vector::<Rect>::new();

            let x_factor = frame_small.cols() as f32 / INPUT_SIZE as f32;
            let y_factor = frame_small.rows() as f32 / INPUT_SIZE as f32;
            let data = output.data_typed::<f32>()?;

            for p in 0..num_preds {
                let mut max_class_score = 0.0f32;
                let mut class_id = 0i32;
                let confidence: f32;
                let cx: f32;
                let cy: f32;
                let w: f32;
                let h: f32;

                if is_yolov8 {
                    cx = data[0 * num_preds + p];
                    cy = data[1 * num_preds + p];
                    w = data[2 * num_preds + p];
                    h = data[3 * num_preds + p];

                    for j in 0..num_classes {
                        let score = data[(4 + j) * num_preds + p];
                        if score > max_class_score {
                            max_class_score = score;
                            class_id = j as i32;
                        }
                    }
                    confidence = max_class_score;
                } else {
                    let base_idx = p * num_attrs;
                    cx = data[base_idx + 0];
                    cy = data[base_idx + 1];
                    w = data[base_idx + 2];
                    h = data[base_idx + 3];
                    let objectness = data[base_idx + 4];

                    for j in 0..num_classes {
                        let score = data[base_idx + 5 + j];
                        if score > max_class_score {
                            max_class_score = score;
                            class_id = j as i32;
                        }
                    }
                    confidence = objectness * max_class_score;
                }

                if confidence >= CONF_THRESHOLD && class_id == PERSON_CLASS_ID {
                    let left = ((cx - w / 2.0) * x_factor) as i32;
                    let top = ((cy - h / 2.0) * y_factor) as i32;
                    let width = (w * x_factor) as i32;
                    let height = (h * y_factor) as i32;

                    class_ids.push(class_id);
                    confidences.push(confidence);
                    boxes.push(Rect::new(left, top, width, height));
                }
            }

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

            // 4. CONVERTIMOS A NUESTRO STRUCT 'Detection' EN FORMATO x1, y1, x2, y2
            let mut detections = Vec::new();
            for idx in indices {
                let i = idx as usize;
                let rect = boxes.get(i)?;
                let conf = confidences.get(i)?;

                // Escalamos y convertimos a [x1, y1, x2, y2]
                let x1 = rect.x as f32 * inv_scale;
                let y1 = rect.y as f32 * inv_scale;
                let x2 = (rect.x + rect.width) as f32 * inv_scale;
                let y2 = (rect.y + rect.height) as f32 * inv_scale;

                detections.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    score: conf,
                    feat: Vec::new(), // Sin ReID de momento
                });
            }

            // Actualizamos nuestro tracker (pasamos lista vacía a dets_95 de momento)
            // last_tracks = tracker.update(detections, Vec::new());
            // --- EMPIEZA CRONÓMETRO RUST ---
            let start_time = Instant::now();
            tracker.update(detections, Vec::new());
            let elapsed = start_time.elapsed().as_secs_f64();
            last_tracks = tracker
                .tracks
                .iter()
                .filter(|t| t.state == TrackState::Confirmed || t.state == TrackState::New)
                .cloned()
                .collect();
            
            total_tracker_time += elapsed;
            tracker_frames += 1;
            // --- TERMINA CRONÓMETRO RUST ---
        } else {
            // --- EMPIEZA CRONÓMETRO RUST ---
            let start_time = Instant::now();
            // FRAME SIN YOLO: Dejamos que el Filtro de Kalman siga la inercia
            tracker.update_without_detections();

            // Extraemos los tracks activos para dibujarlos (ya actualizados por el Kalman)
            last_tracks = tracker
                .tracks
                .iter()
                .filter(|t| t.state == TrackState::Confirmed || t.state == TrackState::New) // <--- AÑADE ESTO AQUÍ TAMBIÉN                .cloned()
                .cloned()
                .collect();
            let elapsed = start_time.elapsed().as_secs_f64();
            total_tracker_time += elapsed;
            tracker_frames += 1;
            // --- TERMINA CRONÓMETRO RUST ---
        }

        // 5. DIBUJAR RESULTADOS
        for track in &last_tracks {
            // Usamos tu método x1y1wh() para convertir el [x1,y1,x2,y2] de vuelta a un formato compatible con OpenCV Rect
            let bbox = track.x1y1wh();

            let rect_draw = Rect::new(
                (bbox[0] / inv_scale) as i32,
                (bbox[1] / inv_scale) as i32,
                (bbox[2] / inv_scale) as i32,
                (bbox[3] / inv_scale) as i32,
            );

            imgproc::rectangle(
                &mut frame_small,
                rect_draw,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2, // Línea un poco más gruesa
                imgproc::LINE_8,
                0,
            )?;

            let label = format!("ID: {}", track.track_id);
            let pos = Point::new(
                (bbox[0] / inv_scale) as i32,
                (bbox[1] / inv_scale - 10.0) as i32,
            );
            imgproc::put_text(
                &mut frame_small,
                &label,
                pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        highgui::imshow(window, &frame_small)?;

        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    // IMPRIMIR EL RESULTADO FINAL FUERA DEL BUCLE
    println!("=== RESULTADOS RUST ===");
    println!(
        "Tiempo total en el tracker: {:.4} segundos",
        total_tracker_time
    );
    println!(
        "Velocidad media: {:.2} FPS",
        tracker_frames as f64 / total_tracker_time
    );

    Ok(())
}
