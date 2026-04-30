// Este ejemplo combina YOLOv8 para detección de objetos con Norfair para tracking,
// permitiendo seguimiento de personas en video con predicción de Kalman y re-identificación.
// Usa OpenCV para procesamiento de video e inferencia DNN.

use nalgebra::DMatrix;
use norfair_rs::distances::{DistanceFunction, ScalarDistance};
use norfair_rs::filter::FilterFactoryEnum;
use norfair_rs::filter::OptimizedKalmanFilterFactory;
use norfair_rs::{Detection, TrackedObject, Tracker, TrackerConfig};
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    videoio, Result,
};

// ── Constantes de configuración ──────────────────────────────────────────────
// Centralizar aquí todos los parámetros facilita ajustes sin tocar la lógica

/// Factor de escala al reducir el frame antes de la inferencia YOLO
/// 0.25 = 25% del tamaño original → 4× más rápido con pérdida mínima de precisión
const SCALE: f64 = 0.25;

/// Tamaño de entrada requerido por el modelo YOLO (cuadrado)
const INPUT_SIZE: i32 = 640;

/// Ejecutar detección solo cada N frames para ahorrar CPU
const DETECT_EVERY_N_FRAMES: usize = 2;

/// Confianza mínima para aceptar una detección como válida
const CONF_THRESHOLD: f32 = 0.5;

/// Umbral de IoU para Non-Maximum Suppression: suprime cajas que se solapen más de este valor
const NMS_THRESHOLD: f32 = 0.4;

/// Índice de clase COCO que corresponde a "persona"
/// Índices de clase COCO habituales:
///   0 = persona,  1 = bicicleta, 2 = coche,       3 = motocicleta,
///   5 = autobús,  7 = camión,   16 = perro,       17 = caballo
const PERSON_CLASS_ID: i32 = 0;

// ── Función de distancia normalizada por tamaño de caja ──────────────────────
// Divide la distancia euclidiana por la diagonal de la detección,
// así objetos pequeños y grandes tienen el mismo umbral relativo.
// El distance_threshold del Tracker debe ser ~0.5 (en vez de píxeles crudos).
fn bbox_distance(det: &Detection, track: &TrackedObject) -> f64 {
    // Extraer coordenadas de la detección actual
    let dx1 = det.points[(0, 0)] - track.estimate[(0, 0)];
    let dy1 = det.points[(0, 1)] - track.estimate[(0, 1)];
    let dx2 = det.points[(1, 0)] - track.estimate[(1, 0)];
    let dy2 = det.points[(1, 1)] - track.estimate[(1, 1)];

    // Calcular ancho y alto de la detección
    let det_w = (det.points[(1, 0)] - det.points[(0, 0)]).abs().max(1.0);
    let det_h = (det.points[(1, 1)] - det.points[(0, 1)]).abs().max(1.0);
    // Diagonal de la caja delimitadora (normalizador)
    let diag = (det_w * det_w + det_h * det_h).sqrt();

    // Distancia euclidiana entre esquinas superiores izquierdas
    let dist1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    // Distancia euclidiana entre esquinas inferiores derechas
    let dist2 = (dx2 * dx2 + dy2 * dy2).sqrt();

    // Promedio de distancias normalizado por el tamaño de la caja
    (dist1 + dist2) / (2.0 * diag)
}

// ── Inferencia YOLO: devuelve detecciones en coordenadas de frame_small ──────
fn run_yolo(frame_small: &Mat, net: &mut dnn::Net) -> Result<Vec<Detection>> {
    let blob = dnn::blob_from_image(
        frame_small,
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

    let x_factor = frame_small.cols() as f32 / INPUT_SIZE as f32;
    let y_factor = frame_small.rows() as f32 / INPUT_SIZE as f32;

    let data = output.data_typed::<f32>()?;

    let mut class_ids = Vector::<i32>::new(); // Ahora mismo innecesario porque solo filtramos "person", pero lo guardamos por si queremos usar otras clases
    let mut confidences = Vector::<f32>::new();
    let mut boxes = Vector::<Rect>::new();

    for p in 0..num_preds {
        let (cx, cy, w, h, confidence, class_id);

        if is_yolov8 {
            cx = data[p];
            cy = data[1 * num_preds + p];
            w = data[2 * num_preds + p];
            h = data[3 * num_preds + p];

            let mut max_score = 0.0f32;
            let mut best_cls = 0i32;
            for j in 0..num_classes {
                let score = data[(4 + j) * num_preds + p];
                if score > max_score {
                    max_score = score;
                    best_cls = j as i32;
                }
            }
            confidence = max_score;
            class_id = best_cls;
        } else {
            let base = p * num_attrs;
            cx = data[base];
            cy = data[base + 1];
            w = data[base + 2];
            h = data[base + 3];
            let objectness = data[base + 4];

            let mut max_score = 0.0f32;
            let mut best_cls = 0i32;
            for j in 0..num_classes {
                let score = data[base + 5 + j];
                if score > max_score {
                    max_score = score;
                    best_cls = j as i32;
                }
            }
            confidence = objectness * max_score;
            class_id = best_cls;
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

    // Coordenadas directamente en frame_small (sin escalar a full-res)
    let mut detections = Vec::with_capacity(indices.len());
    for idx in &indices {
        let r = boxes.get(idx as usize)?;
        let x1 = r.x as f64;
        let y1 = r.y as f64;
        let x2 = (r.x + r.width) as f64;
        let y2 = (r.y + r.height) as f64;
        let det = Detection::new(DMatrix::from_row_slice(2, 2, &[x1, y1, x2, y2]))
            .map_err(|e| opencv::Error::new(0, format!("Detection creation failed: {:?}", e)))?;
        detections.push(det);
    }

    Ok(detections)
}

// ── Dibuja un track: caja + ID + trayectoria ─────────────────────────────────
fn draw_track(frame: &mut Mat, track: &TrackedObject) -> Result<()> {
    let est = &track.estimate;
    let x1 = est[(0, 0)] as i32;
    let y1 = est[(0, 1)] as i32;
    let x2 = est[(1, 0)] as i32;
    let y2 = est[(1, 1)] as i32;

    let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let yellow = Scalar::new(0.0, 200.0, 255.0, 0.0);

    // Caja principal
    imgproc::rectangle(
        frame,
        Rect::new(x1, y1, x2 - x1, y2 - y1),
        green,
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Etiqueta con ID
    imgproc::put_text(
        frame,
        &format!("ID:{}", track.id.unwrap_or(0)),
        Point::new(x1, (y1 - 6).max(0)),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.45,
        green,
        1,
        imgproc::LINE_8,
        false,
    )?;

    // ── Trayectoria (past_detections) ────────────────────────────────────────
    // Usamos los puntos pasados que Norfair ha acumulado para dibujar
    // el rastro del objeto como una secuencia de círculos.
    for past in &track.past_detections {
        let cx = ((past.points[(0, 0)] + past.points[(1, 0)]) / 2.0) as i32;
        let cy = ((past.points[(0, 1)] + past.points[(1, 1)]) / 2.0) as i32;
        imgproc::circle(frame, Point::new(cx, cy), 2, yellow, -1, imgproc::LINE_8, 0)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    // 1. Abrir el video desde un archivo
    // "test.mp4" debe existir en el directorio actual
    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    // 2. Cargar el modelo ONNX de YOLOv8 y configurar la inferencia en CPU
    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    // 3. Configurar el tracker Norfair con función de distancia personalizada
    // Umbral 0.5 (adimensional) en lugar de 100 píxeles crudos
    // let mut tracker = Tracker::new(TrackerConfig {
    //     distance_function: DistanceFunction::Frobenius(ScalarDistance::new(bbox_distance)),
    //     distance_threshold: 0.3,
    //     hit_counter_max: 15,
    //     initialization_delay: 3,
    //     pointwise_hit_counter_max: 4,
    //     detection_threshold: 0.5,
    //     filter_factory: FilterFactoryEnum::default(),
    //     past_detections_length: 4,
    //     reid_distance_function: None,   // Se podría reidentificar con una funcion propia
    //     reid_distance_threshold: 100.0,
    //     reid_hit_counter_max: Some(50),
    //     filter_factory: Box::new(OptimizedKalmanFilterFactory::new(
    //             4.0,   // R (measurement noise)
    //             0.1,   // Q (process noise)
    //             10.0,  // P (initial covariance)
    //             0.0,   // pos_variance
    //             1.0,   // vel_variance
    //         ))
    // })
    // .map_err(|e| opencv::Error::new(0, format!("Tracker creation failed: {:?}", e)))?;
    let mut tracker = Tracker::new(TrackerConfig {
        distance_function: DistanceFunction::Frobenius(ScalarDistance::new(bbox_distance)),
        distance_threshold: 0.3,
        hit_counter_max: 30,
        initialization_delay: 3,
        pointwise_hit_counter_max: 4,
        detection_threshold: 0.0,
        filter_factory: FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::new(
            15.0, // R (Ruido de medida): > Alto = YOLO tiembla, no te fíes de la caja cruda, suaviza el movimiento
            0.05, // Q (Ruido de proceso): < Bajo = Asume movimiento fluido. Evita que la caja dé tirones bruscos
            10.0, // P (Incertidumbre inicial): Margen de flexibilidad para calcular la velocidad cuando nace un ID nuevo
            0.1,  // pos_variance: Tolerancia al error en la posición pura (x, y).
            0.01, // vel_variance: Tolerancia a la velocidad inicial. ¡Bajo! Evita que la caja salga disparada (drift) y se deforme.
        )),
        past_detections_length: 20,
        reid_distance_function: None,
        reid_distance_threshold: 1.0,
        reid_hit_counter_max: None,
    })
    .map_err(|e| opencv::Error::new(0, format!("Tracker creation failed: {:?}", e)))?;

    // 4. Configurar la ventana de visualización
    let window = "YOLO v8 + Norfair-rs";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_small = Mat::default();
    let mut frame_num: usize = 0;

    println!("Iniciando... Presiona 'q' o 'esc' para salir");

    // BUCLE PRINCIPAL DE PROCESAMIENTO
    // lectura de frame -> detección opcional -> tracking -> dibujo y visualización
    loop {
        // Leer el siguiente frame del video
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video. Frames procesados: {}", frame_num);
            break;
        }
        frame_num += 1;

        // Reducir el frame según SCALE para acelerar la inferencia YOLO
        imgproc::resize(
            &frame,
            &mut frame_small,
            Size::new(0, 0),
            SCALE,
            SCALE,
            imgproc::INTER_AREA,
        )?;

        // ── CLAVE: el tracker se llama SIEMPRE, cada frame ───────────────────
        // En frames sin detección se pasa vec vacío: Norfair avanza el filtro
        // de Kalman internamente y predice la posición, evitando cajas congeladas.
        let detections = if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            run_yolo(&frame_small, &mut net)?
        } else {
            vec![] // Sin detección → Norfair predice con Kalman
        };

        let last_tracks = tracker.update(detections, frame_num as i32, None);

        // ── Dibujar solo tracks confirmados (superaron initialization_delay) ─
        for track in &last_tracks {
            if track.is_initializing {
                continue;
            }
            draw_track(&mut frame_small, track)?;
        }

        highgui::imshow(window, &frame_small)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    Ok(())
}
