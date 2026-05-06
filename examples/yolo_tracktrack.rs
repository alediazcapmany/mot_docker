use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    video::KalmanFilter,
    videoio, Result,
};

// HUECO 1: Aquí irán los imports de ort (para FastReID), ndarray (para similitud coseno)
// y nuestras futuras estructuras del tracker (Track, KalmanFilter, etc.)
// Estado del track para benchmark
// enum TrackState {
//     Confirmed,
//     Tentative,
//     Lost,
// }

const SCALE: f64 = 0.25;
const INPUT_SIZE: i32 = 640;
const DETECT_EVERY_N_FRAMES: usize = 5;
const CONF_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;
const PERSON_CLASS_ID: i32 = 0;

fn main() -> Result<()> {
    let inv_scale = (1.0 / SCALE) as f32;

    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    // Cargar YOLOv8
    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    // ------------------------------------------------------------------------
    // HUECO 2: Cargar el modelo FastReID (ONNX)
    // Aquí inicializaremos la sesión de `ort` para extraer las características
    // de las personas recortadas.
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // HUECO 3: Inicializar nuestro gestor TrackTrack
    // let mut tracker = TrackTrack::new();
    // ------------------------------------------------------------------------

    let window = "YOLO v8 + TrackTrack + FastReID";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_num: usize = 0;

    // Almacenamos los tracks activos para dibujarlos en frames donde no hay detección
    // (Necesitaremos definir bien qué es un 'Track' luego)
    // let mut active_tracks = Vec::new();

    println!("Iniciando... Presiona 'q' o 'esc' para salir");

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

        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
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

            // Convertir las detecciones válidas a Rects en el frame original
            let mut current_detections = Vec::new();
            for idx in indices {
                let i = idx as usize;
                let rect = boxes.get(i)?;

                let bbox = Rect::new(
                    (rect.x as f32 * inv_scale) as i32,
                    (rect.y as f32 * inv_scale) as i32,
                    (rect.width as f32 * inv_scale) as i32,
                    (rect.height as f32 * inv_scale) as i32,
                );
                current_detections.push(bbox);
            }

            // ------------------------------------------------------------------------
            // HUECO 4: LÓGICA TRACKTRACK (TPA)
            // Aquí es donde ocurre la magia. El flujo que programaremos será:
            //
            // 1. Predecir tracks (Kalman):

            // 2. Contruir conjuntos de candidatos para cada track:
            //    - detecciones de alta confianza
            //    - detecciones de baja confianza

            // 3. FOR para cada track:
            //    a. filtrar candidatos (gating: IoU / Mahalanobis)
            //    b. calcular apariencia SOLO para candidatos
            //    c. calcular costo = α IoU + β similitud coseno
            //    d. asignar mejor match

            // 4. Mark:
            //    - matched → actualizar track (Posicion + apariencia)
            //    - unmatched tracks → perdidos
            //    - unmatched detections → nuevo track candidato

            // 5. TAI:
            //    - no new track if overlaps existing track
            //    - no nuevo track si se solapa con un track activo

            // 6. Prune lost tracks
            // active_tracks = tracker.update(current_detections, &frame, &mut fastreid_session);
            // ------------------------------------------------------------------------
        }

        for track in &active_tracks {
            // ------------------------------------------------------------------------
            // HUECO 5: DIBUJAR LOS RESULTADOS
            // Igual que antes, pero iterando sobre nuestra propia estructura `active_tracks`.
            // for track in &active_tracks {
            //     // ... dibujar rect_draw y put_text usando track.bbox y track.id
            // }
            // ------------------------------------------------------------------------
            let bbox = track.tlwh;
            // Convertir las coordenadas del track a formato Rect para dibujar en OpenCV
            let rect_draw = Rect::new(
                (bbox[0] / inv_scale) as i32,
                (bbox[1] / inv_scale) as i32,
                (bbox[2] / inv_scale) as i32,
                (bbox[3] / inv_scale) as i32,
            );

            // Dibujar la caja verde alrededor de la persona
            imgproc::rectangle(
                &mut frame_small,
                rect_draw,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            )?;

            // Dibujar el ID del track sobre la caja
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
                0.5,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
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

    Ok(())
}
