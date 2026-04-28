// Ejemplo de MOT básico usando YOLO (Soporte Universal v8/v8) y ByteTrack.
use opencv::{
    Result,
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    videoio,
};
use trackforge::trackers::byte_track::ByteTrack;

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;

    // 1. Cargar el modelo ONNX
    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    // 2. Inicializamos Trackforge
    let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.5);
    let mut last_tracks = Vec::new();

    let window = "MOT con YOLO Universal + Trackforge";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_num: usize = 0;
    const DETECT_EVERY_N_FRAMES: usize = 3;

    println!("Iniciando inferencia nativa... Presiona 'q' para salir.");

    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,
            &mut frame_small,
            Size::new(0, 0),
            0.5,
            0.5,
            imgproc::INTER_AREA,
        )?;

        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            let blob = dnn::blob_from_image(
                &frame_small,
                1.0 / 255.0,
                Size::new(640, 640),
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
            
            // Detectar la arquitectura de forma dinámica
            // YOLOv8: típicamente [1, 25200, 85] (dim 1 > dim 2)
            // YOLOv8: típicamente [1, 84, 8400] (dim 1 < dim 2)
            let is_yolov8 = size[1] < size[2];
            let num_preds = if is_yolov8 { size[2] as usize } else { size[1] as usize };
            let num_attrs = if is_yolov8 { size[1] as usize } else { size[2] as usize };

            let mut class_ids = Vector::<i32>::new();
            let mut confidences = Vector::<f32>::new();
            let mut boxes = Vector::<Rect>::new();

            let x_factor = frame_small.cols() as f32 / 640.0;
            let y_factor = frame_small.rows() as f32 / 640.0;

            let data = output.data_typed::<f32>()?;

            // Iteramos sobre el número de predicciones (8400 o 25200)
            for p in 0..num_preds {
                let mut max_class_score = 0.0;
                let mut class_id = 0;
                let confidence: f32;
                let cx: f32; let cy: f32; let w: f32; let h: f32;

                if is_yolov8 {
                    // LÓGICA YOLOv8: [1, 84, 8400]
                    // Los atributos son filas, las predicciones son columnas.
                    // Para acceder al atributo A de la predicción P: data[A * num_preds + P]
                    cx = data[0 * num_preds + p];
                    cy = data[1 * num_preds + p];
                    w  = data[2 * num_preds + p];
                    h  = data[3 * num_preds + p];

                    for j in 0..80 { // 80 clases de COCO
                        let score = data[(4 + j) * num_preds + p];
                        if score > max_class_score {
                            max_class_score = score;
                            class_id = j as i32;
                        }
                    }
                    // YOLOv8 no tiene objectness, la confianza es directamente el score de la clase
                    confidence = max_class_score; 

                } else {
                    // LÓGICA YOLOv8 / YOLOv7: [1, 25200, 85]
                    // Las predicciones son filas, los atributos son columnas contiguas.
                    let base_idx = p * num_attrs;
                    cx = data[base_idx + 0];
                    cy = data[base_idx + 1];
                    w  = data[base_idx + 2];
                    h  = data[base_idx + 3];
                    let objectness = data[base_idx + 4];

                    for j in 0..80 {
                        let score = data[base_idx + 5 + j];
                        if score > max_class_score {
                            max_class_score = score;
                            class_id = j as i32;
                        }
                    }
                    confidence = objectness * max_class_score;
                }

                // Filtrar por confianza y asegurarnos de que es la clase 0 (Persona)
                if confidence >= 0.5 && class_id == 0 {
                    let left = ((cx - w / 2.0) * x_factor) as i32;
                    let top = ((cy - h / 2.0) * y_factor) as i32;
                    let width = (w * x_factor) as i32;
                    let height = (h * y_factor) as i32;

                    class_ids.push(class_id);
                    confidences.push(confidence);
                    boxes.push(Rect::new(left, top, width, height));
                }
            }

            // Aplicar NMS nativo
            let mut indices = Vector::<i32>::new();
            dnn::nms_boxes(&boxes, &confidences, 0.5, 0.4, &mut indices, 1.0, 0)?;

            let mut detections = Vec::new();
            for idx in indices {
                let i = idx as usize;
                let rect = boxes.get(i)?;
                let conf = confidences.get(i)?;
                
                let bbox = [
                    rect.x as f32 * 2.0,
                    rect.y as f32 * 2.0,
                    rect.width as f32 * 2.0,
                    rect.height as f32 * 2.0,
                ];
                detections.push((bbox, conf, 0));
            }

            last_tracks = tracker.update(detections);
        }

        // PINTAR
        for track in &last_tracks {
            let bbox = track.tlwh;
            let rect_draw = Rect::new(
                (bbox[0] / 2.0) as i32,
                (bbox[1] / 2.0) as i32,
                (bbox[2] / 2.0) as i32,
                (bbox[3] / 2.0) as i32,
            );

            imgproc::rectangle(&mut frame_small, rect_draw, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;

            let label = format!("Persona ID: {}", track.track_id);
            let pos = Point::new((bbox[0] / 2.0) as i32, (bbox[1] / 2.0 - 10.0) as i32);
            imgproc::put_text(&mut frame_small, &label, pos, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, false)?;
        }

        highgui::imshow(window, &frame_small)?;

        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }

        frame_num += 1;
    }

    Ok(())
}