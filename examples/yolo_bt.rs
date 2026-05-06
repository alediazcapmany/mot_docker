// Este ejemplo carga un modelo YOLO ONNX, ejecuta inferencia con OpenCV DNN,
// filtra detecciones de personas y envía los resultados al tracker ByteTrack
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    videoio, Result,
};
use trackforge::trackers::byte_track::ByteTrack;

// ── Constantes de configuración ──────────────────────────────────────────────
// Centralizar aquí todos los parámetros facilita ajustes sin tocar la lógica

/// Factor de escala al reducir el frame antes de la inferencia
/// 0.25 = 25% del tamaño original → 4× más rápido con pérdida mínima de precisión
const SCALE: f64 = 0.25;

/// Tamaño de entrada requerido por el modelo YOLO (cuadrado)
const INPUT_SIZE: i32 = 640;

/// Ejecutar detección solo cada N frames para ahorrar CPU
const DETECT_EVERY_N_FRAMES: usize = 5;

/// Confianza mínima para aceptar una detección como válida
const CONF_THRESHOLD: f32 = 0.5;

/// Umbral de IoU para Non-Maximum Suppression: suprime cajas que se solapen más de este valor
const NMS_THRESHOLD: f32 = 0.4;

/// Índice de clase COCO que corresponde a "persona"
/// Índices de clase COCO habituales:
///   0 = persona,  1 = bicicleta, 2 = coche,       3 = motocicleta,
///   5 = autobús,  7 = camión,   16 = perro,       17 = caballo
const PERSON_CLASS_ID: i32 = 0;

fn main() -> Result<()> {
    // Factor inverso derivado de SCALE: convierte coordenadas del frame reducido
    // al frame original (y viceversa dividiendo). Calculado una sola vez
    let inv_scale = (1.0 / SCALE) as f32; // = 4.0 con SCALE = 0.25

    // 1. Abrir el video desde un archivo
    // "test.mp4" debe existir en el directorio actual
    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        // Si no se puede abrir, se detiene la ejecución y se muestra un mensaje claro
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    // 2. Cargar el modelo ONNX de YOLOv8 y configurar la inferencia en CPU
    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?; // Se puede usar YOLO v5/v7/v8 con este código
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    // 3. Inicializar ByteTrack para seguimiento de objetos
    //   track_thresh: umbral de confianza mínimo para aceptar una detección
    //   max_age: número de frames que un track puede sobrevivir sin detección
    //   match_thresh: umbral de confianza para emparejar detecciones con tracks existentes
    //   det_thresh: umbral de confianza mínimo para considerar una detección en el proceso de matching
    let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.5);
    let mut last_tracks = Vec::new();

    // 4. Configurar la ventana de visualización
    let window = "YOLO v8 (OpenCV) + ByteTrack (TrackForge)";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // Variables del bucle principal
    let mut frame = Mat::default(); // Matriz OpenCV que contendrá cada frame leído del video
    let mut frame_num: usize = 0; // Contador de frames procesados

    println!("Iniciando... Presiona 'q' o 'esc' para salir");

    loop {
        // Leer el siguiente frame del video.
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video, han habido {} frames.", frame_num);
            break;
        }

        // Incrementar al inicio para que se haga aunque haya breaks posteriores
        frame_num += 1;

        // Reducir el frame según SCALE para acelerar la inferencia YOLO
        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,              // Frame original de alta resolución
            &mut frame_small,    // Destino del frame reducido
            Size::new(0, 0),     // Tamaño deseado (0,0 = calcular de los factores)
            SCALE,               // Escala horizontal
            SCALE,               // Escala vertical
            imgproc::INTER_AREA, // Método de interpolación óptimo para reducción
        )?;

        // Ejecutar detección cada N frames; en frames intermedios se reutilizan
        // los tracks del frame anterior para que las cajas no parpadeen.
        // El tracker solo se actualiza cuando hay detecciones reales: así el
        // max_age cuenta ciclos de detección, no frames, que es el comportamiento correcto.
        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            last_tracks = {
                // Crear el blob de entrada normalizado para YOLO.
                let blob = dnn::blob_from_image(
                    &frame_small,                      // Imagen de entrada (frame reducido)
                    1.0 / 255.0, // Factor para normalizar el pixel (1/255 para [0,1])
                    Size::new(INPUT_SIZE, INPUT_SIZE), // Tamaño requerido por la red
                    Scalar::default(), // Resta de media por canal (ninguna por defecto)
                    true,        // true para intercambiar canales R y B si la red lo necesita
                    false,       // recortar a false, sólo redimensionar
                    core::CV_32F, // tipo de dato de salida (CV_32F -> float32)
                )?;

                net.set_input(&blob, "", 1.0, Scalar::default())?;

                // Ejecutar la red y obtener las capas de salida
                let mut output_blobs: Vector<Mat> = Vector::new();
                let out_names = net.get_unconnected_out_layers_names()?; // Nombres de las capas de salida (varían según el modelo)
                net.forward(&mut output_blobs, &out_names)?; // RESULTADO DE LA INFERENCIA !!!!!

                let output = output_blobs.get(0)?;
                let size = output.mat_size(); // Devuelve tamaño salida: (1, num_preds, num_attrs) o (1, num_attrs, num_preds) según el modelo

                // Sacar predicciones y atributos, independientemente del formato. El numero de predicciones es siempre mayor al de atributos
                let is_yolov8 = size[1] < size[2]; // yolov8 tiene formato (1, num_attrs, num_preds), mientras que v5/v7 tienen (1, num_preds, num_attrs)
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

                // Inferir el número de clases desde el tensor para soportar modelos
                // con distinto número de clases (COCO-80, COCO-91, modelos custom…).
                //   YOLOv8: los primeros 4 atributos son cx,cy,w,h → el resto son clases
                //   YOLOv5/v7: los primeros 5 atributos son cx,cy,w,h,objectness → el resto son clases
                let num_classes = if is_yolov8 {
                    num_attrs - 4
                } else {
                    num_attrs - 5
                };

                let mut class_ids = Vector::<i32>::new();
                let mut confidences = Vector::<f32>::new();
                let mut boxes = Vector::<Rect>::new();

                // Factores de escala para convertir coordenadas de INPUT_SIZExINPUT_SIZE
                // al tamaño real del frame reducido
                let x_factor = frame_small.cols() as f32 / INPUT_SIZE as f32;
                let y_factor = frame_small.rows() as f32 / INPUT_SIZE as f32;

                let data = output.data_typed::<f32>()?;

                // Recorrer todas las predicciones devueltas por la red
                for p in 0..num_preds {
                    // Inicializar variables para almacenar la clase con mayor score y su confianza
                    let mut max_class_score = 0.0f32;
                    let mut class_id = 0i32;
                    let confidence: f32;
                    let cx: f32;
                    let cy: f32;
                    let w: f32;
                    let h: f32;

                    if is_yolov8 {
                        // YOLOv8: atributos en filas y predicciones en columnas
                        cx = data[0 * num_preds + p];
                        cy = data[1 * num_preds + p];
                        w = data[2 * num_preds + p];
                        h = data[3 * num_preds + p];

                        for j in 0..num_classes {
                            let score = data[(4 + j) * num_preds + p]; // El score de clase j para la predicción p
                            if score > max_class_score {
                                // Encontrar la clase con el score más alto para esta predicción
                                max_class_score = score;
                                class_id = j as i32; // Representa el índice de la clase con mayor score
                            }
                        }
                        confidence = max_class_score; // La confianza es igual al score de la clase
                    } else {
                        // YOLOv5/v7: predicciones en filas y atributos en columnas
                        let base_idx = p * num_attrs;
                        cx = data[base_idx + 0];
                        cy = data[base_idx + 1];
                        w = data[base_idx + 2];
                        h = data[base_idx + 3];
                        let objectness = data[base_idx + 4]; // Probabilidad de que haya un objeto (independientemente de la clase)

                        for j in 0..num_classes {
                            let score = data[base_idx + 5 + j];
                            if score > max_class_score {
                                max_class_score = score;
                                class_id = j as i32; // Índice de clase con mayor probabilidad.
                            }
                        }
                        confidence = objectness * max_class_score; // La confianza combina objectness con el score de clase
                    }

                    // Filtrar solo personas y detecciones con suficiente confianza
                    if confidence >= CONF_THRESHOLD && class_id == PERSON_CLASS_ID {
                        // Convertir las coordenadas de centro (cx, cy) y tamaño (w, h) a formato Rect (x, y, width, height)
                        // y escalar al tamaño del frame reducido usando los factores calculados
                        let left = ((cx - w / 2.0) * x_factor) as i32;
                        let top = ((cy - h / 2.0) * y_factor) as i32;
                        let width = (w * x_factor) as i32;
                        let height = (h * y_factor) as i32;

                        // Almacenar la detección en los vectores
                        class_ids.push(class_id);
                        confidences.push(confidence);
                        boxes.push(Rect::new(left, top, width, height));
                    }
                }

                // Aplicar Non-Maximum Suppression para eliminar detecciones redundantes
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
                //   boxes:           cajas candidatas
                //   scores:          confianza asociada a cada caja
                //   score_threshold: mínimo score para considerar la caja
                //   nms_threshold:   umbral de supresión entre cajas
                //   indices:         salida con las cajas filtradas
                //   eta:             factor de ajuste del umbral NMS
                //   top_k:           límite de cajas a retener tras NMS (0 = sin límite)

                // Construir el vector de detecciones para ByteTrack
                // Las coordenadas se escalan al frame original no reducido
                let mut detections = Vec::new();
                for idx in indices {
                    let i = idx as usize; // Índice de la detección filtrada por NMS
                    let rect = boxes.get(i)?; // Caja delimitadora: (x, y, width, height) en frame_small
                    let conf = confidences.get(i)?; // Confianza de la detección

                    let bbox = [
                        rect.x as f32 * inv_scale, // → coordenadas en el frame original
                        rect.y as f32 * inv_scale,
                        rect.width as f32 * inv_scale,
                        rect.height as f32 * inv_scale,
                    ];
                    detections.push((bbox, conf, 0));
                }

                // Actualizar el tracker con las detecciones del frame actual
                tracker.update(detections) // Devuelve una lista de tracks con IDs y posiciones actualizadas
                                           // RESULTADO DE LA ACTUALIZACIÓN DEL TRACKER !
            };
        }

        // Dibujar resultados de tracking en el frame reducido
        // Los tracks almacenan coordenadas del frame original → obtener la posición correcta sobre frame_small
        for track in &last_tracks {
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

        // Mostrar el frame procesado
        highgui::imshow(window, &frame_small)?;

        // Salir si se presiona 'q' o ESC
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    Ok(())
}
