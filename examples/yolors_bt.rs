// Este ejemplo usa el crate yolo_detector para simplificar la inferencia YOLO:
// ya no hay que crear blobs manualmente, parsear tensores ni aplicar NMS a mano
// El crate encapsula todo eso en YoloDetector::detect() y get_detections_with_classes().
//
// Flujo:
//   VideoCapture → frame_small → YoloDetector::detect() → get_detections_with_classes()
//   → filtrar "person" → ByteTrack → dibujar cajas con ID
use opencv::{
    core::{Point, Rect, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio, Result,
};
use trackforge::trackers::byte_track::ByteTrack;
use yolo_detector::YoloDetector;

// ── Constantes de configuración ──────────────────────────────────────────────

/// Factor de escala al reducir el frame antes de la inferencia.
/// 0.25 = 25% del tamaño original → 4× más rápido con pérdida mínima de precisión.
const SCALE: f64 = 0.25;

/// Tamaño de entrada requerido por el modelo YOLO (cuadrado).
const INPUT_SIZE: i32 = 640;

/// Ejecutar detección solo cada N frames para ahorrar CPU.
const DETECT_EVERY_N_FRAMES: usize = 2;

/// Umbral de confianza mínimo para aceptar una detección.
/// NOTA: yolo_detector no devuelve la confianza individual de cada detección
/// en get_detections_with_classes(), así que también usamos este valor como
/// confianza fija al construir las detecciones para ByteTrack. Las detecciones
/// ya superaron este umbral, así que es una aproximación conservadora válida
const CONF_THRESHOLD: f32 = 0.5;

/// Umbral de IoU para Non-Maximum Suppression interno del crate
const NMS_THRESHOLD: f32 = 0.4;

/// Nombre de la clase objetivo según coco.names
/// NOTA: yolo_detector devuelve los nombres con comillas literales incluidas (p.ej. '"person"' en lugar de 'person')
const TARGET_CLASS: &str = "person";

fn main() -> Result<()> {
    // Factor inverso derivado de SCALE: convierte coordenadas del frame reducido
    // al frame original (y viceversa dividiendo). Calculado una sola vez.
    let inv_scale = (1.0 / SCALE) as f32; // = 4.0 con SCALE = 0.25

    // 1. Abrir el video desde un archivo
    // "test.mp4" debe existir en el directorio actual
    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        // Si no se puede abrir, se detiene la ejecución y se muestra un mensaje claro
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    // 2. Inicializar el detector YOLO con el crate yolo_detector.
    let detector = YoloDetector::new("yolov8n.onnx", "coco.names", INPUT_SIZE)
        .expect("No se pudo cargar el modelo YOLO. Revisa las rutas de .onnx y .names.");

    // 3. Inicializar ByteTrack para seguimiento de objetos
    //   track_thresh: umbral de confianza mínimo para aceptar una detección
    //   max_age:      número de frames que un track puede sobrevivir sin detección
    //   match_thresh: umbral de IoU para emparejar detecciones con tracks existentes
    //   det_thresh:   umbral de confianza mínimo para considerar una detección en el proceso de matching
    let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.5);
    let mut last_tracks = Vec::new();

    // 4. Configurar la ventana de visualización
    let window = "YOLO v8 (yolo_detector) + ByteTrack (TrackForge)";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // Variables del bucle principal
    let mut frame = Mat::default(); // Matriz OpenCV que contendrá cada frame leído del video
    let mut frame_num: usize = 0; // Contador de frames procesados

    println!("Iniciando... Presiona 'q' o ESC para salir.");

    loop {
        // Leer el siguiente frame del video
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video.");
            break;
        }

        // Incrementar al inicio para que los breaks no alteren la paridad del contador
        frame_num += 1;

        // Reducir el frame según SCALE para acelerar la detección YOLO
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
            let (detections, original_size) = detector.detect(&frame_small)?;

            // Devuelve una lista de detecciones con sus clases asociadas
            let detections_with_classes = detector.get_detections_with_classes(
                detections,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
                original_size,
            );

            let mut byte_track_detections = Vec::new();
            for (class_name, rect) in detections_with_classes {
                // El crate incluye comillas literales en el nombre: '"person"' → trim antes de comparar
                if class_name.trim_matches('"') == TARGET_CLASS {
                    let bbox = [
                        rect.x as f32 * inv_scale,
                        rect.y as f32 * inv_scale,
                        rect.width as f32 * inv_scale,
                        rect.height as f32 * inv_scale,
                    ];
                    // Pasamos 1.0 como confianza porque las detecciones ya superaron CONF_THRESHOLD
                    byte_track_detections.push((bbox, 1.0_f32, 0));
                }
            }

            // Actualizar el tracker con las detecciones del frame actual.
            // Devuelve una lista de tracks con IDs y posiciones actualizadas.
            last_tracks = tracker.update(byte_track_detections);
        }

        // Si no toca detectar, se reutiliza last_tracks sin modificar

        // Dibujar resultados de tracking en el frame reducido
        // Los tracks almacenan coordenadas del frame original → dividir por inv_scale
        // para obtener la posición correcta sobre frame_small
        for track in &last_tracks {
            let bbox = track.tlwh; // Posición del track: (x, y, width, height) en coordenadas del frame original
            let rect_draw = Rect::new(
                // Convertir a coords de frame_small para dibujar
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
