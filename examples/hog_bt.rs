// Ejemplo de MOT básico usando HOG para detección (con OpenCV) y ByteTrack para tracking (con TrackForge).

// Dependencias
use opencv::{
    Result,
    core::{Point, Rect, Scalar, Size, Vector},
    highgui, imgproc, objdetect,
    prelude::*,
    videoio,
};

use trackforge::trackers::byte_track::ByteTrack; // Importar el tracker ByteTrack desde el paquete trackforge.

// ── Constantes de configuración ──────────────────────────────────────────────
// Centralizar aquí todos los parámetros facilita ajustes sin tocar la lógica.

/// Factor de escala al reducir el frame antes de la detección HOG.
/// 0.25 = 25% del tamaño original → detección ~16× más rápida con pérdida mínima de precisión.
const SCALE: f64 = 0.25;

/// Ejecutar detección HOG solo cada N frames para ahorrar CPU.
const DETECT_EVERY_N_FRAMES: usize = 2;

/// Confianza mínima del SVM para aceptar una detección como válida (filtro de falsos positivos).
const WEIGHT_THRESHOLD: f32 = 0.5;

fn main() -> Result<()> {
    // Factor inverso derivado de SCALE: convierte coordenadas del frame reducido
    // al frame original (y viceversa dividiendo). Calculado una sola vez.
    let inv_scale = (1.0 / SCALE) as f32; // = 4.0 con SCALE = 0.25

    // 1. Abrir el video desde un archivo
    // "test.mp4" debe existir en la misma carpeta donde se ejecuta el programa.
    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        // Si no se puede abrir, se detiene la ejecución y se muestra un mensaje claro.
        panic!("No se pudo abrir el video. Revisa la ruta.");
    }

    // 2. Configurar el descriptor HOG para detección de personas
    let mut hog = objdetect::HOGDescriptor::default()?;

    // Configurar el detector SVM (Support Vector Machine) integrado para detección de personas
    hog.set_svm_detector(objdetect::HOGDescriptor::get_default_people_detector()?); // Devuelve un vector de pesos preentrenados para detectar personas usando HOG+SVM

    // 3. Configurar el tracker ByteTrack para seguimiento de objetos
    //   track_thresh: umbral de confianza mínimo para aceptar una detección
    //   max_age:      frames sin detección antes de eliminar un track
    //   match_thresh: umbral de IoU para emparejar detecciones con tracks existentes
    //   det_thresh:   confianza mínima para considerar una detección en el proceso de matching
    let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.5);

    // 4. Configurar la ventana de visualización
    let window = "MOT Basico - Fase 1";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // Variables para el bucle principal
    let mut frame = Mat::default();      // Matriz OpenCV que contendrá cada frame leído del video
    let mut frame_num: usize = 0;        // Contador de frames procesados
    let mut last_tracks = Vec::new();    // Tracks del último frame con detección; se reutilizan
                                         // en frames intermedios para evitar parpadeo de cajas

    println!("Iniciando procesamiento... Presiona 'q' o ESC en la ventana para salir.");

    // BUCLE PRINCIPAL DE PROCESAMIENTO
    // lectura de frame -> detección opcional -> tracking -> dibujo y visualización
    loop {
        // 1. Leer el siguiente frame del video en la matriz 'frame'
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video.");
            break;
        }

        // Incrementar al inicio para que los breaks no alteren la paridad del contador.
        frame_num += 1;

        // 2. Redimensionar el frame según SCALE para acelerar la detección HOG.
        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,              // Frame original de alta resolución
            &mut frame_small,    // Destino del frame reducido
            Size::new(0, 0),     // Tamaño deseado (0,0 = calcular de los factores)
            SCALE,               // Escala horizontal
            SCALE,               // Escala vertical
            imgproc::INTER_AREA, // Método de interpolación óptimo para reducción
        )?;

        // Detección de personas con HOG cada N frames.
        // En frames intermedios se reutilizan last_tracks para que las cajas no parpadeen.
        // El tracker solo se actualiza cuando hay detecciones reales: así el max_age
        // cuenta ciclos de detección, no frames, que es el comportamiento correcto.
        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            let mut found_locations = Vector::<Rect>::new(); // Rectángulos detectados por HOG
            let mut found_weights   = Vector::<f64>::new();  // Pesos de confianza asociados a cada detección (score del SVM)

            // 3. Ejecutar detección multi-escala con pesos usando el descriptor HOG configurado.
            // detect_multi_scale_weights detecta objetos en la imagen a múltiples escalas
            // y devuelve tanto las ubicaciones (bounding boxes) como los pesos (scores de confianza)
            // para cada detección.
            hog.detect_multi_scale_weights(
                &frame_small,         // Imagen de entrada (resolución reducida)
                &mut found_locations, // Salida: bounding boxes detectados
                &mut found_weights,   // Salida: scores de confianza
                0.0,                  // hit_threshold: umbral detector
                Size::new(16, 16),    // win_stride: paso ventana
                Size::new(0, 0),      // padding: acolchado adicional
                1.05,                 // scale: factor escala pirámide
                1.05,                 // final_threshold: umbral agrupamiento
                false,                // use_meanshift_grouping: sin meanshift
            )?;

            // 4. Preparar las detecciones para el tracker ByteTrack de TrackForge.
            // Cada detección es una tupla (bounding_box, confianza, clase) donde:
            // - bounding_box: array [x, y, width, height] en coordenadas del frame original
            // - confianza:    f32 entre 0.0 y 1.0 (score del detector)
            // - clase:        entero identificando tipo de objeto (0 = persona en MOT estándar)
            let mut detections = Vec::new();

            // 5. Iterar sobre todas las detecciones encontradas por HOG.
            for i in 0..found_locations.len() {
                // Obtener el rectángulo y el peso de la detección actual.
                let rect   = found_locations.get(i)?; // Rectángulo detectado: (x, y, width, height) en frame_small
                let weight = found_weights.get(i)?;   // Score de confianza del SVM para esta detección

                // Filtrar detecciones con baja confianza para reducir ruido y falsos positivos.
                if weight as f32 > WEIGHT_THRESHOLD {
                    // Escalar las coordenadas al frame original multiplicando por inv_scale,
                    // de modo que el tracker opera siempre en el espacio de coordenadas real.
                    let bbox = [
                        rect.x      as f32 * inv_scale,
                        rect.y      as f32 * inv_scale,
                        rect.width  as f32 * inv_scale,
                        rect.height as f32 * inv_scale,
                    ];

                    // Agregar la detección al vector en formato ByteTrack de TrackForge: (bbox, confianza, clase)
                    detections.push((bbox, weight as f32, 0));
                }
            }

            // Actualizar el tracker con las detecciones del frame actual.
            // Devuelve una lista de tracks con IDs y posiciones actualizadas.
            last_tracks = tracker.update(detections);
        }

        // Si no toca detectar, se reutiliza last_tracks sin modificar.

        // 6. Dibujar resultados.
        // Los tracks almacenan coordenadas del frame original → dividir por inv_scale
        // para obtener la posición correcta sobre frame_small.
        for track in &last_tracks {
            let bbox = track.tlwh; // Posición del track en formato (x, y, width, height) en coordenadas del frame original
            let rect = Rect::new(  // Convertir las coordenadas del track a formato Rect para dibujar en OpenCV
                (bbox[0] / inv_scale) as i32,
                (bbox[1] / inv_scale) as i32,
                (bbox[2] / inv_scale) as i32,
                (bbox[3] / inv_scale) as i32,
            );

            // Dibujar la caja verde alrededor de la persona
            imgproc::rectangle(
                &mut frame_small,
                rect,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
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
                0.6,
                Scalar::new(0.0, 0.0, 255.0, 0.0), // rojo
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        // 7. Mostrar frame procesado
        highgui::imshow(window, &frame_small)?;

        // Salir si se presiona 'q' o ESC
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    // Si el bucle termina correctamente, devolver Ok para indicar éxito.
    Ok(())
}