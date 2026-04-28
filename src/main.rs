// Dependencias
use opencv::{
    Result,
    core::{Point, Rect, Scalar, Size, Vector},
    highgui, imgproc, objdetect,
    prelude::*,
    videoio,
};

use trackforge::trackers::byte_track::ByteTrack; // Importar el tracker ByteTrack desde el paquete trackforge.

fn main() -> Result<()> {
    // 1. Abrir el video desde un archivo.
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
    let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.5);
    // track_thresh: umbral de confianza para detecciones, max_age: frames sin detección antes de eliminar track, n_init: frames para confirmar track, min_confidence: confianza mínima para matching

    // Constante que define cada cuántos frames se ejecuta la detección HOG
    const DETECT_EVERY_N_FRAMES: usize = 2;

    // 4. Configurar la ventana de visualización
    let window = "MOT Basico - Fase 1";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // Variables para el bucle principal
    let mut frame = Mat::default(); // Matriz OpenCV que contendrá cada frame leído del video
    let mut frame_num: usize = 0; // Contador de frames procesados
    let mut last_tracks = Vec::new(); // Vector para almacenar los tracks del último frame donde se ejecutó detección, se reutiliza en frames intermedios

    println!("Iniciando procesamiento... Presiona 'q' en la ventana para salir.");

    // BUCLE PRINCIPAL DE PROCESAMIENTO
    // lectura de frame -> detección opcional -> tracking -> dibujo y visualización
    loop {
        // 1. Leer el siguiente frame del video en la matriz 'frame'
        cam.read(&mut frame)?;

        if frame.empty() {
            // Imprimir mensaje informativo y salir del bucle.
            println!("Fin del video.");
            break;
        }

        // 2. Redimensionar el frame a la mitad para acelerar la detección HOG
        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,              // Frame original de alta resolución
            &mut frame_small,    // Destino del frame reducido
            Size::new(0, 0),     // Tamaño deseado (0,0 = calcular de los factores)
            0.25,                // Escala horizontal
            0.25,                // Escala vertical
            imgproc::INTER_AREA, // Método de interpolación óptimo para reducción
        )?;

        //  Detección de personas con HOG cada N frames
        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            let mut found_locations = Vector::<Rect>::new(); // Rectángulos detectados por HOG
            let mut found_weights = Vector::<f64>::new(); // Pesos de confianza asociados a cada detección (score del SVM)

            // 3. Ejecutar detección multi-escala con pesos usando el descriptor HOG configurado
            // detect_multi_scale_weights es una función que detecta objetos en la imagen a múltiples escalas
            // y devuelve tanto las ubicaciones (bounding boxes) como los pesos (scores de confianza) para cada detección
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

            // 4. Preparar las detecciones para el tracker ByteTrack de TrackForge
            // Cada detección es una tupla (bounding_box, confianza, clase) donde:
            // - bounding_box: array [x, y, width, height] en coordenadas absolutas
            // - confianza: f32 entre 0.0 y 1.0 (score del detector)
            // - clase: entero identificando tipo de objeto (0 = persona en MOT estándar)
            let mut detections = Vec::new();

            // 5. Iterar sobre todas las detecciones encontradas por HOG.
            for i in 0..found_locations.len() {
                // Obtener el rectángulo y el peso de la detección actual.
                let rect = found_locations.get(i)?; // Rectángulo detectado: (x, y, width, height)
                let weight = found_weights.get(i)?; // Score de confianza del SVM para esta detección

                // Filtrar detecciones con baja confianza para reducir ruido y falsos positivos
                if weight > 0.5 {
                    // Duplicar resolución para mostrar al 50%
                    let bbox = [
                        rect.x as f32 * 2.0,      
                        rect.y as f32 * 2.0,      
                        rect.width as f32 * 2.0,  
                        rect.height as f32 * 2.0,
                    ];

                    // Agregar la detección al vector en formato ByteTrack de TrackForge: (bbox, confianza, clase)
                    detections.push((bbox, weight as f32, 0));
                }
            }

            // Actualizar el tracker con las detecciones del frame actual, devuelve una lista de "tracks" con IDs y posiciones actualizadas
            last_tracks = tracker.update(detections);
        }

        // Si no toca detectar, usamos last_tracks

        // 6. Dibujar resultados
        for track in &last_tracks {
            let bbox = track.tlwh;  // Contiene la posición del track en formato (x, y, width, height) en coordenadas absolutas
            let rect = Rect::new(   // Convertir las coordenadas del track a formato Rect para dibujar en OpenCV
                (bbox[0] / 2.0) as i32,
                (bbox[1] / 2.0) as i32,
                (bbox[2] / 2.0) as i32,
                (bbox[3] / 2.0) as i32,
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
            let pos = Point::new((bbox[0] / 2.0) as i32, (bbox[1] / 2.0 - 10.0) as i32);
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

        // 7. Mostrar frame
        // ya al 50%
        highgui::imshow(window, &frame_small)?;

        // Esperar 1 ms y comprobar si se presiona la tecla 'q' -> Velocidad de reproducción del video
        if highgui::wait_key(1)? == 113 {
            break;
        }

        frame_num += 1;
    }

    // Si el bucle termina correctamente, devolver Ok para indicar éxito.
    Ok(())
}
