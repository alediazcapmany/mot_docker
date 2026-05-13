// Compensación de Movimiento de Cámara (CMC) -> La saca de archivo pre-calculado por el tracker GMC -> MOT17,MOT20 y Dancetrack
// Se encarga de leer las matrices de warp frame a frame y aplicarlas a los tracks para compensar el movimiento de cámara
use super::track::{StateMat, Track}; // Importar estructuras desde track.rs
use std::fs::File; // Para manejar archivos
use std::io::{BufRead, BufReader}; // Para lectura de archivos

pub struct Cmc {
    reader: Option<BufReader<File>>, // Option para manejar la no existencia del archivo de CMC
}

impl Cmc {
    // Lee el archivo pre-calculado de movimientos de cámara
    pub fn new(vid_name: &str) -> Self {
        let mut parsed_name = vid_name.to_string();

        // Ajustamos el nombre del archivo según el dataset
        if vid_name.contains("MOT17") {
            parsed_name = vid_name
                .split("-FRCNN")
                .next()
                .unwrap_or(vid_name)
                .to_string();
        } else if vid_name.contains("dance") {
            if let Some(suffix) = vid_name.split("dancetrack").nth(1) {
                parsed_name = format!("dancetrack{}", suffix);
            }
        }

        // Abrimos el archivo correspondiente al video
        let file_path = format!("./src/tracktrack/cmc/GMC-{}.txt", parsed_name);
        let reader = match File::open(&file_path) {
            Ok(file) => Some(BufReader::new(file)),
            Err(_) => {
                println!("[AVISO] Archivo CMC no encontrado: {}. El tracker funcionará sin compensación de cámara.", file_path);
                None
            }
        };

        Self { reader }
    }

    // Devuelve la matriz frame a frame
    pub fn get_warp_matrix(&mut self) -> Option<[[f32; 3]; 2]> {
        // Si no se encontró el archivo al inicio, no hacemos nada y devolvemos None
        let reader = self.reader.as_mut()?;

        let mut line = String::new();

        // Lee siguiente línea, si no hay termina archivo delvoviendo None
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return None; // Fin del archivo
        }

        // Parseamos la línea para extraer la matriz de warp
        let tokens: Vec<&str> = line.trim().split('\t').collect();
        if tokens.len() < 7 {
            return None;
        }

        // Construimos la matriz de warp a partir de los tokens
        let mut warp_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        warp_matrix[0][0] = tokens[1].parse().unwrap_or(1.0);
        warp_matrix[0][1] = tokens[2].parse().unwrap_or(0.0);
        warp_matrix[0][2] = tokens[3].parse().unwrap_or(0.0);

        warp_matrix[1][0] = tokens[4].parse().unwrap_or(0.0);
        warp_matrix[1][1] = tokens[5].parse().unwrap_or(1.0);
        warp_matrix[1][2] = tokens[6].parse().unwrap_or(0.0);

        Some(warp_matrix)
    }
}

/// Aplica la compensación de cámara a los Tracks modificando su estado de Kalman
pub fn apply_cmc(tracks: &mut [Track], warp_matrix: &[[f32; 3]; 2]) {
    for track in tracks.iter_mut() {
        // Si el track no tiene Kalman inicializado aún, no hacemos nada
        if let (Some(mean), Some(cov)) = (&mut track.mean, &mut track.covariance) {
            // 1. Matriz de rotación 8x8 con warp, sin traslación
            let mut rot_8x8 = StateMat::zeros();
            for i in 0..4 {
                rot_8x8[(i * 2, i * 2)] = warp_matrix[0][0];
                rot_8x8[(i * 2, i * 2 + 1)] = warp_matrix[0][1];
                rot_8x8[(i * 2 + 1, i * 2)] = warp_matrix[1][0];
                rot_8x8[(i * 2 + 1, i * 2 + 1)] = warp_matrix[1][1];
            }

            // 2. Extraemos la traslación
            let trans_x = warp_matrix[0][2];
            let trans_y = warp_matrix[1][2];

            // 3. Warp del mean: rot_8x8 @ mean + traslación
            *mean = rot_8x8 * (*mean);
            mean[0] += trans_x;
            mean[1] += trans_y;

            // 4. Warp a la Covarianza: rot_8x8 @ cov @ rot_8x8.T
            *cov = rot_8x8 * (*cov) * rot_8x8.transpose();
        }
    }
}
