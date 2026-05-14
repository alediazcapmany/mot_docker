use nalgebra::{SMatrix, SVector}; // Para vectores y matrices de estado
use std::collections::HashMap; // Para almacenar el historial de cada track

use super::kalman_filter::KalmanFilter; // Importa tu archivo de KalmanFilter

pub type StateVec = SVector<f32, 8>; // [cx, cy, w, h, vx, vy, vw, vh]
pub type StateMat = SMatrix<f32, 8, 8>; // Matriz de covarianza 8x8

// --- 1. UTILIDADES Y ESTADOS ------------------------------------------------------------------------
/// Función para calcular la velocidad entre dos cajas (x1y1x2y2)
fn get_vel(b_1: &[f32; 4], b_2: &[f32; 4]) -> [[f32; 2]; 4] {
    // Diferencia de posición entre las dos cajas
    let deltas = [
        b_2[0] - b_1[0],
        b_2[1] - b_1[1],
        b_2[2] - b_1[2],
        b_2[3] - b_1[3],
    ];

    let epsilon = 1e-5; // Para evitar división por cero
                        // Normas para normalizar las velocidades
    let norm_lt = (deltas[0].powi(2) + deltas[1].powi(2)).sqrt() + epsilon;
    let norm_lb = (deltas[0].powi(2) + deltas[3].powi(2)).sqrt() + epsilon;
    let norm_rt = (deltas[2].powi(2) + deltas[1].powi(2)).sqrt() + epsilon;
    let norm_rb = (deltas[2].powi(2) + deltas[3].powi(2)).sqrt() + epsilon;

    // Devuelve las velocidades normalizadas para cada esquina (lt, lb, rt, rb)
    [
        [(b_2[0] - b_1[0]) / norm_lt, (b_2[1] - b_1[1]) / norm_lt],
        [(b_2[0] - b_1[0]) / norm_lb, (b_2[3] - b_1[3]) / norm_lb],
        [(b_2[2] - b_1[2]) / norm_rt, (b_2[1] - b_1[1]) / norm_rt],
        [(b_2[2] - b_1[2]) / norm_rb, (b_2[3] - b_1[3]) / norm_rb],
    ]
}

/// Estados posibles de un track
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    New = 0,
    Confirmed = 1,
    Lost = 2,
    Deleted = 3,
}

/// Contador de IDs para asignar a cada nuevo track
pub struct TrackCounter {
    track_count: usize,
}

impl TrackCounter {
    pub fn new() -> Self {
        Self { track_count: 0 }
    }
    pub fn get_track_id(&mut self) -> usize {
        self.track_count += 1;
        self.track_count
    }
}

/// Estructura con el historial de cada track
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub box_x1y1x2y2: [f32; 4], // Caja en formato x1y1x2y2
    pub score: f32,             // Score de la detección
    pub mean: StateVec,         // Estado del filtro de Kalman (cx, cy, w, h, vx, vy, vw, vh)
    pub covariance: StateMat,   // Matriz de covarianza del filtro de Kalman
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub feat: Vec<f32>, // Para la distancia coseno
}

// --- 2. INFO DEL TRACK ------------------------------------------------------------------------
#[derive(Clone)]
pub struct Track {
    // Info general del track
    pub track_id: usize,     // ID único del track
    pub end_frame_id: usize, // Ultimo frame donde se detecto el track
    pub state: TrackState,   // Estado actual del track (New, Confirmed, Lost, Deleted)

    // Info
    pub box_x1y1x2y2: [f32; 4], // Caja en formato x1y1x2y2
    pub score: f32,             // Score de la detección
    pub feat: Vec<f32>,         // Características para la distancia coseno (si las usas)

    // Info para calcular la velocidad
    pub delta_t: usize, // Número de frames para calcular la velocidad (ajustable)
    pub history: HashMap<usize, HistoryEntry>, // Historial de detecciones para calcular la velocidad
    pub max_history: usize, // Número máximo de entradas en el historial (ajustable)

    // Filtro de Kalman
    pub kalman_filter: Option<KalmanFilter>, // Filtro de Kalman para este track (inicialmente None)

    // Estado actual del filtro de Kalman
    pub mean: Option<StateVec>, // Estado actual del filtro de Kalman (cx, cy, w, h, vx, vy, vw, vh)
    pub covariance: Option<StateMat>, // Matriz de covarianza actual del filtro de Kalman
    pub velocity: [[f32; 2]; 4], // Velocidad de cada esquina (lt, lb, rt, rb) calculada a partir del historial

    // Parámetros para la lógica de tracking
    pub min_len: usize, // Número mínimo de detecciones para ser "Confirmed"
    pub is_dance_dataset: bool,
}

impl Track {
    // Crea un nuevo track con los parámetros dados
    pub fn new(box_x1y1x2y2: [f32; 4], score: f32, min_len: usize, is_dance_dataset: bool) -> Self {
        Self {
            track_id: 0,             // Se asigna en initiate()
            end_frame_id: 0,         // Se actualiza en initiate() y update()
            state: TrackState::New,  // Estado inicial "New"
            box_x1y1x2y2,            // Caja inicial
            score,                   // Score inicial
            feat: Vec::new(),        // Características vacías al inicio (si las usas)
            delta_t: 3,              // Frames para calcular la velocidad (ajustable)
            history: HashMap::new(), // Historial vacío al inicio
            max_history: 30,         // Máximo número de entradas en el historial (ajustable)
            kalman_filter: None,     // Se inicializa en initiate()
            mean: None,              // Se inicializa en initiate()
            covariance: None,        // Se inicializa en initiate()
            velocity: [[0.0; 2]; 4], // Velocidad inicial cero
            min_len,                 // Número mínimo de detecciones para "Confirmed"
            is_dance_dataset,
        }
    }

    // Métodos para actualizar el estado del track
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost; // Perdido pero recuperable
    }

    pub fn mark_deleted(&mut self) {
        self.state = TrackState::Deleted; // Demasiado perdido, se elimina
    }

    // Inicia el track y configura el filtro de Kalman
    pub fn initiate(&mut self, frame_id: usize, counter: &mut TrackCounter) {
        self.track_id = counter.get_track_id();

        let kf = KalmanFilter::new();
        // let current_cxcywh = self.cxcywh();
        let (mean, covariance) = kf.initiate(&self.cxcywh());

        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.kalman_filter = Some(kf);

        // Guardamos la detección inicial en el historial
        self.history.insert(
            frame_id,
            HistoryEntry {
                box_x1y1x2y2: self.box_x1y1x2y2,
                score: self.score,
                mean: self.mean.unwrap(),
                covariance: self.covariance.unwrap(),
            },
        );

        self.end_frame_id = frame_id; // Frame donde se inicia el track
        self.state = TrackState::New; // Estado inicial "New"
    }

    /// Predecir proxima posición con Kalman
    pub fn predict(&mut self) {
        // Si el track no está confirmado y es del dataset de baile, bloqueamos las velocidades de tamaño (vw, vh)
        if let Some(ref mut mean) = self.mean {
            if self.state != TrackState::Confirmed && self.is_dance_dataset {
                mean[6] = 0.0;
                mean[7] = 0.0;
            }
        }

        // Delegamos la predicción al filtro de Kalman
        if let (Some(kf), Some(mean), Some(cov)) =
            (&self.kalman_filter, &self.mean, &self.covariance)
        {
            let (new_mean, new_cov) = kf.predict(mean, cov);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }
    }

    /// Actualiza el track con una nueva detección y actualiza el filtro de Kalman
    pub fn update(&mut self, frame_id: usize, det: &Detection) {
        // Convertir caja a cxcywh para actualizar el filtro
        let cx = (det.bbox[0] + det.bbox[2]) / 2.0;
        let cy = (det.bbox[1] + det.bbox[3]) / 2.0;
        let w = det.bbox[2] - det.bbox[0];
        let h = det.bbox[3] - det.bbox[1];
        let det_cxcywh = [cx, cy, w, h];

        // Actualiza el filtro de Kalman con la nueva detección
        if let (Some(kf), Some(mean), Some(cov)) =
            (&mut self.kalman_filter, &self.mean, &self.covariance)
        {
            let (new_mean, new_cov) = kf.update(mean, cov, &det_cxcywh, det.score);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }

        // Actualiza historial con la nueva detección y el estado del filtro
        if let (Some(mean), Some(cov)) = (&self.mean, &self.covariance) {
            self.history.insert(
                frame_id,
                HistoryEntry {
                    box_x1y1x2y2: det.bbox,
                    score: det.score,
                    mean: *mean,
                    covariance: *cov,
                },
            );
            // Mantiene el historial dentro del límite máximo
            if self.history.len() > self.max_history {
                if let Some(&oldest_frame) = self.history.keys().min() {
                    self.history.remove(&oldest_frame);
                }
            }
        }

        // Actualiza velocidades de caja
        self.velocity = [[0.0; 2]; 4]; // Reinicia velocidades

        // Calcula la velocidad promedio de cada esquina usando el historial de detecciones
        for d_t in 1..=self.delta_t {
            if frame_id >= d_t {
                let target_frame = frame_id - d_t;

                // Busca la detección más cercana en el historial para el frame objetivo
                if let Some(prev_entry) = self.history.get(&target_frame) {
                    let prev_box = prev_entry.box_x1y1x2y2;
                    let vels = get_vel(&prev_box, &det.bbox);

                    for i in 0..4 {
                        self.velocity[i][0] += vels[i][0] / d_t as f32;
                        self.velocity[i][1] += vels[i][1] / d_t as f32;
                    }
                }
            }
        }

        // Promedia las velocidades si se han calculado para al menos un frame
        for i in 0..4 {
            self.velocity[i][0] /= self.delta_t as f32;
            self.velocity[i][1] /= self.delta_t as f32;
        }

        // Actualiza la caja, score, frame final y estado del track
        self.box_x1y1x2y2 = det.bbox;
        self.score = det.score;
        self.end_frame_id = frame_id;

        self.state = if self.history.len() >= self.min_len {
            TrackState::Confirmed
        } else {
            TrackState::New
        };

        // Esto va dentro de la función de tu struct Track que se llama al confirmar un match.
        // Asumiendo que recibes la nueva detección como `det`.

        // Si el track aún no tiene vector (estaba vacío), simplemente lo copiamos
        if self.feat.is_empty() {
            self.feat = det.feat.clone();
        } else if !det.feat.is_empty() {
            // Mezclamos: 90% memoria antigua, 10% nueva vista (Alfa = 0.9 suele ser el estándar)
            let alpha = 0.9_f32;
            let mut sum_sq = 0.0;

            for i in 0..self.feat.len() {
                self.feat[i] = alpha * self.feat[i] + (1.0 - alpha) * det.feat[i];
                sum_sq += self.feat[i] * self.feat[i];
            }

            // Como hemos alterado los números, tenemos que volver a aplicar la Normalización L2
            // para que la distancia Coseno siga funcionando en el próximo frame.
            let norm = sum_sq.sqrt().max(1e-12);
            for i in 0..self.feat.len() {
                self.feat[i] /= norm;
            }
        }
    }

    // --- 3. FORMATO DE CAJAS ------------------------------------------------------------------------
    // Caja en formato cxcywh
    pub fn cxcywh(&self) -> [f32; 4] {
        if let Some(ref mean) = self.mean {
            [mean[0], mean[1], mean[2], mean[3]]
        } else {
            let cx = (self.box_x1y1x2y2[0] + self.box_x1y1x2y2[2]) / 2.0;
            let cy = (self.box_x1y1x2y2[1] + self.box_x1y1x2y2[3]) / 2.0;
            let w = self.box_x1y1x2y2[2] - self.box_x1y1x2y2[0];
            let h = self.box_x1y1x2y2[3] - self.box_x1y1x2y2[1];
            [cx, cy, w, h]
        }
    }
    // Caja en formato x1y1wh
    pub fn x1y1wh(&self) -> [f32; 4] {
        if let Some(ref mean) = self.mean {
            let x1 = mean[0] - mean[2] / 2.0;
            let y1 = mean[1] - mean[3] / 2.0;
            [x1, y1, mean[2], mean[3]]
        } else {
            let w = self.box_x1y1x2y2[2] - self.box_x1y1x2y2[0];
            let h = self.box_x1y1x2y2[3] - self.box_x1y1x2y2[1];
            [self.box_x1y1x2y2[0], self.box_x1y1x2y2[1], w, h]
        }
    }
    // Caja en formato x1y1x2y2
    pub fn x1y1x2y2(&self) -> [f32; 4] {
        if let Some(ref mean) = self.mean {
            let x1 = mean[0] - mean[2] / 2.0;
            let y1 = mean[1] - mean[3] / 2.0;
            let x2 = mean[0] + mean[2] / 2.0;
            let y2 = mean[1] + mean[3] / 2.0;
            [x1, y1, x2, y2]
        } else {
            self.box_x1y1x2y2
        }
    }
}
