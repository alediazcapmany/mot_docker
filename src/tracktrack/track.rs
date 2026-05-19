use nalgebra::{SMatrix, SVector};
use std::collections::HashMap;

use super::kalman_filter::KalmanFilter;

// --- DEFINICIONES DE TIPOS ---
// El vector de estado incluye: [centro_x, centro_y, ancho, alto, vel_cx, vel_cy, vel_ancho, vel_alto]
pub type StateVec = SVector<f64, 8>;
// Matriz de covarianza 8x8 que representa la incertidumbre del estado actual
pub type StateMat = SMatrix<f64, 8, 8>;

// --- 1. UTILIDADES Y ESTADOS ------------------------------------------------------------------------
/// Calcula la velocidad direccional normalizada de las 4 esquinas entre dos cajas
fn get_vel(b_1: &[f64; 4], b_2: &[f64; 4]) -> [[f64; 2]; 4] {
    // Diferencias de posición puras (x, y, w, h)
    let deltas = [
        b_2[0] - b_1[0],
        b_2[1] - b_1[1],
        b_2[2] - b_1[2],
        b_2[3] - b_1[3],
    ];

    let epsilon = 1e-5; // Previene divisiones por cero

    // Calculamos la magnitud del movimiento para cada esquina (Top-Left, Bottom-Left, Top-Right, Bottom-Right)
    let norm_lt = (deltas[0].powi(2) + deltas[1].powi(2)).sqrt() + epsilon;
    let norm_lb = (deltas[0].powi(2) + deltas[3].powi(2)).sqrt() + epsilon;
    let norm_rt = (deltas[2].powi(2) + deltas[1].powi(2)).sqrt() + epsilon;
    let norm_rb = (deltas[2].powi(2) + deltas[3].powi(2)).sqrt() + epsilon;

    // Retorna los vectores de velocidad [vx, vy] normalizados por esquina
    [
        [(b_2[0] - b_1[0]) / norm_lt, (b_2[1] - b_1[1]) / norm_lt],
        [(b_2[0] - b_1[0]) / norm_lb, (b_2[3] - b_1[3]) / norm_lb],
        [(b_2[2] - b_1[2]) / norm_rt, (b_2[1] - b_1[1]) / norm_rt],
        [(b_2[2] - b_1[2]) / norm_rb, (b_2[3] - b_1[3]) / norm_rb],
    ]
}

/// Ciclo de vida de un objeto trackeado
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    New = 0,       // Recién detectado, aún no es fiable
    Confirmed = 1, // Consolidado (ha aparecido varios frames seguidos)
    Lost = 2,      // No se ve, pero Kalman intenta adivinar dónde está
    Deleted = 3,   // Demasiado tiempo perdido, se elimina del sistema
}

/// Generador simple de IDs incrementales para nuevos tracks
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

/// Estado actual del track
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub box_x1y1x2y2: [f64; 4], // Posición real
    pub score: f64,             // Confianza de la detección en ese momento
    pub mean: StateVec,         // Estado interno de Kalman
    pub covariance: StateMat,   // Incertidumbre de Kalman
}

/// Detección pura que entra al tracker desde el modelo de IA (ej. YOLO)
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: [f64; 4], // [x1, y1, x2, y2]
    pub score: f64,
    pub feat: Vec<f64>, // Embeddings visuales (ReID) para calcular similitud coseno
}

// --- 2. CLASE PRINCIPAL: TRACK ----------------------------------------------------------------------
#[derive(Clone)]
pub struct Track {
    // --- Identidad básica ---
    pub track_id: usize,
    pub end_frame_id: usize, // Último frame donde se vió
    pub state: TrackState,

    // --- Datos de la última detección ---
    pub box_x1y1x2y2: [f64; 4],
    pub score: f64,
    pub feat: Vec<f64>, // Embeddings suavizados a lo largo del tiempo

    // --- Historial y Físicas ---
    pub delta_t: usize, // Frames hacia atrás para calcular la velocidad
    pub history: HashMap<usize, HistoryEntry>, // Estados pasados
    pub max_history: usize, // Límite de memoria

    // --- Filtro de Kalman ---
    pub kalman_filter: Option<KalmanFilter>, // Motor predictivo
    pub mean: Option<StateVec>,              // Predicción actual (posición y velocidad)
    pub covariance: Option<StateMat>,        // Incertidumbre actual
    pub velocity: [[f64; 2]; 4],             // Velocidad suavizada de las 4 esquinas de la caja

    // --- Lógica y Reglas de Negocio ---
    pub min_len: usize, // Frames para pasar de 'New' a 'Confirmed'
    pub is_dance_dataset: bool, // Flag para aplicar reglas especiales (ej. no deformar la caja)
}

impl Track {
    /// Inicializa un track temporal (Estado: New)
    pub fn new(box_x1y1x2y2: [f64; 4], score: f64, min_len: usize, is_dance_dataset: bool) -> Self {
        Self {
            track_id: 0, // Se le dará un ID real si se confirma en initiate()
            end_frame_id: 0,
            state: TrackState::New,
            box_x1y1x2y2,
            score,
            feat: Vec::new(),
            delta_t: 3,
            history: HashMap::new(),
            max_history: 30,
            kalman_filter: None,
            mean: None,
            covariance: None,
            velocity: [[0.0; 2]; 4],
            min_len,
            is_dance_dataset,
        }
    }

    // --- Control de Estados ---
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    pub fn mark_deleted(&mut self) {
        self.state = TrackState::Deleted;
    }

    /// Asigna un ID oficial y arranca el motor de Kalman para este objeto
    pub fn initiate(&mut self, frame_id: usize, counter: &mut TrackCounter) {
        self.track_id = counter.get_track_id();

        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.initiate(&self.cxcywh());

        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.kalman_filter = Some(kf);

        // Guardamos la foto inicial
        self.history.insert(
            frame_id,
            HistoryEntry {
                box_x1y1x2y2: self.box_x1y1x2y2,
                score: self.score,
                mean: self.mean.unwrap(),
                covariance: self.covariance.unwrap(),
            },
        );

        self.end_frame_id = frame_id;
        self.state = TrackState::New;
    }

    /// Proyecta la posición futura usando la inercia del Filtro de Kalman
    pub fn predict(&mut self) {
        // Truco específico: Si el objeto aún no es estable y los movimientos son erráticos (datasets de baile),
        // congelamos la velocidad de deformación (ancho/alto) para que la caja no crezca
        if let Some(ref mut mean) = self.mean {
            if self.state != TrackState::Confirmed && self.is_dance_dataset {
                mean[6] = 0.0; // Velocidad del ancho a 0
                mean[7] = 0.0; // Velocidad del alto a 0
            }
        }

        // Ejecuta la predicción matemática
        if let (Some(kf), Some(mean), Some(cov)) =
            (&self.kalman_filter, &self.mean, &self.covariance)
        {
            let (new_mean, new_cov) = kf.predict(mean, cov);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }
    }

    /// Corrige la predicción de Kalman usando una nueva detección real validada
    pub fn update(&mut self, frame_id: usize, det: &Detection) {
        // 1. Transformar caja al formato que entiende Kalman (Centro, Ancho, Alto)
        let cx = (det.bbox[0] + det.bbox[2]) / 2.0;
        let cy = (det.bbox[1] + det.bbox[3]) / 2.0;
        let w = det.bbox[2] - det.bbox[0];
        let h = det.bbox[3] - det.bbox[1];
        let det_cxcywh: [f64; 4] = [cx, cy, w, h];

        // 2. Corregir el estado interno de Kalman
        if let (Some(kf), Some(mean), Some(cov)) =
            (&mut self.kalman_filter, &self.mean, &self.covariance)
        {
            let (new_mean, new_cov) = kf.update(mean, cov, &det_cxcywh, det.score);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }

        // 3. Registrar el estado en el historial y eliminar lo viejo
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

            if self.history.len() > self.max_history {
                if let Some(&oldest_frame) = self.history.keys().min() {
                    self.history.remove(&oldest_frame);
                }
            }
        }

        // 4. Recalcular las velocidades de las esquinas usando una ventana de tiempo (delta_t)
        self.velocity = [[0.0; 2]; 4];
        for d_t in 1..=self.delta_t {
            // Buscamos la mejor caja previa en el historial
            let prev_box: [f64; 4] = super::utils::get_prev_box(&self.history, frame_id, d_t);
            let vels = get_vel(&prev_box, &det.bbox);

            for i in 0..4 {
                self.velocity[i][0] += vels[i][0] / d_t as f64;
                self.velocity[i][1] += vels[i][1] / d_t as f64;
            }
        }

        // Promediamos
        for i in 0..4 {
            self.velocity[i][0] /= self.delta_t as f64;
            self.velocity[i][1] /= self.delta_t as f64;
        }

        // 5. Actualizar los datos duros
        self.box_x1y1x2y2 = det.bbox;
        self.score = det.score;
        self.end_frame_id = frame_id;

        // Si ya sobrevivió suficientes frames pasa a Confirmaded
        self.state = if self.history.len() >= self.min_len {
            TrackState::Confirmed
        } else {
            TrackState::New
        };

        // 6. Actualización de ReID (Exponential Moving Average)
        // Mezclamos el vector visual anterior con el nuevo para mantener una apariencia fluida
        if self.feat.is_empty() {
            self.feat = det.feat.clone();
        } else if !det.feat.is_empty() {
            let alpha = 0.95_f64;
            // Damos más peso a la foto nueva si el score (confianza) es muy alto
            let beta = alpha + (1.0 - alpha) * (1.0 - det.score);
            let mut sum_sq = 0.0;

            for i in 0..self.feat.len() {
                self.feat[i] = beta * self.feat[i] + (1.0 - beta) * det.feat[i];
                sum_sq += self.feat[i] * self.feat[i];
            }

            // Normalización del vector resultante
            let norm = sum_sq.sqrt().max(1e-12);
            for i in 0..self.feat.len() {
                self.feat[i] /= norm;
            }
        }
    }

    // --- 3. EXPORTADORES DE COORDENADAS -------------------------------------------------------------
    /// Devuelve: [centro_x, centro_y, ancho, alto]
    pub fn cxcywh(&self) -> [f64; 4] {
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

    /// Devuelve: [x_arriba_izq, y_arriba_izq, ancho, alto]
    pub fn x1y1wh(&self) -> [f64; 4] {
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

    /// Devuelve: [x_arriba_izq, y_arriba_izq, x_abajo_der, y_abajo_der]
    pub fn x1y1x2y2(&self) -> [f64; 4] {
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
