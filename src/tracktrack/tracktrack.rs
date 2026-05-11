
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    video::KalmanFilter,
    videoio, Result,
};

use nalgebra::{DMatrix, SVector};

// ── 1. ESTADO DEL TRACK ──────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum TrackState {
    Posible,   // age < 3 frames confirmados
    Confirmed, // age >= 3 frames confirmados
    Lost,      // No match en este frame, guardamos por si reaparece
    Deleted,   // Demasiado perdido, se borra de la lista
}

// ── 2. INFO DEL TRACK (PERSONA) ─────────────────────────────────────────────────
pub type StateVec = SVector<f32, 8>;
pub type MeasureVec = SVector<f32, 4>;

pub type StateMat = SMatrix<f32, 8, 8>;
pub type MeasureMat = SMatrix<f32, 4, 8>;

pub struct Track {
    // Todo lo que quiero que guarde cada track de persona que estoy siguiendo
    pub track_id: usize,
    pub state: TrackState,

    // Estado Kalman completo
    pub mean: StateVec,

    // Covarianza
    pub covariance: StateMat,

    // Contadores para la lógica de TrackTrack
    pub age: usize, // Total de frames que lleva existiendo
    pub time_since_update: usize, // Frames seguidos sin hacer match (si llega a 30, state = Deleted)

                                  // HUECO PARA EL FUTURO: Aquí guardaremos el vector de FastReID
                                  // pub embedding: Option<Vec<f32>>,
}

impl Track {
    // Implemento el track, con su construcción y funciones (metodos) para actualizarlo facilmente
    /// Crea nuevo Track con una detección huérfana
    pub fn new(id: usize, tlwh: [f32; 4]) -> Self {
        // Convertimos la caja de formato tlwh a cx, cy, a, h para el KF
        let (cx, cy, a, h) = tlwh_to_xyah(tlwh);

        // Inicializamos el estado del KF con la posición de la caja y velocidades a 0
        let mut mean = StateVec::zeros();

        // mean = [cx, cy, a, h, 0, 0, 0, 0]
        mean[0] = cx;
        mean[1] = cy;
        mean[2] = a;
        mean[3] = h;

        // La covarianza inicial puede ser una matriz diagonal con valores grandes (incertidumbre alta)
        let covariance = StateMat::identity() * 10.0;

        Self {
            track_id: id,
            state: TrackState::Posible,
            mean,
            covariance,
            age: 1,
            time_since_update: 0,
        } // Devuelve nuevo track con: ID, estado Posible, estado KF inicializado, edad 1 y sin tiempo perdido
    }

    /// Predice la nueva posición del track usando el filtro de Kalman (modelo de movimiento constante) y actualiza su estado interno
    pub fn predict(&mut self) {
        let dt = 1.0; // Suponemos que el tiempo entre frames es constante (1 segundo para simplificar)

        let mut motion_mat = StateMat::identity(); // Matriz de transición de estado para movimiento constante

        motion_mat[(0, 4)] = dt;
        motion_mat[(1, 5)] = dt;
        motion_mat[(2, 6)] = dt;
        motion_mat[(3, 7)] = dt;

        // Ruido de proceso (incertidumbre en el modelo de movimiento)
        let process_noise = StateMat::identity() * 1e-2;

        // Predice el nuevo estado multiplicando la matriz de movimiento por el estado actual
        self.mean = motion_mat * self.mean;

        // Actualiza la covarianza con la fórmula de predicción del KF: P' = F * P * F^T + Q
        self.covariance = motion_mat * self.covariance * motion_mat.transpose() + process_noise;
    }

    /// Actualiza el track con una nueva detección (posición) usando la fórmula de actualización del filtro de Kalman
    pub fn update(&mut self, tlwh: [f32; 4]) {
        let (cx, cy, a, h) = tlwh_to_xyah(tlwh);    // Convertimos la caja de formato tlwh a cx, cy, a, h para el KF

        let z = MeasureVec::from_row_slice(&[cx, cy, a, h]);

        let mut update_mat = MeasureMat::zeros();

        update_mat[(0, 0)] = 1.0;
        update_mat[(1, 1)] = 1.0;
        update_mat[(2, 2)] = 1.0;
        update_mat[(3, 3)] = 1.0;

        let measurement_noise = nalgebra::SMatrix::<f32, 4, 4>::identity() * 1e-1;

        // Innovation
        let y = z - update_mat * self.mean;

        // Innovation covariance
        let s = update_mat * self.covariance * update_mat.transpose() + measurement_noise;

        // Kalman gain
        let k = self.covariance * update_mat.transpose() * s.try_inverse().unwrap();

        // Update mean
        self.mean = self.mean + k * y;

        // Update covariance
        let i = StateMat::identity();

        self.covariance = (i - k * update_mat) * self.covariance;

        self.time_since_update = 0;

        self.age += 1;
    }

    fn tlwh_to_xyah(tlwh: [f32; 4]) -> (f32, f32, f32, f32) {
        let x = tlwh[0];
        let y = tlwh[1];
        let w = tlwh[2];
        let h = tlwh[3];

        let cx = x + w / 2.0;
        let cy = y + h / 2.0;

        let aspect_ratio = w / h.max(1e-6);

        (cx, cy, aspect_ratio, h)
    }

    /// Actualiza track al hacer match con una detección (posición, estado y apariencia)
    pub fn mark_matched(&mut self, new_bbox: [f32; 4]) {
        self.time_since_update = 0; //Reinicia ultima detección
        self.age += 1; // Aumenta edad
        self.tlwh = new_bbox; // Actualiza pos caja

        // En función de age meter track en Confirmed o mantenerlo en Posible
        if self.state == TrackState::Posible && self.age >= 3 {
            self.state = TrackState::Confirmed;
        } else if self.state == TrackState::Lost {
            self.state = TrackState::Confirmed;
        }

        // (Aquí actualizaríamos el embedding combinando el viejo con el nuevo)
    }

    /// Actualiza track cuando NO encuentra pareja
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost; // Pasa a estado perdido
        self.time_since_update += 1; // Aumenta frame sin detección
    }
}

// ── 3. TRACK TRACK ───────────────────────────────────────────────
pub struct TrackTrack {
    // Listas que separan a las personas según su estado
    pub active_tracks: Vec<Track>, // Los que estamos viendo y siguiendo
    pub lost_tracks: Vec<Track>,   // Los que se ocultaron

    // Parámetros de configuración
    next_id: usize,       // Contador de IDs
    max_time_lost: usize, // Frames que se guarda track perdido
}

impl TrackTrack {
    pub fn new(max_time_lost: usize) -> Self {
        // Inicializa el tracker tracktrack
        Self {
            active_tracks: Vec::new(),
            lost_tracks: Vec::new(),
            next_id: 1,
            max_time_lost,
        }
    }

    /// ID para nuevo track
    pub fn next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// El bucle principal del tracker. Se ejecuta en cada frame.
    pub fn update(&mut self, detections: Vec<Rect>) -> Result<()> {
        // 1. PREDICCIÓN (Filtro de Kalman)
        // Pedimos a cada persona que prediga dónde estará en este frame
        for track in self.active_tracks.iter_mut() {
            track.predict()?; // Llamamos al método que hicimos antes
        }
        for track in self.lost_tracks.iter_mut() {
            track.predict()?;
        }

        let mut remaining_detections = detections;

        // 2. SEPARAR VIPs y NOVATOS (Posibles)
        let mut vips = Vec::new();
        let mut novices = Vec::new();

        // Vaciamos temporalmente la lista activa para repartirlos
        for track in self.active_tracks.drain(..) {
            if track.state == TrackState::Confirmed {
                vips.push(track);
            } else {
                novices.push(track); // Si es TrackState::Posible
            }
        }

        // Repartimos a los perdidos que estaban en la recámara
        for track in self.lost_tracks.drain(..) {
            // Usamos su estado para saber si eran VIPs antes de perderse
            if track.state == TrackState::Confirmed {
                vips.push(track);
            } else {
                novices.push(track);
            }
        }

        // CODIGO REVISADO HASTA AQUÍ

        // 3. RONDA 1: Asociación VIP (TPA)
        // Cruzamos los VIPs contra TODAS las detecciones
        let (matched_vips, unmatched_vips, leftovers_r1) =
            self.tpa_match(vips, &remaining_detections);

        remaining_detections = leftovers_r1;

        // 4. RONDA 2: Asociación Novatos (TPA)
        // Cruzamos los Posibles (novatos) SOLO contra las detecciones que sobraron
        let (matched_novices, unmatched_novices, leftovers_r2) =
            self.tpa_match(novices, &remaining_detections);

        remaining_detections = leftovers_r2;

        // 5. TAI: Track-Aware Initialization (El nacimiento)
        // Extraemos las cajas de los que sí hicieron match para usarlas como "Anclas"
        let mut anchors = Vec::new();
        anchors.extend(matched_vips.iter().map(|t| t.tlwh));
        anchors.extend(matched_novices.iter().map(|t| t.tlwh));

        // Metemos las anclas y las cajas huérfanas a la barredora (NMS)
        let new_tracks = self.apply_tai(&anchors, &remaining_detections)?;

        // 6. GESTIÓN DE ESTADOS Y LIMPIEZA
        // Reconstruimos la lista principal para el próximo frame
        self.active_tracks.clear();
        self.active_tracks.extend(matched_vips);
        self.active_tracks.extend(matched_novices);
        self.active_tracks.extend(new_tracks);

        // Los que no encontraron pareja (ni en Ronda 1 ni en 2), se marcan como perdidos
        for mut lost_track in unmatched_vips
            .into_iter()
            .chain(unmatched_novices.into_iter())
        {
            lost_track.mark_lost();
            self.lost_tracks.push(lost_track);
        }

        // Limpiar la memoria: borrar definitivamente los que llevan perdidos mucho tiempo
        self.lost_tracks
            .retain(|t| t.time_since_update < self.max_time_lost);

        Ok(())
    }

    // ─── STUBS DE LAS FUNCIONES MATEMÁTICAS ─────────────────────────────────────

    /// El algoritmo TPA: Calcula la matriz de costos y busca emparejamientos mutuos
    fn tpa_match(
        &self,
        tracks: Vec<Track>,
        detections: &Vec<Rect>,
    ) -> (Vec<Track>, Vec<Track>, Vec<Rect>) {
        // HUECO: Aquí irá el cálculo de la matriz de Intersección sobre Unión (IoU).

        // De momento, devolvemos todo sin emparejar para que el código compile
        (Vec::new(), tracks, detections.clone())
    }

    /// Filtro TAI: Usa NMS para evitar que nazcan tracks duplicados sobre personas existentes
    fn apply_tai(
        &mut self,
        anchors: &Vec<[f32; 4]>,
        orphan_detections: &Vec<Rect>,
    ) -> Result<Vec<Track>> {
        let mut new_tracks = Vec::new();

        // HUECO: Aquí llamaremos a dnn::nms_boxes

        /*
        // Bucle conceptual de creación de tracks:
        for det in orphans_supervivientes {
            let track = Track::new(self.next_id(), det_format_array)?;
            new_tracks.push(track);
        }
        */

        Ok(new_tracks)
    }
}
