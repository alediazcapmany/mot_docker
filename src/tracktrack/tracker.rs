use super::cmc::Cmc;
use super::track::{Detection, Track, TrackCounter, TrackState};
use super::utils;
use ndarray::Array1;

// --- DEFINICIONES DE TIPOS -----------------------------------------------------------

// Estructura para los argumentos de configuración del tracker
#[derive(Clone, Debug)]
pub struct Args {
    pub max_time_lost: usize,
    pub det_thr: f32,
    pub match_thr: f32,
    pub penalty_p: f32,
    pub penalty_q: f32,
    pub reduce_step: f32,
    pub init_thr: f32,
    pub tai_thr: f32, // Track Aware NMS threshold
}

// --- TRACKER -------------------------------------------------------------------------
pub struct Tracker {
    pub args: Args,            // Argumentos de configuración del tracker
    pub max_time_lost: usize,  // Frames que track perdido sobrevive
    pub tracks: Vec<Track>,    // Lista de tracks activos
    pub frame_id: usize,       // ID del frame actual
    pub counter: TrackCounter, // Contador de IDs
    pub cmc: Cmc,              // Compensación de movimiento de cámara
}

impl Tracker {
    pub fn new(args: Args, vidname: &str) -> Self {
        let cmc = Cmc::new(vidname);
        let max_time_lost = args.max_time_lost;
        Self {
            args,
            max_time_lost,
            tracks: Vec::new(),
            frame_id: 0,
            counter: TrackCounter::new(),
            cmc,
        }
    }

    // Inicia nuevos tracks a partir de detecciones huérfanas (que no se asignaron a ningún track)
    pub fn init_tracks(&mut self, dets: Vec<Detection>) {
        if dets.is_empty() {
            // Si no hay detecciones, no hay huerfanas
            return;
        }

        // Obtener tracks vivos (Tracked / Confirmed o New)
        let alive_tracks: Vec<&Track> = self
            .tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed || t.state == TrackState::New)
            .collect();

        // Extraer las cajas para calcular la similitud
        let mut all_boxes: Vec<[f32; 4]> = alive_tracks.iter().map(|t| t.x1y1x2y2()).collect();
        let det_boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
        all_boxes.extend(&det_boxes);

        // Matriz de similitud IoU
        let iou_sim = utils::bbox_overlaps(&all_boxes, &all_boxes);
        let scores = Array1::from_iter(dets.iter().map(|d| d.score));

        // NMS Consciente de Tracks
        // (Track Aware NMS): Solo iniciamos tracks de detecciones que no estén muy cerca de los tracks vivos
        let allow_indices = utils::track_aware_nms(
            &iou_sim,                   // Matriz de similitud IoU entre tracks vivos y detecciones
            scores.as_slice().unwrap(), // Scores de las detecciones
            alive_tracks.len(), // Número de tracks vivos (para separar la matriz de similitud)
            self.args.tai_thr,  // Umbral de NMS consciente de tracks
            self.args.init_thr, // Umbral de score para iniciar tracks
        );

        // Iniciamos los que sobreviven
        for (idx, &flag) in allow_indices.iter().enumerate() {
            if flag {
                let mut new_track = Track::new(dets[idx].bbox, dets[idx].score, 3, false);
                new_track.feat = dets[idx].feat.clone();
                new_track.initiate(self.frame_id, &mut self.counter);
                self.tracks.push(new_track);
            }
        }
    }

    pub fn update(&mut self, dets: Vec<Detection>, dets_95: Vec<Detection>) -> Vec<Track> {
        // ==============================================================================================================
        self.frame_id += 1;

        // Extraemos las cajas puras para la función matemática
        let dets_boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
        let dets_95_boxes: Vec<[f32; 4]> = dets_95.iter().map(|d| d.bbox).collect();

        // Recuperar detections eliminadas
        let dets_del_indices = utils::find_deleted_detections(&dets_boxes, &dets_95_boxes);
        let mut dets_del = Vec::new();
        for idx in dets_del_indices {
            dets_del.push(dets_95[idx].clone());
        }

        // Divide detections en alta y baja confianza según el umbral de detección
        let mut dets_high = Vec::new();
        let mut dets_low = Vec::new();
        for d in dets {
            if d.score > self.args.det_thr {
                dets_high.push(d);
            } else {
                dets_low.push(d);
            }
        }

        // Divide detections eliminadas en alta confianza según el umbral de detección
        let mut dets_del_high = Vec::new();
        for d in dets_del {
            if d.score > self.args.det_thr {
                dets_del_high.push(d);
            }
        }

        // Separar tracks en "lost" y "new" para aplicar CMC solo a los primeros
        let mut tracked_lost_conf = Vec::new();  // Confirmed o Lost
        let mut new_tracks = Vec::new(); 
        // let mut other_tracks = Vec::new(); // Para los que ya estaban borrados

        // Recorremos los tracks actuales y los separamos según su estado
        let current_tracks = std::mem::take(&mut self.tracks);
        for t in current_tracks {
            if t.state == TrackState::Confirmed || t.state == TrackState::Lost {
                tracked_lost_conf.push(t);
            } else if t.state == TrackState::New {
                new_tracks.push(t);
            } 
        }

        // CMC a los tracks Confirmed y Lost, no a los New (porque no tienen Kalman)
        if let Some(warp_matrix) = self.cmc.get_warp_matrix() {
            crate::tracktrack::cmc::apply_cmc(&mut tracked_lost_conf, &warp_matrix);
            crate::tracktrack::cmc::apply_cmc(&mut new_tracks, &warp_matrix);
        }

        // Predict de todos los tracks (los que van a ser asociados y los nuevos)
        for t in tracked_lost_conf.iter_mut() {
            t.predict();
        }
        for t in new_tracks.iter_mut() {
            t.predict();
        }

        // ==============================================================================================================
        // Association 1: (tracked and lost tracks) & (high confidence detections)
        let mut all_dets = Vec::new();
        all_dets.extend(dets_high.clone());
        all_dets.extend(dets_low.clone());
        all_dets.extend(dets_del_high.clone());

        let (matches, u_tracks, u_dets) = utils::iterative_assignment(
            &tracked_lost_conf,
            &dets_high,
            &dets_low,
            &dets_del_high,
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            3,
        );

        // Update matched tracks
        for m in matches {
            let t = m[0];
            let d = m[1];
            // Actualizamos caja, score y feature (si lo necesitas)
            tracked_lost_conf[t].update(self.frame_id, all_dets[d].bbox, all_dets[d].score);
            tracked_lost_conf[t].feat = all_dets[d].feat.clone(); // Reemplaza update_features de Python
        }

        // Mark "lost" to unmatched tracks
        for t in u_tracks {
            tracked_lost_conf[t].mark_lost();
        }

        // ==============================================================================================================
        // Get remained high confidence detections
        let mut dets_high_left = Vec::new();
        for &d in &u_dets {
            if d < dets_high.len() {
                dets_high_left.push(all_dets[d].clone());
            }
        }

        // Association 2: (new tracks) & (left high confidence detections)
        let empty_dets: Vec<Detection> = Vec::new();
        let (matches_new, u_tracks_new, u_dets_new) = utils::iterative_assignment(
            &new_tracks,
            &dets_high_left,
            &empty_dets, // Pasamos listas vacías como en Python: [], []
            &empty_dets,
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            3,
        );

        // Update matched tracks
        for m in matches_new {
            let t = m[0];
            let d = m[1];
            new_tracks[t].update(
                self.frame_id,
                dets_high_left[d].bbox,
                dets_high_left[d].score,
            );
            new_tracks[t].feat = dets_high_left[d].feat.clone();
        }

        // Mark "remove" to unmatched tracks
        for t in u_tracks_new {
            new_tracks[t].mark_deleted(); // Deleted es tu equivalente a Removed
        }

        // ==============================================================================================================
        // Juntamos todos los tracks de vuelta a la clase principal
        self.tracks.extend(tracked_lost_conf);
        self.tracks.extend(new_tracks);
        // self.tracks.extend(other_tracks);

        // Mark "remove" lost tracks which are too old
        for track in self.tracks.iter_mut() {
            if self.frame_id.saturating_sub(track.end_frame_id) > self.max_time_lost {
                track.mark_deleted();
            }
        }

        // Filter out the removed tracks
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // Init new tracks con las detecciones huérfanas
        let mut unmatched_dets = Vec::new();
        for &udx in &u_dets_new {
            unmatched_dets.push(dets_high_left[udx].clone());
        }
        self.init_tracks(unmatched_dets);

        // Retornamos una copia de los tracks activos (como hace el Python)
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .cloned()
            .collect()
    }

    pub fn update_without_detections(&mut self) -> Vec<Track> {
        // Update frame id
        self.frame_id += 1;

        // Solo mantenemos Confirmed o Lost, tiramos los New
        // self.tracks.retain(|t| t.state != TrackState::New);

        // Camera motion compensation
        if let Some(warp_matrix) = self.cmc.get_warp_matrix() {
            crate::tracktrack::cmc::apply_cmc(&mut self.tracks, &warp_matrix);
        }

        // Change every track as lost tracks and predict
        for t in self.tracks.iter_mut() {
            t.predict();
            t.mark_lost();

            // Si llevan perdidos demasiado tiempo, se marcan para borrar
            if self.frame_id.saturating_sub(t.end_frame_id) > self.max_time_lost {
                t.mark_deleted();
            }
        }

        // Filter out the removed tracks
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // Return empty list como en Python
        Vec::new()
    }
}
