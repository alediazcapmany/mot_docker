use super::cmc::Cmc;
use super::track::{Detection, Track, TrackCounter, TrackState};
use super::utils;
use ndarray::Array1;

// --- CONFIGURACIÓN -------------------------------------------------------------------

/// Hiperparámetros que dictan el comportamiento del motor de tracking
#[derive(Clone, Debug)]
pub struct Args {
    pub max_time_lost: usize, // Límite de frames para recordar a un objeto perdido
    pub det_thr: f64,         // Umbral para clasificar detecciones en alta/baja confianza
    pub match_thr: f64,       // Límite mínimo de similitud (IoU) para enlazar cajas
    pub penalty_p: f64,       // Penalización en el coste si la detección es de baja confianza
    pub penalty_q: f64,       // Penalización si la detección había sido descartada por NMS
    pub reduce_step: f64,     // Paso de reducción del umbral en la asignación iterativa
    pub init_thr: f64,        // Score mínimo exigido para atreverse a iniciar un nuevo track
    pub tai_thr: f64,         // Track-Aware NMS: Límite de solapamiento para ignorar ruido cerca de tracks vivos
    pub min_len: usize,       // Frames seguidos requeridos para graduar un track de 'New' a 'Confirmed'
}

// --- CLASE PRINCIPAL: TRACKER --------------------------------------------------------
pub struct Tracker {
    pub args: Args,            
    pub max_time_lost: usize,  
    pub tracks: Vec<Track>,    // Memoria principal: Lista de todos los objetos trackeados
    pub frame_id: usize,       // Reloj interno del tracker
    pub counter: TrackCounter, // Dispensador de IDs únicos
    pub cmc: Cmc,              // Módulo de Compensación de Movimiento de Cámara (estabiliza el fondo)
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

    // --- INICIALIZACIÓN DE TRACKS ----------------------------------------------------

    /// Crea nuevos tracks a partir de detecciones que se han quedado "huérfanas" tras la fase de asociación
    pub fn init_tracks(&mut self, dets: Vec<Detection>) {
        if dets.is_empty() {
            return;
        }

        // 1. Recopilamos los tracks que están activos (Confirmados o Nuevos)
        let alive_tracks: Vec<&Track> = self
            .tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed || t.state == TrackState::New)
            .collect();

        // 2. Juntamos las cajas de los tracks vivos y las nuevas detecciones para compararlas
        let mut all_boxes: Vec<[f64; 4]> = alive_tracks.iter().map(|t| t.x1y1x2y2()).collect();
        let det_boxes: Vec<[f64; 4]> = dets.iter().map(|d| d.bbox).collect();
        all_boxes.extend(&det_boxes);

        // 3. Calculamos cuánto se solapan (IoU) para aplicar Track-Aware NMS
        let iou_sim = utils::bbox_overlaps(&all_boxes, &all_boxes);
        let scores = Array1::from_iter(dets.iter().map(|d| d.score));

        // 4. Track-Aware NMS: Filtramos las detecciones. 
        // Si una detección nueva solapa demasiado con un track que ya existe, la ignoramos (suele ser ruido o doble caja).
        let allow_indices = utils::track_aware_nms(
            &iou_sim,                   
            scores.as_slice().unwrap(), 
            alive_tracks.len(),         // Punto de corte en la matriz para separar tracks vs detecciones
            self.args.tai_thr,          
            self.args.init_thr,         
        );

        // 5. Los que sobreviven al filtro se convierten en tracks oficiales en estado 'New'
        for (idx, &flag) in allow_indices.iter().enumerate() {
            if flag {
                let mut new_track =
                    Track::new(dets[idx].bbox, dets[idx].score, self.args.min_len, false);
                new_track.feat = dets[idx].feat.clone();
                new_track.initiate(self.frame_id, &mut self.counter);
                self.tracks.push(new_track);
            }
        }
    }

    // --- BUCLE PRINCIPAL DE ACTUALIZACIÓN --------------------------------------------

    /// El corazón del tracker. Recibe detecciones del frame actual y las enlaza con la memoria (tracks)
    pub fn update(&mut self, dets: Vec<Detection>, dets_95: Vec<Detection>) -> Vec<Track> {
        self.frame_id += 1;

        let dets_boxes: Vec<[f64; 4]> = dets.iter().map(|d| d.bbox).collect();
        let dets_95_boxes: Vec<[f64; 4]> = dets_95.iter().map(|d| d.bbox).collect();

        // 1. Rescatamos detecciones que NMS borró pero que podrían ser útiles si cruzamos con detecciones previas
        let dets_del_indices = utils::find_deleted_detections(&dets_boxes, &dets_95_boxes);
        let mut dets_del = Vec::new();
        for idx in dets_del_indices {
            dets_del.push(dets_95[idx].clone());
        }

        // 2. Clasificamos las detecciones según la confianza (Alta vs Baja)
        let mut dets_high = Vec::new();
        let mut dets_low = Vec::new();
        for d in dets {
            if d.score > self.args.det_thr {
                dets_high.push(d);
            } else {
                dets_low.push(d);
            }
        }

        let mut dets_del_high = Vec::new();
        for d in dets_del {
            if d.score > self.args.det_thr {
                dets_del_high.push(d);
            }
        }

        // 3. Organizamos nuestra memoria de tracks. 
        // Separamos los maduros (Confirmed/Lost) de los novatos (New).
        let mut tracked_lost_conf = Vec::new();
        let mut new_tracks = Vec::new();
        
        let current_tracks = std::mem::take(&mut self.tracks);
        for t in current_tracks {
            if t.state == TrackState::Confirmed || t.state == TrackState::Lost {
                tracked_lost_conf.push(t);
            } else if t.state == TrackState::New {
                new_tracks.push(t);
            }
        }

        // 4. CMC (Camera Motion Compensation): Alineamos el mundo pasado al frame actual si la cámara se movió
        if let Some(warp_matrix) = self.cmc.get_warp_matrix() {
            crate::tracktrack::cmc::apply_cmc(&mut tracked_lost_conf, &warp_matrix);
            crate::tracktrack::cmc::apply_cmc(&mut new_tracks, &warp_matrix);
        }

        // 5. Predicción Matemática: Kalman estima dónde deberían estar todos ahora mismo
        for t in tracked_lost_conf.iter_mut() {
            t.predict();
        }
        for t in new_tracks.iter_mut() {
            t.predict();
        }

        // --- ASIGNACIÓN 1: Emparejar Tracks Maduros ---
        let mut all_dets = Vec::new();
        all_dets.extend(dets_high.clone());
        all_dets.extend(dets_low.clone());
        all_dets.extend(dets_del_high.clone());

        // El algoritmo Húngaro (asignación iterativa) busca quién es quién basándose en IoU y scores
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

        // Actualizamos los tracks que encontraron pareja
        for m in matches {
            let t = m[0];
            let d = m[1];
            tracked_lost_conf[t].update(self.frame_id, &all_dets[d]);
        }

        // Los que no encontraron pareja pasan a estado "Perdido"
        for t in u_tracks {
            tracked_lost_conf[t].mark_lost();
        }

        // --- ASIGNACIÓN 2: Emparejar Tracks Novatos (New) ---
        // Extraemos las detecciones de alta confianza que sobraron de la Asignación 1
        let mut dets_high_left = Vec::new();
        for &d in &u_dets {
            if d < dets_high.len() {
                dets_high_left.push(all_dets[d].clone());
            }
        }

        let empty_dets: Vec<Detection> = Vec::new();
        let (matches_new, u_tracks_new, u_dets_new) = utils::iterative_assignment(
            &new_tracks,
            &dets_high_left,
            &empty_dets, // Aquí no usamos detecciones bajas ni eliminadas
            &empty_dets,
            self.args.match_thr,
            self.args.penalty_p,
            self.args.penalty_q,
            self.args.reduce_step,
            self.frame_id,
            3,
        );

        for m in matches_new {
            let t = m[0];
            let d = m[1];
            new_tracks[t].update(self.frame_id, &dets_high_left[d]);
            new_tracks[t].feat = dets_high_left[d].feat.clone();
        }

        // Si un track novato no encuentra pareja rápido, lo eliminamos directamente (es ruido casi seguro)
        for t in u_tracks_new {
            new_tracks[t].mark_deleted();
        }

        // --- FASE DE LIMPIEZA Y CIERRE ---
        // Devolvemos todos los tracks a la lista unificada
        self.tracks.extend(tracked_lost_conf);
        self.tracks.extend(new_tracks);

        // Purgamos los tracks perdidos que llevan demasiado tiempo sin aparecer
        for track in self.tracks.iter_mut() {
            if self.frame_id.saturating_sub(track.end_frame_id) > self.max_time_lost {
                track.mark_deleted();
            }
        }
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // Las detecciones buenas que sobraron del todo inician tracks nuevos
        let mut unmatched_dets = Vec::new();
        for &udx in &u_dets_new {
            unmatched_dets.push(dets_high_left[udx].clone());
        }
        self.init_tracks(unmatched_dets);

        // Solo exportamos los confirmados para no meter basura en el log
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .cloned()
            .collect()
    }

    // --- ACTUALIZACIÓN A CIEGAS ------------------------------------------------------

    /// Se ejecuta en los frames donde no pasamos la IA (YOLO). Solo proyecta inercia.
    pub fn update_without_detections(&mut self) -> Vec<Track> {
        self.frame_id += 1;

        // Limpiamos los "New" directamente porque sin IA no podemos confirmarlos
        self.tracks.retain(|t| t.state != TrackState::New);

        // Aplicamos estabilización de cámara si existe
        if let Some(warp_matrix) = self.cmc.get_warp_matrix() {
            crate::tracktrack::cmc::apply_cmc(&mut self.tracks, &warp_matrix);
        }

        // Proyectamos a todos a ciegas con Kalman y los marcamos como perdidos momentáneamente
        for t in self.tracks.iter_mut() {
            t.predict();
            t.mark_lost();

            // Purgado por tiempo
            if self.frame_id.saturating_sub(t.end_frame_id) > self.max_time_lost {
                t.mark_deleted();
            }
        }

        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // Devolvemos una lista vacía para ser coherentes con la firma de Python (los tracks ciegos no se pintan)
        Vec::new()
    }
}