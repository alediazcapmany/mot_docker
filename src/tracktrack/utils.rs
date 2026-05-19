use ndarray::Array2;
use std::collections::HashMap;

use crate::tracktrack::track::{Detection, HistoryEntry, Track};

// =========================================================================================
// 1. CÁLCULOS DE SUPERPOSICIÓN (BOUNDING BOXES)
// =========================================================================================

/// Calcula la matriz de similitud IoU (Intersection over Union) entre dos listas de cajas.
/// Compara cada caja de `a_boxes` contra todas las de `b_boxes`.
/// Formato esperado: [x1, y1, x2, y2]
pub fn bbox_overlaps(a_boxes: &[[f64; 4]], b_boxes: &[[f64; 4]]) -> Array2<f64> {
    let num_a = a_boxes.len();
    let num_b = b_boxes.len();
    let mut overlaps = Array2::<f64>::zeros((num_a, num_b));

    for n_b in 0..num_b {
        let b = b_boxes[n_b];
        let box_area = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);

        for n_a in 0..num_a {
            let a = a_boxes[n_a];

            // Ancho de la intersección. Si no se tocan, da negativo, por lo que limitamos a 0.0
            let iw = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
            if iw > 0.0 {
                let ih = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
                if ih > 0.0 {
                    // IoU = Área de Intersección / (Área A + Área B - Intersección)
                    let ua = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0) + box_area - iw * ih;
                    overlaps[[n_a, n_b]] = (iw * ih) / ua;
                }
            }
        }
    }
    overlaps
}

/// Rescata detecciones de alta confianza que el NMS de YOLO eliminó por error.
/// Devuelve los índices de las cajas en `dets_95` que no se solapan de forma crítica con `dets`.
pub fn find_deleted_detections(dets: &[[f64; 4]], dets_95: &[[f64; 4]]) -> Vec<usize> {
    let ious: Array2<f64> = bbox_overlaps(dets, dets_95);
    let mut deleted_indices = Vec::new();

    let num_a = dets.len();
    let num_b = dets_95.len();

    if num_a == 0 || num_b == 0 {
        return deleted_indices;
    }

    for j in 0..num_b {
        let mut max_iou = 0.0_f64;
        for i in 0..num_a {
            if ious[[i, j]] > max_iou {
                max_iou = ious[[i, j]];
            }
        }

        // Si su máximo solapamiento con las cajas supervivientes es < 0.97, la rescatamos
        if max_iou < 0.97 {
            deleted_indices.push(j);
        }
    }
    deleted_indices
}

// =========================================================================================
// 2. MATRICES DE COSTO (DISTANCIAS MATEMÁTICAS)
// =========================================================================================
// Nota: En los trackers, el "Costo" o "Distancia" es lo inverso a la "Similitud".
// A menor costo, más probable es que el track y la detección sean la misma persona.

/// Costo de posición (IoU Distancia). 
/// Combina el IoU normal con el H-IoU (Height IoU) para penalizar si las cajas cambian bruscamente de altura.
pub fn iou_distance(a_tracks: &[Track], b_dets: &[Detection]) -> (Array2<f64>, Array2<f64>) {
    // Usamos x1y1x2y2() para invocar la predicción del Kalman si está disponible
    let a_boxes: Vec<[f64; 4]> = a_tracks.iter().map(|t| t.x1y1x2y2()).collect();
    let b_boxes: Vec<[f64; 4]> = b_dets.iter().map(|d| d.bbox).collect();

    if a_boxes.is_empty() || b_boxes.is_empty() {
        return (
            Array2::<f64>::zeros((a_boxes.len(), b_boxes.len())),
            Array2::<f64>::ones((a_boxes.len(), b_boxes.len()))
        );
    }

    // HIoU: Similitud vertical (Muy útil en peatones ya que la altura varía menos que el ancho)
    let mut h_iou = Array2::<f64>::zeros((a_boxes.len(), b_boxes.len()));
    for (i, a) in a_boxes.iter().enumerate() {
        for (j, b) in b_boxes.iter().enumerate() {
            let ih_intersect = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let ih_union = (a[3].max(b[3]) - a[1].min(b[1])).max(0.0);
            if ih_union > 0.0 {
                h_iou[[i, j]] = ih_intersect / ih_union;
            }
        }
    }

    // IoU Normal (Similitud)
    let iou_sim = bbox_overlaps(&a_boxes, &b_boxes);
    let mut iou_dist = Array2::<f64>::zeros((a_boxes.len(), b_boxes.len()));
    
    // Distancia final = 1 - (Similitud_H * Similitud_IoU)
    for i in 0..a_boxes.len() {
        for j in 0..b_boxes.len() {
            iou_dist[[i, j]] = 1.0 - (h_iou[[i, j]] * iou_sim[[i, j]]);
        }
    }
    
    (iou_sim, iou_dist)
}

/// Costo de apariencia visual (FastReID). Calcula la distancia Coseno entre los embeddings.
pub fn cos_distance(tracks: &[Track], dets: &[Detection]) -> Array2<f64> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut cos_dist = Array2::<f64>::zeros((num_t, num_d));
    for i in 0..num_t {
        for j in 0..num_d {
            // Producto punto de los vectores normalizados
            let dot: f64 = tracks[i]
                .feat
                .iter()
                .zip(dets[j].feat.iter())
                .map(|(a, b)| a * b)
                .sum();

            cos_dist[[i, j]] = (1.0_f64 - dot).clamp(0.0_f64, 1.0_f64);
        }
    }
    cos_dist
}

/// Costo de confianza (Score). 
/// Penaliza a una detección si su nivel de confianza es muy diferente a la confianza histórica del track.
pub fn conf_distance(tracks: &[Track], dets: &[Detection]) -> Array2<f64> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut conf_dist = Array2::<f64>::zeros((num_t, num_d));
    for i in 0..num_t {
        let t = &tracks[i];

        // Rescatamos el score del último frame registrado
        let mut frame_ids: Vec<&usize> = t.history.keys().collect();
        frame_ids.sort_unstable_by(|a, b| b.cmp(a)); 

        let prev_score = if frame_ids.is_empty() {
            t.score
        } else {
            let idx = 1.min(frame_ids.len() - 1);
            t.history.get(frame_ids[idx]).unwrap().score
        };

        // Proyectamos linealmente la tendencia del score
        let t_score_proj = t.score + (t.score - prev_score);

        for j in 0..num_d {
            conf_dist[[i, j]] = (t_score_proj - dets[j].score).abs();
        }
    }
    conf_dist
}

/// Extrae de forma segura la caja de un track ocurrida `dt` frames atrás.
pub fn get_prev_box(history: &HashMap<usize, HistoryEntry>, frame_id: usize, dt: usize) -> [f64; 4] {
    let target_key = frame_id.saturating_sub(dt); 
    if let Some(entry) = history.get(&target_key) {
        return entry.box_x1y1x2y2; 
    }
    // Fallback: Si no existe exactamente en (frame - dt), devolvemos la última conocida
    if let Some(&max_key) = history.keys().max() {
        return history.get(&max_key).unwrap().box_x1y1x2y2;
    }
    [0.0; 4]
}

/// Costo angular direccional. 
/// Evalúa si la detección candidata está físicamente en la misma trayectoria/dirección
/// hacia la que el track se estaba moviendo.
pub fn angle_distance(tracks: &[Track], dets: &[Detection], frame_id: usize, d_t: usize) -> Array2<f64> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut angle_dist = Array2::<f64>::zeros((num_t, num_d));

    for i in 0..num_t {
        let b_1 = get_prev_box(&tracks[i].history, frame_id, d_t);
        let vel_t = tracks[i].velocity; // Velocidad inercial de las 4 esquinas

        for j in 0..num_d {
            let b_2 = dets[j].bbox;

            let corners_1 = [[b_1[0], b_1[1]], [b_1[0], b_1[3]], [b_1[2], b_1[1]], [b_1[2], b_1[3]]];
            let corners_2 = [[b_2[0], b_2[1]], [b_2[0], b_2[3]], [b_2[2], b_2[1]], [b_2[2], b_2[3]]];

            let mut angle_sum = 0.0;

            for c in 0..4 {
                // Vector de movimiento real detectado
                let dx = corners_2[c][0] - corners_1[c][0];
                let dy = corners_2[c][1] - corners_1[c][1];
                let norm = (dx * dx + dy * dy).sqrt() + 1e-5;
                let vel_t_d_x = dx / norm;
                let vel_t_d_y = dy / norm;

                // Desviación angular respecto a la velocidad inercial
                let dot = vel_t[c][0] * vel_t_d_x + vel_t[c][1] * vel_t_d_y;
                let angle = dot.clamp(-1.0_f64, 1.0_f64).acos().abs() / std::f64::consts::PI;
                angle_sum += angle / 4.0; // Promedio de las 4 esquinas
            }

            // Ponderamos el costo por la confianza real de la detección (Menos penalización si estamos seguros de la detección)
            angle_dist[[i, j]] = angle_sum * dets[j].score;
        }
    }
    angle_dist
}

// =========================================================================================
// 3. MOTORES DE ASIGNACIÓN Y NMS (ALGORITMO HÚNGARO SIMPLIFICADO)
// =========================================================================================

/// Motor TPA (Track-Pairwise Association). 
/// Resuelve la asignación buscando coincidencias mutuas mínimas.
/// Solo se enlaza si Track A cree que Det B es la mejor opción Y SIMULTÁNEAMENTE Det B cree que Track A es su mejor opción.
pub fn associate(cost: &Array2<f64>, match_thr: f64) -> Vec<[usize; 2]> {
    let mut matches = Vec::new();

    if cost.nrows() > 0 && cost.ncols() > 0 {
        let mut min_ddx = vec![0; cost.nrows()]; // Mejor detección para cada track
        let mut min_tdx = vec![0; cost.ncols()]; // Mejor track para cada detección

        // argmin(axis=1): Para cada track, buscar el costo mínimo
        for (i, row) in cost.rows().into_iter().enumerate() {
            let mut min_val = f64::MAX;
            for (j, &val) in row.into_iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_ddx[i] = j;
                }
            }
        }

        // argmin(axis=0): Para cada detección, buscar el costo mínimo
        for (j, col) in cost.columns().into_iter().enumerate() {
            let mut min_val = f64::MAX;
            for (i, &val) in col.into_iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_tdx[j] = i;
                }
            }
        }

        // Validación Mutua: Solo emparejamos si el "amor es correspondido" y el costo no supera el límite
        for (tdx, &ddx) in min_ddx.iter().enumerate() {
            if min_tdx[ddx] == tdx && cost[[tdx, ddx]] < match_thr {
                matches.push([tdx, ddx]);
            }
        }
    }
    matches
}

/// Algoritmo Core: Asignación Iterativa.
/// Calcula una matriz de costo global combinando todas las métricas, y luego extrae 
/// parejas reduciendo progresivamente la exigencia (el umbral) iteración a iteración.
pub fn iterative_assignment(
    tracks: &[Track],
    dets_high: &[Detection],
    dets_low: &[Detection],
    dets_del_high: &[Detection],
    mut match_thr: f64, 
    penalty_p: f64,     
    penalty_q: f64,     
    reduce_step: f64,   
    frame_id: usize,
    d_t: usize,
) -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
    let mut matches = Vec::new();

    // Consolida todas las detecciones candidatas para crear la matriz global
    let mut dets = Vec::new();
    dets.extend_from_slice(dets_high);
    dets.extend_from_slice(dets_low);
    dets.extend_from_slice(dets_del_high);

    // 1. Calcular todos los componentes del costo
    let (iou_sim, iou_dist) = iou_distance(tracks, &dets);
    let cos_dist = cos_distance(tracks, &dets);
    let conf_dist = conf_distance(tracks, &dets);
    let ang_dist = angle_distance(tracks, &dets, frame_id, d_t);

    let num_t = tracks.len();
    let num_d = dets.len();
    let mut cost = Array2::<f64>::zeros((num_t, num_d));

    // 2. Fusión de métricas y penalizaciones
    for i in 0..num_t {
        for j in 0..num_d {
            // Peso balanceado para favorecer IoU y Apariencia
            let mut c = 0.50 * iou_dist[[i, j]]
                + 0.50 * cos_dist[[i, j]]
                + 0.10 * conf_dist[[i, j]]
                + 0.05 * ang_dist[[i, j]];

            // Penalizamos (aumentamos el costo) a las detecciones de menor calidad original
            if j >= dets_high.len() && j < dets_high.len() + dets_low.len() {
                c += penalty_p; 
            } else if j >= dets_high.len() + dets_low.len() {
                c += penalty_q; 
            }

            // Hard Gate: Si físicamente no se tocan (IoU < 10%), imposibilitamos el match
            if iou_sim[[i, j]] <= 0.10 {
                c = 1.0;
            }

            cost[[i, j]] = c.clamp(0.0_f64, 1.0_f64);
        }
    }

    // 3. Extracción TPA Iterativa: Vamos relajando el límite (match_thr) progresivamente
    loop {
        let matches_ = associate(&cost, match_thr);
        match_thr -= reduce_step;

        if matches_.is_empty() {
            break; 
        }

        for &m in &matches_ {
            matches.push(m);
            let tdx = m[0];
            let ddx = m[1];

            // Bloqueamos la fila y columna del match asignándoles un costo inviable
            // para que no vuelvan a participar en la siguiente iteración
            for j in 0..cost.ncols() {
                cost[[tdx, j]] = 1.0;
            }
            for i in 0..cost.nrows() {
                cost[[i, ddx]] = 1.0;
            }
        }
    }

    // 4. Informe final: Quién se ha quedado sin asignar
    let mut u_tracks = Vec::new();
    let mut u_dets = Vec::new();

    let matched_tracks: Vec<usize> = matches.iter().map(|m| m[0]).collect(); 
    let matched_dets: Vec<usize> = matches.iter().map(|m| m[1]).collect(); 

    for i in 0..num_t {
        if !matched_tracks.contains(&i) {
            u_tracks.push(i);
        }
    }
    for j in 0..num_d {
        if !matched_dets.contains(&j) {
            u_dets.push(j);
        }
    }

    (matches, u_tracks, u_dets)
}

/// Track-Aware Non-Maximum Suppression (TA-NMS)
/// Purga nuevas detecciones asegurándose de que no sean "fantasmas" o sombras de un Track que ya estamos siguiendo.
pub fn track_aware_nms(
    pair_sims: &Array2<f64>, // Matriz IoU Track/Det
    scores: &[f64],          
    num_tracks: usize,       
    nms_thresh: f64,         
    score_thresh: f64,       
) -> Vec<bool> {
    let num_dets = pair_sims.nrows() - num_tracks;
    let mut allow_indices = vec![false; num_dets];

    // Criba Nivel 1: Nivel mínimo de IA requerido
    for idx in 0..num_dets {
        if scores[idx] > score_thresh {
            allow_indices[idx] = true;
        }
    }

    for idx in 0..num_dets {
        if !allow_indices[idx] {
            continue;
        }

        // Criba Nivel 2: ¿La detección es en realidad un Track conocido?
        if num_tracks > 0 {
            let mut max_sim_with_track = 0.0_f64;
            for t in 0..num_tracks {
                let sim = pair_sims[[num_tracks + idx, t]];
                if sim > max_sim_with_track {
                    max_sim_with_track = sim;
                }
            }

            // Si está literalmente encima de un Track vivo, lo destruimos asumiendo redundancia de YOLO
            if max_sim_with_track > nms_thresh {
                allow_indices[idx] = false;
                continue;
            }
        }

        // Criba Nivel 3: NMS Tradicional (Si dos cajas nuevas se pisan, sobrevive la del Score más alto)
        for jdx in 0..num_dets {
            if idx != jdx && allow_indices[jdx] && scores[idx] > scores[jdx] {
                if pair_sims[[num_tracks + idx, num_tracks + jdx]] > nms_thresh {
                    allow_indices[jdx] = false;
                }
            }
        }
    }
    allow_indices
}