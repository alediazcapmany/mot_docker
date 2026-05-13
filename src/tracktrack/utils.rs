use ndarray::Array2;
use std::collections::HashMap;

// IMPORTACIÓN CORREGIDA: Apuntamos a tus structs exactos
use crate::tracktrack::track::{Detection, HistoryEntry, Track};

/// Calcula la matriz de IoU entre dos listas de cajas: matriz de similitud entre tracks y detecciones devuelve
/// Las cajas deben estar en formato [x1, y1, x2, y2]
pub fn bbox_overlaps(a_boxes: &[[f32; 4]], b_boxes: &[[f32; 4]]) -> Array2<f32> {
    let num_a = a_boxes.len();
    let num_b = b_boxes.len();
    let mut overlaps = Array2::<f32>::zeros((num_a, num_b));

    for n_b in 0..num_b {
        let b = b_boxes[n_b]; // [x1, y1, x2, y2] de la caja b
        let box_area = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0); // Área de la caja b

        for n_a in 0..num_a {
            let a = a_boxes[n_a]; // [x1, y1, x2, y2] de la caja a

            let iw = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0); // Ancho de la intersección (si es negativo, no hay intersección, por eso max con 0)
            if iw > 0.0 {
                let ih = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0); // Alto de la intersección
                if ih > 0.0 {
                    // Si hay intersección, calculamos el área de la intersección y la unión
                    let ua = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0) + box_area - iw * ih;
                    overlaps[[n_a, n_b]] = (iw * ih) / ua; // IoU = Área de Intersección / Área de Unión
                }
            }
        }
    }
    overlaps // Devuelve la matriz de IoU entre cada caja de a_boxes y cada caja de b_boxes
}

/// Encuentra detecciones de alta confianza que fueron eliminadas (ej. por NMS)
/// Devuelve los índices de las cajas en `dets_95` que hay que rescatar.
pub fn find_deleted_detections(dets: &[[f32; 4]], dets_95: &[[f32; 4]]) -> Vec<usize> {
    // Usamos nuestra propia función matemática de arriba
    let ious: Array2<f32> = bbox_overlaps(dets, dets_95);
    let mut deleted_indices = Vec::new();

    let num_a = dets.len();
    let num_b = dets_95.len();

    // Si una de las listas está vacía, no hay nada que rescatar
    if num_a == 0 || num_b == 0 {
        return deleted_indices;
    }

    for j in 0..num_b {
        let mut max_iou = 0.0_f32;

        for i in 0..num_a {
            let current_iou = ious[[i, j]];
            if current_iou > max_iou {
                max_iou = current_iou;
            }
        }

        // Si su máximo solapamiento es menor a 0.97, la rescatamos
        if max_iou < 0.97 {
            deleted_indices.push(j);
        }
    }
    deleted_indices
}

pub fn iou_distance(a_tracks: &[Track], b_dets: &[Detection]) -> (Array2<f32>, Array2<f32>) {
    // AQUI ESTA LA MAGIA: Llamamos al método .x1y1x2y2() que aplica el Kalman si existe
    let a_boxes: Vec<[f32; 4]> = a_tracks.iter().map(|t| t.x1y1x2y2()).collect();
    let b_boxes: Vec<[f32; 4]> = b_dets.iter().map(|d| d.bbox).collect();

    // Calcular distancia IoU
    if a_boxes.is_empty() || b_boxes.is_empty() {
        // Si no hay tracks o detecciones, devolvemos una matriz vacía
        let iou_sim = Array2::<f32>::zeros((a_boxes.len(), b_boxes.len()));
        let iou_dist = Array2::<f32>::ones((a_boxes.len(), b_boxes.len()));
        return (iou_sim, iou_dist);
    } else {
        // Calcular HIoU
        let mut h_iou = Array2::<f32>::zeros((a_boxes.len(), b_boxes.len()));
        for (i, a) in a_boxes.iter().enumerate() {
            for (j, b) in b_boxes.iter().enumerate() {
                let iw = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
                let ih = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
                let denom = (a[3].max(b[3]) - a[1].min(b[1]) + 1.0).max(0.0);

                if ih > 0.0 && denom > 0.0 {
                    h_iou[[i, j]] = (iw * ih / (ih + 1.0)) / denom;
                }
            }
        }

        // Calcular HMIoU
        let iou_sim = bbox_overlaps(&a_boxes, &b_boxes);

        let mut iou_dist = Array2::<f32>::zeros((a_boxes.len(), b_boxes.len()));
        for i in 0..a_boxes.len() {
            for j in 0..b_boxes.len() {
                iou_dist[[i, j]] = 1.0 - (h_iou[[i, j]] * iou_sim[[i, j]]);
            }
        }
        return (iou_sim, iou_dist);
    }
}

// -----------------------------------------------------------------------------------------
// NUEVAS FUNCIONES FÍSICAS Y MATEMÁTICAS TRADUCIDAS DE PYTHON
// -----------------------------------------------------------------------------------------

pub fn cos_distance(tracks: &[Track], dets: &[Detection]) -> Array2<f32> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut cos_dist = Array2::<f32>::zeros((num_t, num_d));
    for i in 0..num_t {
        for j in 0..num_d {
            // Producto punto entre los features del track y la detección
            let dot: f32 = tracks[i]
                .feat
                .iter()
                .zip(dets[j].feat.iter())
                .map(|(a, b)| a * b)
                .sum();

            cos_dist[[i, j]] = (1.0_f32 - dot).clamp(0.0_f32, 1.0_f32);
        }
    }
    cos_dist
}

pub fn conf_distance(tracks: &[Track], dets: &[Detection]) -> Array2<f32> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut conf_dist = Array2::<f32>::zeros((num_t, num_d));
    for i in 0..num_t {
        let t = &tracks[i];

        // Obtener la puntuación anterior del historial
        let mut frame_ids: Vec<&usize> = t.history.keys().collect();
        frame_ids.sort_unstable_by(|a, b| b.cmp(a)); // Orden inverso (mayor a menor)

        let prev_score = if frame_ids.is_empty() {
            t.score
        } else {
            let idx = 1.min(frame_ids.len() - 1);
            // Usamos .score porque HistoryEntry es un struct
            t.history.get(frame_ids[idx]).unwrap().score
        };

        // Linear projection
        let t_score_proj = t.score + (t.score - prev_score);

        for j in 0..num_d {
            conf_dist[[i, j]] = (t_score_proj - dets[j].score).abs();
        }
    }
    conf_dist
}

// Actualizado para usar HistoryEntry
fn get_prev_box(history: &HashMap<usize, HistoryEntry>, frame_id: usize, dt: usize) -> [f32; 4] {
    let target_key = frame_id.saturating_sub(dt); // Evita error de underflow en Rust
    if let Some(entry) = history.get(&target_key) {
        return entry.box_x1y1x2y2; // Obtenemos el campo real del struct
    }
    // Si no hay observación reciente, devuelve la más actual que tenga
    if let Some(&max_key) = history.keys().max() {
        return history.get(&max_key).unwrap().box_x1y1x2y2;
    }
    [0.0; 4]
}

pub fn angle_distance(
    tracks: &[Track],
    dets: &[Detection],
    frame_id: usize,
    d_t: usize,
) -> Array2<f32> {
    let num_t = tracks.len();
    let num_d = dets.len();
    if num_t == 0 || num_d == 0 {
        return Array2::ones((num_t, num_d));
    }

    let mut angle_dist = Array2::<f32>::zeros((num_t, num_d));

    for i in 0..num_t {
        let b_1 = get_prev_box(&tracks[i].history, frame_id, d_t);
        // velocity se asume como [[f32; 2]; 4] (vel en las 4 esquinas)
        let vel_t = tracks[i].velocity;

        for j in 0..num_d {
            let b_2 = dets[j].bbox;

            // Extraemos las 4 esquinas de b_1 y b_2
            // 0: left-top, 1: left-bottom, 2: right-top, 3: right-bottom
            let corners_1 = [
                [b_1[0], b_1[1]],
                [b_1[0], b_1[3]],
                [b_1[2], b_1[1]],
                [b_1[2], b_1[3]],
            ];
            let corners_2 = [
                [b_2[0], b_2[1]],
                [b_2[0], b_2[3]],
                [b_2[2], b_2[1]],
                [b_2[2], b_2[3]],
            ];

            let mut angle_sum = 0.0;

            for c in 0..4 {
                // Cálculo de la velocidad normalizada hacia la detección
                let dx = corners_2[c][0] - corners_1[c][0];
                let dy = corners_2[c][1] - corners_1[c][1];
                let norm = (dx * dx + dy * dy).sqrt() + 1e-5;
                let vel_t_d_x = dx / norm;
                let vel_t_d_y = dy / norm;

                // Producto punto y ángulo (calc_angle)
                let dot = vel_t[c][0] * vel_t_d_x + vel_t[c][1] * vel_t_d_y;
                let angle = dot.clamp(-1.0_f32, 1.0_f32).acos().abs() / std::f32::consts::PI;
                angle_sum += angle / 4.0;
            }

            // Fuse score
            angle_dist[[i, j]] = angle_sum * dets[j].score;
        }
    }
    angle_dist
}

// -----------------------------------------------------------------------------------------
// ASOCIACIÓN Y NMS
// -----------------------------------------------------------------------------------------

/// Calcula la asociación mutua entre tracks y detecciones dada una matriz de costos (distancias) y un umbral de coincidencia
/// Devuelve una lista de pares [tdx, ddx] donde tdx es el índice del track y ddx es el índice de la detección asociada
/// TPA: Track-Pairwise Association -> cada track se asocia con la detección que tiene el costo mínimo,
/// y cada detección se asocia con el track que tiene el costo mínimo
/// Solo se acepta la asociación si ambos coinciden mutuamente (match mutuo) y el costo es menor que el umbral
pub fn associate(cost: &Array2<f32>, match_thr: f32) -> Vec<[usize; 2]> {
    let mut matches = Vec::new();

    if cost.nrows() > 0 && cost.ncols() > 0 {
        let mut min_ddx = vec![0; cost.nrows()]; // Índice de detección mínima para cada track
        let mut min_tdx = vec![0; cost.ncols()]; // Índice de track mínimo para cada detección

        // argmin(axis=1): Para cada track, buscar la mejor detección
        for (i, row) in cost.rows().into_iter().enumerate() {
            let mut min_val = f32::MAX;
            for (j, &val) in row.into_iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_ddx[i] = j;
                }
            }
        }

        // argmin(axis=0): Para cada detección, buscar el mejor track
        for (j, col) in cost.columns().into_iter().enumerate() {
            let mut min_val = f32::MAX;
            for (i, &val) in col.into_iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_tdx[j] = i;
                }
            }
        }

        // Match mutuo (TPA)
        for (tdx, &ddx) in min_ddx.iter().enumerate() {
            if min_tdx[ddx] == tdx && cost[[tdx, ddx]] < match_thr {
                matches.push([tdx, ddx]);
            }
        }
    }
    matches
}

/// Realiza la asociación iterativa entre tracks y detecciones utilizando TODAS las similitudes y penalizaciones.
/// Devuelve una tupla con la lista de matches (pares [tdx, ddx]), la lista de índices de tracks sin match (u_tracks) y la lista de índices de detecciones sin match (u_dets)       
/// El proceso es el siguiente:
/// 1. Calcular la matriz de similitud y costos combinando IoU, Cosine, Confianza y Ángulo.
/// 2. En cada iteración, usar la función associate para encontrar los matches mutuos con el umbral actual, agregar esos matches a la lista de matches finales, y "borrar" a los tracks y detecciones emparejados poniendo su costo al máximo para que no se vuelvan a emparejar en iteraciones posteriores
/// 3. Reducir el umbral y repetir el proceso hasta que no queden más matches posibles
/// 4. Al final, identificar cuáles tracks y detecciones quedaron sin match (solteros) y devolverlos junto con los matches encontrados      
pub fn iterative_assignment(
    tracks: &[Track],
    dets_high: &[Detection],
    dets_low: &[Detection],
    dets_del_high: &[Detection],
    mut match_thr: f32, // Umbral inicial de coincidencia
    penalty_p: f32,     // Penalización por baja confianza
    penalty_q: f32,     // Penalización por detecciones eliminadas
    reduce_step: f32,   // Cantidad en la que se reduce el umbral en cada iteración
    frame_id: usize,
    d_t: usize,
) -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
    let mut matches = Vec::new();

    // Juntamos todas las detecciones en una sola lista temporal para los cálculos
    let mut dets = Vec::new();
    dets.extend_from_slice(dets_high);
    dets.extend_from_slice(dets_low);
    dets.extend_from_slice(dets_del_high);

    // 1. Calculate preliminaries
    let (iou_sim, iou_dist) = iou_distance(tracks, &dets);
    let cos_dist = cos_distance(tracks, &dets);
    let conf_dist = conf_distance(tracks, &dets);
    let ang_dist = angle_distance(tracks, &dets, frame_id, d_t);

    let num_t = tracks.len();
    let num_d = dets.len();
    let mut cost = Array2::<f32>::zeros((num_t, num_d));

    // Calculate cost & Give penalties
    for i in 0..num_t {
        for j in 0..num_d {
            // let mut c = 0.50 * iou_dist[[i, j]]
            //     + 0.50 * cos_dist[[i, j]]
            //     + 0.10 * conf_dist[[i, j]]
            //     + 0.05 * ang_dist[[i, j]];
            let mut c =
                0.70 * iou_dist[[i, j]] + 0.20 * conf_dist[[i, j]] + 0.10 * ang_dist[[i, j]];

            // Give penalty según de qué lista proviene la detección
            if j >= dets_high.len() && j < dets_high.len() + dets_low.len() {
                c += penalty_p; // Era dets_low
            } else if j >= dets_high.len() + dets_low.len() {
                c += penalty_q; // Era dets_del_high
            }

            // Restricción: Si la similitud IoU es muy baja, el costo es 1.0 (inviable)
            if iou_sim[[i, j]] <= 0.10 {
                c = 1.0;
            }

            cost[[i, j]] = c.clamp(0.0_f32, 1.0_f32);
        }
    }

    // 2. Bucle iterativo de asociación (TPA Loop)
    loop {
        let matches_ = associate(&cost, match_thr);
        match_thr -= reduce_step;

        if matches_.is_empty() {
            break; // Si ya no hay parejas posibles, salimos
        }

        for &m in &matches_ {
            //  Agregar los matches encontrados en esta iteración a la lista final
            matches.push(m);
            let tdx = m[0];
            let ddx = m[1];

            // "Borramos" a los que ya se emparejaron poniendo su costo al máximo
            for j in 0..cost.ncols() {
                cost[[tdx, j]] = 1.0;
            }
            for i in 0..cost.nrows() {
                cost[[i, ddx]] = 1.0;
            }
        }
    }

    // 3. Identificar quiénes se han quedado solteros
    let mut u_tracks = Vec::new();
    let mut u_dets = Vec::new();

    let matched_tracks: Vec<usize> = matches.iter().map(|m| m[0]).collect(); // Índices de tracks que sí se emparejaron
    let matched_dets: Vec<usize> = matches.iter().map(|m| m[1]).collect(); // Índices de detecciones que sí se emparejaron

    // Cualquiera que no esté en matched_tracks o matched_dets es un soltero
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

/// Equivalente a `track_aware_nms`: Filtra las nuevas detecciones asegurándose
/// de que no se superponen con los tracks que ya estamos siguiendo (anclas).
pub fn track_aware_nms(
    pair_sims: &Array2<f32>, // Matriz de similitud (IoU) entre todas las cajas
    scores: &[f32],          // Confianzas de las detecciones huérfanas
    num_tracks: usize,       // Cantidad de anclas (Tracks emparejados)
    nms_thresh: f32,         // Límite de solapamiento permitido
    score_thresh: f32,       // Límite de confianza para existir
) -> Vec<bool> {
    let num_dets = pair_sims.nrows() - num_tracks;
    let mut allow_indices = vec![false; num_dets];

    // Check 1: Inicializar permitidos según si superan el score mínimo
    for idx in 0..num_dets {
        if scores[idx] > score_thresh {
            allow_indices[idx] = true;
        }
    }

    for idx in 0..num_dets {
        if !allow_indices[idx] {
            continue;
        }

        // Check 2: ¿Se solapa demasiado con un Track Activo (Ancla)?
        if num_tracks > 0 {
            let mut max_sim_with_track = 0.0_f32;
            for t in 0..num_tracks {
                let sim = pair_sims[[num_tracks + idx, t]];
                if sim > max_sim_with_track {
                    max_sim_with_track = sim;
                }
            }

            // Si choca con un track que ya existe, lo eliminamos asumiendo que es un duplicado
            if max_sim_with_track > nms_thresh {
                allow_indices[idx] = false;
                continue;
            }
        }

        // Check 3: Supresión estándar (NMS) entre las propias detecciones nuevas.
        // Si dos cajas nuevas se pisan entre sí, sobrevive la de mayor puntuación.
        for jdx in 0..num_dets {
            if idx != jdx && allow_indices[jdx] && scores[idx] > scores[jdx] {
                if pair_sims[[num_tracks + idx, num_tracks + jdx]] > nms_thresh {
                    allow_indices[jdx] = false;
                }
            }
        }
    }

    // Devuelve un vector de booleanos donde `true` significa que la caja sobrevive y nace como track
    allow_indices
}
