use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    video::KalmanFilter,
    videoio, Result,
};

// ── 1. ESTADO DEL TRACK ──────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum TrackState {
    Posible, // age < 3 frames confirmados
    Confirmed, // age >= 3 frames confirmados
    Lost,      // No match en este frame, guardamos por si reaparece
    Deleted,   // Demasiado perdido, se borra de la lista
}

// ── 2. INFO DEL TRACK (PERSONA) ─────────────────────────────────────────────────
pub struct Track {  // Todo lo que quiero que guarde cada track de persona que estoy siguiendo
    pub track_id: usize,
    pub state: TrackState,
    
    // Pos actual: [top_left_x, top_left_y, width, height]
    pub tlwh: [f32; 4], 
    
    // Predice pos del track (OpenCV)
    pub kalman_filter: KalmanFilter, 
    
    // Contadores para la lógica de TrackTrack
    pub age: usize,               // Total de frames que lleva existiendo
    pub time_since_update: usize, // Frames seguidos sin hacer match (si llega a 30, state = Deleted)
    
    // HUECO PARA EL FUTURO: Aquí guardaremos el vector de FastReID
    // pub embedding: Option<Vec<f32>>, 
}

impl Track {    // Implemento el track, con su construcción y funciones (metodos) para actualizarlo facilmente 
    /// Crea nuevo Track con una detección huérfana 
    pub fn new(id: usize, bbox: [f32; 4]) -> Result<Self> {
        // Inicializamos el Filtro de Kalman (requiere configuración matemática)
        let kf = Self::init_kalman(bbox)?;

        Ok(Self {
            track_id: id,
            state: TrackState::Posible, // Todos nacen como Posibles
            tlwh: bbox,
            kalman_filter: kf,
            age: 1,
            time_since_update: 0,
            // embedding: None, 
        })  //Devuelve nuevo track con: ID, caja y el KF para predecir movimiento
    }

    /// Método interno para configurar las matrices de OpenCV KalmanFilter
    use opencv::core::{Mat, Rect, CV_32F};

    /// Método interno para configurar las matrices de OpenCV KalmanFilter
    /// HECHO POR GEMINI, MIRAR SI HAY FILTRO KALMAN EN RUST ESTANDARD O SI HAY QUE HACERLO MANUALMENTE !!!!!!!!!! 
    fn init_kalman(bbox: [f32; 4]) -> Result<KalmanFilter> { 
        // 8 variables de estado (cx, cy, ratio, altura, vel_cx, vel_cy, vel_ratio, vel_altura)
        // 4 variables de medida (cx, cy, ratio, altura)
        let mut kf = KalmanFilter::new(8, 4, 0, CV_32F)?;
        
        // 1. Convertir la caja [top_left_x, top_left_y, w, h] al formato Kalman [cx, cy, ratio, h]
        let w = bbox[2];
        let h = bbox[3];
        let cx = bbox[0] + w / 2.0;
        let cy = bbox[1] + h / 2.0;
        let ratio = w / h; // La proporción del cuerpo (ancho / alto)

        // 2. Matriz de Transición (F): Relaciona la posición actual con la siguiente basándose en la velocidad.
        // Matemáticamente: posicion_nueva = posicion_actual + velocidad * dt
        let mut transition_matrix = Mat::eye(8, 8, CV_32F)?.to_mat()?;
        for i in 0..4 {
            *transition_matrix.at_2d_mut::<f32>(i, i + 4)? = 1.0; // Enlazamos posición con velocidad
        }
        kf.set_transition_matrix(transition_matrix);

        // 3. Matriz de Medida (H): Mapea las 8 variables internas a las 4 reales que detecta YOLO
        let measurement_matrix = Mat::eye(4, 8, CV_32F)?.to_mat()?;
        kf.set_measurement_matrix(measurement_matrix);

        // 4. Estado Inicial: Le decimos dónde está la persona en el frame 1 y asumimos velocidad 0
        let mut state = Mat::zeros(8, 1, CV_32F)?.to_mat()?;
        *state.at_2d_mut::<f32>(0, 0)? = cx;
        *state.at_2d_mut::<f32>(1, 0)? = cy;
        *state.at_2d_mut::<f32>(2, 0)? = ratio;
        *state.at_2d_mut::<f32>(3, 0)? = h;
        kf.set_state_post(state);

        // 5. Matriz de Covarianza del Error (P): Cuánta incertidumbre tenemos al principio
        let mut error_cov = Mat::eye(8, 8, CV_32F)?.to_mat()?;
        // Damos más incertidumbre a las velocidades porque no sabemos hacia dónde va a caminar
        for i in 4..8 {
            *error_cov.at_2d_mut::<f32>(i, i)? = 1000.0;
        }
        // Menos incertidumbre en la posición porque la caja de YOLO es bastante precisa
        for i in 0..4 {
            *error_cov.at_2d_mut::<f32>(i, i)? = 10.0;
        }
        kf.set_error_cov_post(error_cov);

        Ok(kf)
    }

    /// Llama a este método en cada frame para adivinar dónde se ha movido la persona
    pub fn predict(&mut self) -> Result<()> {
        let prediction = self.kalman_filter.predict(&Mat::default())?;
        
        // Extraemos las coordenadas predichas (cx, cy, ratio, h) del tensor
        let cx = *prediction.at_2d::<f32>(0, 0)?;
        let cy = *prediction.at_2d::<f32>(1, 0)?;
        let ratio = *prediction.at_2d::<f32>(2, 0)?;
        let h = *prediction.at_2d::<f32>(3, 0)?;
        
        let w = ratio * h;
        
        // Actualizamos la caja del track (tlwh) con la predicción
        self.tlwh = [
            cx - w / 2.0, // top_left_x
            cy - h / 2.0, // top_left_y
            w,            // width
            h             // height
        ];
        
        Ok(())
    }

    /// Actualiza track al hacer match con una detección (posición, estado y apariencia)
    pub fn mark_matched(&mut self, new_bbox: [f32; 4]) {
        self.time_since_update = 0; //Reinicia ultima detección
        self.age += 1;  // Aumenta edad
        self.tlwh = new_bbox;   // Actualiza pos caja
        
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
        self.state = TrackState::Lost;  // Pasa a estado perdido
        self.time_since_update += 1;    // Aumenta frame sin detección
    }

    
}

// ── 3. TRACK TRACK ───────────────────────────────────────────────
pub struct TrackTrack {
    // Listas que separan a las personas según su estado
    pub active_tracks: Vec<Track>, // Los que estamos viendo y siguiendo
    pub lost_tracks: Vec<Track>,   // Los que se ocultaron
    
    // Parámetros de configuración
    next_id: usize,           // Contador de IDs
    max_time_lost: usize,     // Frames que se guarda track perdido
}

impl TrackTrack {
    pub fn new(max_time_lost: usize) -> Self {  // Inicializa el tracker tracktrack
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
        for mut lost_track in unmatched_vips.into_iter().chain(unmatched_novices.into_iter()) {
            lost_track.mark_lost();
            self.lost_tracks.push(lost_track);
        }

        // Limpiar la memoria: borrar definitivamente los que llevan perdidos mucho tiempo
        self.lost_tracks.retain(|t| t.time_since_update < self.max_time_lost);

        Ok(())
    }

    // ─── STUBS DE LAS FUNCIONES MATEMÁTICAS ─────────────────────────────────────
    
    /// El algoritmo TPA: Calcula la matriz de costos y busca emparejamientos mutuos
    fn tpa_match(
        &self, 
        tracks: Vec<Track>, 
        detections: &Vec<Rect>
    ) -> (Vec<Track>, Vec<Track>, Vec<Rect>) {
        
        // HUECO: Aquí irá el cálculo de la matriz de Intersección sobre Unión (IoU).
        
        // De momento, devolvemos todo sin emparejar para que el código compile
        (Vec::new(), tracks, detections.clone())
    }

    /// Filtro TAI: Usa NMS para evitar que nazcan tracks duplicados sobre personas existentes
    fn apply_tai(
        &mut self, 
        anchors: &Vec<[f32; 4]>, 
        orphan_detections: &Vec<Rect>
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