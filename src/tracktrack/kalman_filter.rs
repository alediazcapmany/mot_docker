use nalgebra::{SMatrix, SVector, Cholesky};

// ── DEFINICIONES DE TIPOS ───────────────────────────────────────────────────
pub type StateVec = SVector<f32, 8>;
pub type MeasureVec = SVector<f32, 4>;
pub type StateMat = SMatrix<f32, 8, 8>;
pub type MeasureMat = SMatrix<f32, 4, 8>;

#[derive(Clone)]
pub struct KalmanFilter {
    motion_mat: StateMat,
    update_mat: MeasureMat,
    std_pos: f32,
    std_vel: f32,
}

impl KalmanFilter {
    pub fn new() -> Self {
        // Matriz de movimiento (F)
        let mut motion_mat = StateMat::identity();
        motion_mat[(0, 4)] = 1.0;
        motion_mat[(1, 5)] = 1.0;
        motion_mat[(2, 6)] = 1.0;
        motion_mat[(3, 7)] = 1.0;

        // Matriz de actualización (H)
        let mut update_mat = MeasureMat::zeros();
        update_mat[(0, 0)] = 1.0;
        update_mat[(1, 1)] = 1.0;
        update_mat[(2, 2)] = 1.0;
        update_mat[(3, 3)] = 1.0;

        Self {
            motion_mat,
            update_mat,
            std_pos: 1.0 / 20.0,
            std_vel: 1.0 / 160.0,
        }
    }

    /// Inicializa la media y la covarianza de un track nuevo
    pub fn initiate(&self, measurement: &[f32; 4]) -> (StateVec, StateMat) {
        let mut mean = StateVec::zeros();
        mean[0] = measurement[0]; // cx
        mean[1] = measurement[1]; // cy
        mean[2] = measurement[2]; // w
        mean[3] = measurement[3]; // h
                                  // La velocidad (índices 4 a 7) ya es 0.0

        let mut covariance = StateMat::zeros();
        for i in 0..8 {
            if i < 4 {
                covariance[(i, i)] = 2.0;
            } else {
                covariance[(i, i)] = 10.0;
            }
        }

        // Multiplicamos por los factores del bbox (mean[2] es width, mean[3] es height)
        let w = mean[2];
        let h = mean[3];

        covariance[(0, 0)] *= w;
        covariance[(2, 2)] *= w;
        covariance[(4, 4)] *= w;
        covariance[(6, 6)] *= w;

        covariance[(1, 1)] *= h;
        covariance[(3, 3)] *= h;
        covariance[(5, 5)] *= h;
        covariance[(7, 7)] *= h;

        // Elevamos al cuadrado todos los elementos (np.square en Python)
        covariance.component_mul_assign(&covariance.clone());

        (mean, covariance)
    }

    /// Predice el próximo estado
    pub fn predict(&self, mean: &StateVec, covariance: &StateMat) -> (StateVec, StateMat) {
        let next_mean = self.motion_mat * mean;

        let mut motion_cov = StateMat::zeros();
        for i in 0..8 {
            if i < 4 {
                motion_cov[(i, i)] = self.std_pos;
            } else {
                motion_cov[(i, i)] = self.std_vel;
            }
        }

        let w = next_mean[2];
        let h = next_mean[3];

        motion_cov[(0, 0)] *= w;
        motion_cov[(2, 2)] *= w;
        motion_cov[(4, 4)] *= w;
        motion_cov[(6, 6)] *= w;

        motion_cov[(1, 1)] *= h;
        motion_cov[(3, 3)] *= h;
        motion_cov[(5, 5)] *= h;
        motion_cov[(7, 7)] *= h;

        motion_cov.component_mul_assign(&motion_cov.clone());

        let next_covariance =
            self.motion_mat * covariance * self.motion_mat.transpose() + motion_cov;

        (next_mean, next_covariance)
    }

    /// Proyecta el estado al espacio de medición (usado internamente por update)
    pub fn project(
        &self,
        mean: &StateVec,
        covariance: &StateMat,
        confidence: f32,
    ) -> (MeasureVec, nalgebra::SMatrix<f32, 4, 4>) {
        let projected_mean = self.update_mat * mean;

        let mut innovation_cov = nalgebra::SMatrix::<f32, 4, 4>::zeros();
        for i in 0..4 {
            innovation_cov[(i, i)] = self.std_pos;
        }

        let w = mean[2];
        let h = mean[3];

        innovation_cov[(0, 0)] *= w;
        innovation_cov[(2, 2)] *= w;

        innovation_cov[(1, 1)] *= h;
        innovation_cov[(3, 3)] *= h;

        innovation_cov.component_mul_assign(&innovation_cov.clone());

        // NSA: Noise Scale Adaptive (Ajusta la incertidumbre según la confianza del detector)
        innovation_cov *= 1.0 - confidence;

        let projected_cov =
            self.update_mat * covariance * self.update_mat.transpose() + innovation_cov;

        (projected_mean, projected_cov)
    }

    /// Actualiza el estado con una nueva medición
    pub fn update(
        &self,
        mean: &StateVec,
        covariance: &StateMat,
        measurement: &[f32; 4],
        confidence: f32,
    ) -> (StateVec, StateMat) {
        let (projected_mean, projected_cov) = self.project(mean, covariance, confidence);

        let z = MeasureVec::from_row_slice(measurement);

        // En lugar de scipy.linalg.cho_solve, usamos nalgebra para invertir (es una matriz de 4x4, rapidísimo)
        // K = P * H^T * S^-1
        let ph_t = covariance * self.update_mat.transpose(); // P * H^T  (8×4)
        let kalman_gain = match Cholesky::new(projected_cov) {
            Some(chol) => {
                // Resuelve S * K^T = (P H^T)^T  →  K = solución^T
                let k_t = chol.solve(&ph_t.transpose()); // 4×8
                k_t.transpose() // 8×4
            }
            None => ph_t * projected_cov.try_inverse().unwrap_or_default(),
        };
        let innovation = z - projected_mean;

        let new_mean = mean + kalman_gain * innovation;
        let new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

        (new_mean, new_covariance)
    }
}
