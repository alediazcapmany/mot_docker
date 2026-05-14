
use opencv::core::{Mat, Rect, Scalar, Size};
use opencv::{core, dnn, prelude::*};
// use ort::session::Session; // ← módulo correcto
// use ort::value::Tensor;
pub struct EmbeddingComputer {
    session: ort::session::Session,
}

impl EmbeddingComputer {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn compute_embedding(&self, frame: &Mat, bbox: &[f32; 4]) -> opencv::Result<Vec<f32>> {
        let img_w = frame.cols();
        let img_h = frame.rows();

        let mut x = bbox[0].round() as i32;
        let mut y = bbox[1].round() as i32;
        let mut w = (bbox[2] - bbox[0]).round() as i32;
        let mut h = (bbox[3] - bbox[1]).round() as i32;

        if x < 0 {
            w += x;
            x = 0;
        }
        if y < 0 {
            h += y;
            y = 0;
        }
        if x + w > img_w {
            w = img_w - x;
        }
        if y + h > img_h {
            h = img_h - y;
        }
        if w <= 0 || h <= 0 {
            return Ok(Vec::new());
        }

        let crop = Mat::roi(frame, Rect::new(x, y, w, h))?;
        let blob = dnn::blob_from_image(
            &crop,
            1.0 / 255.0,
            Size::new(128, 384),
            Scalar::default(),
            true,
            false,
            core::CV_32F,
        )?;

        // ← Array4 → IxDyn para ort 2.x
        let data = blob.data_typed::<f32>()?.to_vec();

        let tensor =
            ort::value::Tensor::<f32>::from_array(([1usize, 3, 384, 128], data.into_boxed_slice()))
                .map_err(|e| opencv::Error::new(0, e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![tensor].map_err(|e| opencv::Error::new(0, e.to_string()))?)
            .map_err(|e| opencv::Error::new(0, e.to_string()))?;

        let feat_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| opencv::Error::new(0, e.to_string()))?;

        let feat_vec: Vec<f32> = feat_tensor.view().iter().cloned().collect();
        let norm = feat_vec
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);

        Ok(feat_vec.iter().map(|v| v / norm).collect())
    }
}
