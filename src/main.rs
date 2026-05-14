use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
    videoio, Result,
};

use mot::fast_re_id::emb_computer::EmbeddingComputer;
use mot::tracktrack::track::Detection;
use mot::tracktrack::tracker::{Args, Tracker};
use std::time::Instant;

const SCALE: f64 = 0.25;
const INPUT_SIZE: i32 = 640;
const DETECT_EVERY_N_FRAMES: usize = 5;
const CONF_THRESHOLD: f32 = 0.3;
const NMS_THRESHOLD: f32 = 0.4;

fn main() -> Result<()> {
    let inv_scale = (1.0 / SCALE) as f32;

    let mut cam = videoio::VideoCapture::from_file("test.mp4", videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("No se pudo abrir el video.");
    }

    let mut net = dnn::read_net_from_onnx("yolov8n.onnx")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    let args = Args {
        max_time_lost: 30,
        det_thr: 0.5,
        match_thr: 0.8,
        penalty_p: 0.1,
        penalty_q: 0.2,
        reduce_step: 0.1,
        init_thr: 0.6,
        tai_thr: 0.4,
    };

    let mut tracker = Tracker::new(args, "test.mp4");
    let _embedder = EmbeddingComputer::new("fastreid_rust.onnx")
        .map_err(|e| opencv::Error::new(0, e.to_string()))?; // LO necesitaremos para FastReID

    let mut frame = Mat::default();
    let mut frame_num: usize = 0;
    let mut total_tracker_time = 0.0f64;
    let mut tracker_frames = 0usize;

    println!("Iniciando benchmark...");

    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            println!("Fin del video: {} frames procesados.", frame_num);
            break;
        }
        frame_num += 1;
        // println!("Frame {}", frame_num);

        let mut frame_small = Mat::default();
        imgproc::resize(
            &frame,
            &mut frame_small,
            Size::new(0, 0),
            SCALE,
            SCALE,
            imgproc::INTER_AREA,
        )?;

        if frame_num % DETECT_EVERY_N_FRAMES == 0 {
            let blob = dnn::blob_from_image(
                &frame_small,
                1.0 / 255.0,
                Size::new(INPUT_SIZE, INPUT_SIZE),
                Scalar::default(),
                true,
                false,
                core::CV_32F,
            )?;
            net.set_input(&blob, "", 1.0, Scalar::default())?;
            let mut output_blobs: Vector<Mat> = Vector::new();
            net.forward(&mut output_blobs, &net.get_unconnected_out_layers_names()?)?;

            let output = output_blobs.get(0)?;
            let size = output.mat_size();
            let is_yolov8 = size[1] < size[2];
            let num_preds = if is_yolov8 {
                size[2] as usize
            } else {
                size[1] as usize
            };
            let num_attrs = if is_yolov8 {
                size[1] as usize
            } else {
                size[2] as usize
            };
            let num_classes = if is_yolov8 {
                num_attrs - 4
            } else {
                num_attrs - 5
            };
            let x_factor = frame_small.cols() as f32 / INPUT_SIZE as f32;
            let y_factor = frame_small.rows() as f32 / INPUT_SIZE as f32;
            let data = output.data_typed::<f32>()?;

            let mut confidences = Vector::<f32>::new();
            let mut boxes = Vector::<Rect>::new();

            for p in 0..num_preds {
                let (cx, cy, w, h, confidence) = if is_yolov8 {
                    let cx = data[0 * num_preds + p];
                    let cy = data[1 * num_preds + p];
                    let w = data[2 * num_preds + p];
                    let h = data[3 * num_preds + p];
                    let conf = (4..4 + num_classes)
                        .map(|j| data[j * num_preds + p])
                        .fold(0f32, f32::max);
                    (cx, cy, w, h, conf)
                } else {
                    let b = p * num_attrs;
                    let conf = data[b + 4]
                        * (5..5 + num_classes)
                            .map(|j| data[b + j])
                            .fold(0f32, f32::max);
                    (data[b], data[b + 1], data[b + 2], data[b + 3], conf)
                };

                if confidence >= CONF_THRESHOLD {
                    boxes.push(Rect::new(
                        ((cx - w / 2.0) * x_factor) as i32,
                        ((cy - h / 2.0) * y_factor) as i32,
                        (w * x_factor) as i32,
                        (h * y_factor) as i32,
                    ));
                    confidences.push(confidence);
                }
            }

            let mut indices = Vector::<i32>::new();
            dnn::nms_boxes(
                &boxes,
                &confidences,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
                &mut indices,
                1.0,
                0,
            )?;

            let mut detections = Vec::new();
            for idx in indices {
                let rect = boxes.get(idx as usize)?;
                let conf = confidences.get(idx as usize)?;
                let bbox = [
                    rect.x as f32 * inv_scale,
                    rect.y as f32 * inv_scale,
                    (rect.x + rect.width) as f32 * inv_scale,
                    (rect.y + rect.height) as f32 * inv_scale,
                ];
                // let feat = embedder.compute_embedding(&frame, &bbox)?;
                detections.push(Detection {
                    bbox,
                    score: conf,
                    feat: Vec::new(),
                });
            }

            let start = Instant::now();
            tracker.update(detections, Vec::new());
            total_tracker_time += start.elapsed().as_secs_f64();
            tracker_frames += 1;
        } else {
            let start = Instant::now();
            tracker.update_without_detections();
            total_tracker_time += start.elapsed().as_secs_f64();
            tracker_frames += 1;
        }
    }

    println!("=== RESULTADOS RUST ===");
    println!("Tiempo total en el tracker: {:.4} s", total_tracker_time);
    println!(
        "Velocidad media: {:.2} FPS",
        tracker_frames as f64 / total_tracker_time
    );

    Ok(())
}
