# Posture Inference and KPI Evaluation

This project performs posture classification inference on multiple video files and computes KPI metrics against ground-truth annotations.

The inference pipeline reproduces the embedded C++ preprocessing implementation, including:

- Bounding box cropping with a 10% horizontal margin
- Aspect-ratio-preserving resize
- Centered black padding (letterbox)
- OpenCV `INTER_LINEAR` interpolation
- Optional body keypoint skeleton overlay
- KPI computation per video and globally

## Features

- Batch processing of multiple videos
- ONNX model inference
- Automatic tracking CSV detection
- Per-frame posture prediction export
- Ground-truth comparison
- Automatic KPI computation:
  - Precision
  - Recall
  - F1 Score
  - TP / TN / FP / FN
- Global KPI aggregation across all processed videos

---

## Project Structure

```text
.
├── data/
│   └── forvia4testset/
│       ├── video_1.mp4
│       ├── video_1.csv
│       ├── video_2.mp4
│       └── video_2.csv
│
├── weights/
│   └── onnx/
│       └── epoch_40_batch_1.onnx
│
├── src/
│   ├── config/
│   │   └── eval_onnx_july_4_forvia_sequences.yaml
│   ├── inference_python/
│   │   └── py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py
│   └── model/
│
└── output/
```

---

## Requirements

Main dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install pandas numpy
pip install tqdm fire
pip install scikit-learn
pip install onnxruntime-gpu
```

---

## Running Inference

Execute the following command from the project root directory:

```bash
python3 src/inference_python/py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py \
    --model_path ./weights/onnx/epoch_40_batch_1.onnx \
    --videos_dir ./data/forvia4testset \
    --with_tatoo_keypoints False \
    --output_dir ./output/forvia4test2/golden_ref_kpi/onnx \
    --config_path ./src/config/eval_onnx_july_4_forvia_sequences.yaml \
    --gt_csv_dir ./output/inference/forvia4test2
```

---

## Input Data

### Videos

All `.mp4` files located in:

```text
./data/forvia4testset
```

will be processed automatically.

### Tracking CSV

Each video must have a matching tracking CSV file:

```text
video_name.mp4
video_name.csv
```

The tracking CSV must contain the occupant bounding box information used for cropping.

### Ground Truth CSV

Ground-truth files are searched in:

```text
./output/inference/forvia4test2
```

Expected formats:

```text
video_name_eval_with_gt.csv
```

or

```text
video_name.csv
```

---

## Output Files

### Prediction CSV

For each video:

```text
output/
└── forvia4test2/
    └── golden_ref_kpi/
        └── onnx/
            ├── video_1_predictions.csv
            ├── video_2_predictions.csv
            └── ...
```

### KPI Summary

A global KPI report is generated automatically:

```text
KPI_summary_videos.csv
```

Containing:

- Sequence
- Class
- TP
- TN
- FP
- FN
- Precision
- Recall
- F1 Score
- Total Frames

---

## Supported Classes

The evaluator computes metrics for the following posture classes:

```text
closetoairbag
legsondashboard
torsodissimetry
reachingback
lateralbodycontact
oneshoulderincontact
pelvisrotation
pelvisdrift
```

---

## Notes

- ONNX inference uses CUDA when available.
- Invalid frames are ignored during KPI computation.
- KPI metrics are computed both per-video and globally across all processed sequences.
- The preprocessing pipeline is aligned with the embedded C++ implementation to ensure result consistency.
