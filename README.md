# Posture Inference

## Environment Setup

Create and activate your conda environment:

```bash
conda create -n posture_inference python=3.10 -y
conda activate posture_inference
```

Install required packages:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy pandas
pip install tqdm fire
pip install scikit-learn
pip install onnxruntime-gpu
```

---

## Run Inference

From the project root directory:

```bash
python3 src/inference_python/py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py \
    --model_path ./weights/onnx/epoch_40_batch_1.onnx \
    --videos_dir ./data/forvia4testset \
    --with_tatoo_keypoints False \
    --output_dir ./output/forvia4test2/golden_ref_kpi/onnx \
    --config_path ./src/config/eval_onnx_july_4_forvia_sequences.yaml \
    --gt_csv_dir ./gt
```

### Inputs

- `--model_path`: ONNX model file
- `--videos_dir`: folder containing `.mp4` files and associated tracking `.csv`
- `--config_path`: model configuration file
- `--gt_csv_dir`: ground-truth CSV folder used for KPI computation

### Outputs

- Per-video prediction CSV files
- `KPI_summary_videos.csv` containing Precision, Recall, F1-Score, TP, TN, FP, and FN metrics
