# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**3D Gaussian Splatting Noise Detection** - A modular pipeline for detecting and masking noisy artifacts in novel views rendered by 3D Gaussian Splatting (3DGS).

### Project Goal
Develop a noise detection model that can identify rendering artifacts in 3DGS novel views through a 6-stage modular pipeline that is:
- ✅ **Dataset-independent**: Works with any scene data
- ✅ **Step-by-step execution**: Each stage runs independently
- ✅ **Checkpointed**: Results saved at each stage
- ✅ **Resumable**: Can restart from any specific stage
- ✅ **Strictly isolated**: Test views never seen during training (COLMAP filtering)

### Core Components
This repository combines two main systems:

1. **Base 3DGS System** (inherited)
   - PyTorch-based 3DGS optimizer (`train.py`)
   - CUDA rasterization and rendering (`gaussian_renderer/`)
   - Real-time OpenGL viewer (`SIBR_viewers/`)
   - Scene representation and utilities (`scene/`, `utils/`)

2. **Noise Detection Pipeline** (this project)
   - 6-stage modular data generation and training pipeline
   - Automated dataset generation from any scene
   - U-Net based noise mask prediction
   - LPIPS + pixel-wise L1 error for ground truth mask generation
   - Strict experimental isolation via COLMAP sparse filtering

---

## 🔄 6-Stage Pipeline Architecture

```
Stage 1: Train/Test Split & COLMAP Filtering
  Input:  data/360_v2/{scene}/input/ (125 images), sparse/ (COLMAP)
  Output: data/processed/{scene}/images/ (93 train images)
          data/processed/{scene}/sparse/ (filtered, 93 cameras only)
          splits/train_indices.json, test_indices.json, test_cameras.json

Stage 2: 3DGS Training (93 images)
  Input:  data/processed/{scene}/images/, sparse/
  Output: data/processed/{scene}/3dgs_output/ (93-image model)

Stage 3: Novel View Rendering (Test views from 93-image model)
  Input:  3dgs_output/, test_cameras.json
  Output: rendered_views/ (render_*.png, gt_*.png for ~25 test views)

Stage 4: Error Map & Dataset Pairs Generation
  Input:  rendered_views/ (rendered + GT)
  Output: dataset_pairs/train/, val/, test/ (triplets: rendered, gt, mask)

Stage 5: Detector Training
  Input:  dataset_pairs/
  Output: experiments/{scene}/checkpoints/best_model.pth

Stage 6: Interpolated View Detection (Validation on 125-image model)
  Input:  output/{scene}/ (125-image model), best_model.pth
  Output: interpolated_results/ (rendered, mask, overlay visualizations)
```

---

## Project Structure

```
GS_noise_masking/
│
├── CLAUDE.md                      # This file - project guidance
├── README.md                      # Original 3DGS documentation
├── claude_modular_pipeline.md     # Detailed pipeline design document
├── environment.yml                # Conda environment
│
├── data/                          # Data directory
│   ├── 360_v2/                    # Original scene data (Mip-NeRF 360 format)
│   │   └── {scene_name}/
│   │       ├── input/             # Input images (125 images)
│   │       └── sparse/0/          # Full COLMAP (125 cameras)
│   │
│   └── processed/                 # Pipeline outputs
│       └── {scene_name}/
│           ├── images/            # Stage 1: Train images only (93)
│           ├── sparse/0/          # Stage 1: Filtered COLMAP (93 cameras)
│           ├── splits/            # Stage 1: train/test indices + test_cameras.json
│           ├── 3dgs_output/       # Stage 2: 93-image trained model
│           ├── rendered_views/    # Stage 3: rendered + GT images
│           ├── dataset_pairs/     # Stage 4: training pairs
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           └── interpolated_results/  # Stage 6: validation results
│
├── src/                           # Pipeline implementation ✅ COMPLETED
│   ├── step1_split.py             # Stage 1: Train/test split + COLMAP filtering
│   ├── step2_train_3dgs.py        # Stage 2: 3DGS training wrapper
│   ├── step3_render_views_v2.py   # Stage 3: Novel view rendering (custom)
│   ├── step4_generate_pairs.py    # Stage 4: Error map & dataset pairs
│   ├── step5_train_detector.py    # Stage 5: Detector training
│   ├── step6_interpolate_detect.py # Stage 6: Validation on interpolated views
│   │
│   ├── utils/                     # Pipeline utilities
│   │   ├── custom_render.py       # Custom rendering for specific views
│   │   ├── camera_interpolation.py # Camera pose interpolation (SLERP)
│   │   ├── detector_inference.py  # Detector model inference wrapper
│   │   └── visualization_utils.py # Visualization and overlay tools
│   │
│   └── models/                    # Detection models
│       ├── simple_unet.py         # U-Net architecture (3→1 channels)
│       ├── losses.py              # Combined BCE + Dice loss
│       └── metrics.py             # IoU, F1-score metrics
│
├── scripts/                       # Execution scripts
│   ├── 01_generate_stump_data.py  # Example: Stump dataset pipeline
│   ├── run_all.py                 # Run full pipeline (to be created)
│   └── run_step.py                # Run specific stage (to be created)
│
├── experiments/                   # Training outputs
│   └── {scene_name}_{exp_id}/
│       ├── checkpoints/
│       ├── logs/
│       └── results/
│
├── output/                        # 3DGS training outputs
│   └── {scene_name}/
│
├── train.py                       # Base 3DGS training script
├── render.py                      # Base 3DGS rendering script
├── metrics.py                     # Base 3DGS evaluation metrics
│
├── scene/                         # 3DGS scene representation
│   ├── gaussian_model.py          # Core Gaussian representation
│   ├── cameras.py                 # Camera utilities
│   └── dataset_readers.py         # Dataset loading
│
├── gaussian_renderer/             # 3DGS rendering
│   └── __init__.py                # Rasterization interface
│
├── utils/                         # 3DGS utilities
│   ├── loss_utils.py              # L1 + SSIM losses
│   ├── graphics_utils.py          # Graphics math
│   └── camera_utils.py            # Camera transformations
│
└── submodules/                    # CUDA extensions
    ├── diff-gaussian-rasterization/
    ├── simple-knn/
    └── fused-ssim/
```

---

## Development Workflow

### Environment Setup

```bash
# Create conda environment
conda env create --file environment.yml
conda activate gaussian_splatting

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Pipeline Execution

#### Full Pipeline (All Stages)
```bash
# Run complete pipeline on a scene
python scripts/run_all.py --scene my_scene --skip_frames 3 --iterations 30000 --epochs 50
```

#### Stage-by-Stage Execution
```bash
# Stage 1: Create train/test split + COLMAP filtering
python src/step1_split.py --scene stump --skip_frames 3
# Output: 93 train images, filtered COLMAP, test camera poses

# Stage 2: Train 3DGS on 93 training views
python src/step2_train_3dgs.py --scene stump --iterations 30000
# Uses only 93 images, creates 93-image 3DGS model

# Stage 3: Render novel test views from 93-image model
python src/step3_render_views_v2.py --scene stump
# Renders ~25 test views using test_cameras.json

# Stage 4: Generate error maps and dataset pairs
python src/step4_generate_pairs.py --scene stump --threshold 0.15 --min_size 30
# ⚠️ CRITICAL: threshold should be 0.15-0.2 (NOT 0.03!)

# Stage 5: Train noise detection model
python src/step5_train_detector.py --scene stump --num_epochs 50 --batch_size 4

# Stage 6: Validate on interpolated views (125-image model)
python src/step6_interpolate_detect.py --scene stump
# Uses output/stump/ (125-image model) for validation
```

#### Direct 3DGS Operations
```bash
# Train 3DGS directly (base system)
python train.py -s data/raw/my_scene

# Render trained model
python render.py -m output/my_scene

# Evaluate metrics
python metrics.py -m output/my_scene
```

---

## Key Implementation Details

### Stage 1: Train/Test Split & COLMAP Filtering ⭐ **KEY INNOVATION**
- **Purpose**: Separate views for 3DGS training vs novel view evaluation **with strict isolation**
- **Strategy**: Every Nth frame becomes a test view (skip_frames=3 → 125 images → 93 train, 32 test)
- **COLMAP Filtering** (critical for experiment validity):
  - Reads original COLMAP sparse (125 cameras, all points3D)
  - **Filters to only train cameras** (93 cameras)
  - **Filters points3D** to only those observed in train views (≥2 observations)
  - Saves test camera poses separately in `test_cameras.json`
  - **Result**: 3DGS training has ZERO information about test views
- **Output**:
  - `images/`: 93 train images only
  - `sparse/0/`: Filtered COLMAP (93 cameras, train points3D only)
  - `splits/train_indices.json`, `test_indices.json`, `test_cameras.json`
- **Implementation**: Uses `utils/read_write_model.py` for COLMAP binary I/O

### Stage 2: 3DGS Training (93 images)
- **Input**: `data/processed/{scene}/` (images/ + sparse/)
- **Process**: Standard 3DGS optimization (30k iterations default)
- **Output**: `3dgs_output/` - Gaussian point cloud trained on 93 views only
- **Key**: `train.py` automatically uses only images present in images/ directory

### Stage 3: Novel View Rendering (Test views from 93-image model)
- **Input**:
  - `3dgs_output/` (93-image trained model)
  - `splits/test_cameras.json` (test camera poses from original COLMAP)
- **Process**: Render from test camera poses using 93-image model
- **Output**: `rendered_views/` with `render_*.png` (rendered) and `gt_*.png` (ground truth)
- **File**: `step3_render_views_v2.py` uses `src/utils/custom_render.py`
- **Result**: ~24-25 rendered test views (some may be filtered during loading)

### Stage 4: Error Map & Dataset Pairs Generation ⚠️ **HYPERPARAMETER SENSITIVE**
- **Methodology**:
  1. Compute pixel-wise L1 error: `|rendered - gt|.mean(axis=-1) / 255.0`
  2. Compute LPIPS perceptual error (NOTE: returns scalar, not spatial map)
  3. Normalize both to [0, 1]
  4. Combine: `combined_error = 0.5 * L1_norm + 0.5 * LPIPS_norm`
  5. Binary threshold: `mask = (combined_error > threshold)`
  6. Remove small regions: connected components < `min_size` pixels
- **Critical Hyperparameters**:
  - `threshold`: **0.15-0.2 recommended** (default 0.03 is TOO LOW!)
  - `min_size`: **20-30 pixels** (default 50 may remove too much)
  - Goal: Noise ratio should be **15-30%** (not 1-3%!)
- **Output**: Triplets of (rendered, GT, mask) split into train/val/test (80/10/10)
- **Common Issue**: If noise_ratio < 5%, threshold is too high → model will fail

### Stage 5: Detector Training
- **Architecture**: Simple U-Net (3→1 channels), image_size=256
- **Input**: Rendered images (resized to 256x256)
- **Target**: Binary noise masks (resized to 256x256)
- **Loss**: BCE (Binary Cross-Entropy)
- **Metrics**: IoU (Intersection over Union)
- **Hyperparameters**:
  - `batch_size`: 4 (default, good for small datasets)
  - `num_epochs`: 50
  - `learning_rate`: 0.001
- **Output**: `experiments/{scene}/checkpoints/best_model.pth`

### Stage 6: Interpolated View Detection (Validation) ⭐ **NEW STAGE**
- **Purpose**: Validate detector on interpolated views from 125-image model
- **Input**:
  - `output/{scene}/` (125-image 3DGS model - separately trained)
  - `experiments/{scene}/checkpoints/best_model.pth`
- **Process**:
  1. Find nearest camera pairs in 125-image model
  2. Create interpolated cameras (SLERP for rotation, linear for position)
  3. Render interpolated views
  4. Apply trained detector
  5. Visualize results (rendered, mask, overlay)
- **Output**: `interpolated_results/` with visualizations and summary.json
- **Expected**: Noise ratio should be **20-40%** for well-trained detector

---

## Hardware Requirements

### For 3DGS Training (Stages 2-3):
- CUDA GPU with Compute Capability 7.0+
- 24 GB VRAM recommended (can work with less for smaller scenes)
- CUDA SDK 11.8 or 12.x

### For Detector Training (Stage 5):
- 8 GB+ VRAM
- Any modern CUDA GPU

---

## Dataset Requirements

### Expected Input Structure
```
data/360_v2/{scene_name}/
├── input/               # RGB images (.jpg, .JPG)
│   ├── _DSC9239.JPG
│   ├── _DSC9240.JPG
│   └── ... (125 images)
└── sparse/              # COLMAP camera information
    └── 0/
        ├── cameras.bin  # Camera intrinsics
        ├── images.bin   # 125 camera poses
        └── points3D.bin # 3D points (from all 125 views)
```

**Note**: This is Mip-NeRF 360 dataset format. COLMAP sparse contains ALL 125 cameras initially, then Step 1 filters it down to 93.

### Dataset Preparation
If you have raw images without COLMAP data:
```bash
python convert.py -s data/raw/{scene_name}
```

---

## Configuration Files

### Pipeline Configuration (Recommended Parameters)

**Step 1 (Train/Test Split):**
```bash
--scene stump
--skip_frames 3        # 125 images → 93 train, 32 test
--data_root ./data/360_v2
--output_root ./data/processed
```

**Step 2 (3DGS Training):**
```bash
--scene stump
--iterations 30000     # Standard for Mip-NeRF 360
--output_root ./data/processed
```

**Step 4 (Error Map Generation) ⚠️ CRITICAL:**
```bash
--scene stump
--threshold 0.15       # 0.15-0.2 recommended (NOT 0.03!)
--min_size 30          # 20-30 pixels
--train_ratio 0.8
--val_ratio 0.1
--test_ratio 0.1
--device cuda
```

**Step 5 (Detector Training):**
```bash
--scene stump
--num_epochs 50
--batch_size 4         # Good for ~20 training samples
--learning_rate 0.001
--device cuda
```

**Step 6 (Validation):**
```bash
--scene stump
--model_125_path output/stump  # 125-image model
--num_pairs 20
```

---

## Debugging and Validation

### Check Pipeline Outputs
```bash
# Stage 1 output
cat data/processed/{scene}/splits/train_indices.json

# Stage 2 output
ls -lh data/processed/{scene}/3dgs_output/point_cloud/iteration_30000/

# Stage 3 output
ls data/processed/{scene}/rendered_views/ | wc -l

# Stage 4 output
tree data/processed/{scene}/dataset_pairs/ -L 2

# Stage 5 output
ls experiments/{scene}/checkpoints/
```

### Visualize Intermediate Results
```python
from PIL import Image
import matplotlib.pyplot as plt

# Load a data pair
pair_dir = "data/processed/scene_001/dataset_pairs/train/pair_0000"
rendered = Image.open(f"{pair_dir}/rendered.png")
gt = Image.open(f"{pair_dir}/gt.png")
mask = Image.open(f"{pair_dir}/mask.png")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rendered); axes[0].set_title('Rendered')
axes[1].imshow(gt); axes[1].set_title('Ground Truth')
axes[2].imshow(mask, cmap='hot'); axes[2].set_title('Noise Mask')
plt.show()
```

---

## Common Issues and Solutions

### ❌ Issue 1: Step 2 - FileNotFoundError for images
**Symptom**: `train.py` tries to load 125 images but only 93 exist
```
FileNotFoundError: /data/processed/stump/images/_DSC9334.JPG
```
**Cause**: COLMAP sparse was not filtered in Step 1
**Solution**: Make sure you're using updated `step1_split.py` that filters COLMAP sparse

### ❌ Issue 2: Step 4 - Training masks too sparse (1-3% noise)
**Symptom**:
```json
{"noise_ratio": 0.01414}  // Only 1.4% noise!
```
**Cause**: `threshold=0.03` is TOO HIGH (filters out most errors)
**Solution**: Use `--threshold 0.15` or `0.2`
```bash
python src/step4_generate_pairs.py --scene stump --threshold 0.15 --min_size 30
```
**Expected**: noise_ratio should be 15-30%

### ❌ Issue 3: Step 6 - Detector predicts 50%+ noise everywhere
**Symptom**:
```json
{"mean_noise_ratio": 0.5397}  // Predicting 54% noise!
```
**Cause**: Model trained on wrong data (1-3% noise) → learned inverted pattern
**Root cause**: Step 4 threshold was too high
**Solution**:
1. Re-run Step 4 with correct threshold (0.15-0.2)
2. Re-train detector in Step 5
3. Re-validate in Step 6

### ❌ Issue 4: Step 4 - LPIPS spatial map error
**Symptom**: `IndexError: tuple index out of range` in `_compute_lpips`
**Cause**: This project's LPIPS returns scalar, not spatial map
**Solution**: Already fixed - LPIPS creates uniform map with scalar value

### ⚠️ Issue 5: Step 2 - Port already in use
**Symptom**: `Address already in use` during training
**Cause**: Previous training still running with network viewer
**Solution**: Already fixed - added `--disable_viewer` flag

### 💡 Best Practice: Validate Each Stage

**After Step 1:**
```bash
ls data/processed/stump/images/ | wc -l  # Should be 93
cat data/processed/stump/splits/test_cameras.json | jq length  # Should be 32
```

**After Step 3:**
```bash
ls data/processed/stump/rendered_views/ | wc -l  # Should be ~50 (25 render + 25 gt)
```

**After Step 4:**
```bash
# Check noise ratios
find data/processed/stump/dataset_pairs/train -name metadata.json -exec cat {} \; | grep noise_ratio
# Should see values like 0.15-0.30 (15-30%)
```

**After Step 5:**
```bash
# Check if model saved
ls experiments/stump/checkpoints/
# Should see best_model.pth
```

**After Step 6:**
```bash
# Check validation results
cat data/processed/stump/interpolated_results/summary.json | jq '.statistics'
# mean_noise_ratio should be 20-40% for good detector
```

---

## Testing and Validation

### Pipeline Validation
Each stage should be validated before proceeding:
- **Stage 1**:
  - Check: 93 train images, 32 test cameras
  - Verify: `sparse/0/` only contains 93 camera entries
  - Check: `test_cameras.json` has 32 entries
- **Stage 2**:
  - Verify: `point_cloud/iteration_30000/point_cloud.ply` exists
  - Check: Training used only 93 images (in log)
- **Stage 3**:
  - Check: ~48-50 files (24-25 render + 24-25 GT)
  - Visually inspect: rendered images should have some artifacts
- **Stage 4**:
  - **CRITICAL**: Check noise_ratio in metadata.json
  - Should be: 15-30% (NOT 1-3%!)
  - If wrong: Adjust threshold and re-run
- **Stage 5**:
  - Check: best_model.pth saved
  - Typical final IoU: 0.3-0.5 (normal for noisy masks)
- **Stage 6**:
  - Check: mean_noise_ratio in summary.json
  - Should be: 20-40% for well-trained detector
  - If 50%+: Re-run from Step 4 with higher threshold

### Quantitative Metrics
```bash
# Check 3DGS rendering quality (93-image model)
python metrics.py -m data/processed/stump/3dgs_output/

# Check noise detection statistics (Step 6)
cat data/processed/stump/interpolated_results/summary.json | jq '.statistics'
```

---

## 🎯 Key Lessons Learned

### 1. Experimental Design
**Problem**: How to ensure test views are truly novel (unseen during training)?
**Solution**: COLMAP sparse filtering in Step 1
- Filter `images.bin` to only train cameras
- Filter `points3D.bin` to only train-observed points
- Save test camera poses separately
- **Result**: 3DGS training has ZERO information leak

### 2. Hyperparameter Sensitivity
**Problem**: Model predicted 50%+ noise everywhere (inverted learning)
**Root Cause**: Training data had only 1-3% noise (threshold=0.03 too high)
**Solution**: Threshold 0.15-0.2 to get 15-30% noise in training
**Lesson**: Check training data distribution BEFORE training detector!

### 3. LPIPS Implementation Variance
**Problem**: IndexError when computing LPIPS spatial map
**Discovery**: This project's LPIPS returns scalar, not per-pixel map
**Solution**: Create uniform map filled with scalar value
**Lesson**: Don't assume all LPIPS implementations are identical

### 4. Two-Model Strategy
**Why**: Need both 93-image and 125-image models
- **93-image**: Generate training data with artifacts
- **125-image**: Validate detector on high-quality interpolated views
**Lesson**: Validation data should be different from training scenario

### 5. Batch Size vs Dataset Size
**Observation**: Dataset is small (~20 training pairs)
**Finding**: batch_size=4 works well, larger doesn't help
**Lesson**: Batch size should match dataset characteristics

### 6. Class Imbalance Handling
**Problem**: Noise masks highly imbalanced (1-3% vs 97-99%)
**Mistake**: Training on such imbalanced data caused inverted learning
**Solution**: Adjust threshold to get more balanced data (15-30% noise)
**Lesson**: Class balance matters even for pixel-level tasks

### 7. Modular Pipeline Benefits
**Achievement**: Successfully completed 6-stage pipeline
**Benefit**: Can iterate on Step 4-6 without re-running Steps 1-3
**Example**: Tuning threshold doesn't require retraining 3DGS
**Lesson**: Checkpointing at each stage enables rapid experimentation

---

## Important Base 3DGS Parameters

### Optimization Parameters
- `--iterations`: Total training iterations (default: 30000)
- `--position_lr_init`: Initial position learning rate (0.00016)
- `--densify_grad_threshold`: Densification gradient threshold (0.0002)
- `--lambda_dssim`: SSIM loss weight (0.2)

### Memory Optimization
- `--data_device`: Store data on CPU to save VRAM
- `--test_iterations -1`: Disable test evaluations during training

---

## References

- **Design Document**: See `claude_modular_pipeline.md` for detailed pipeline design
- **Base 3DGS**: Original implementation from [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Paper**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

---

## Development Priorities

### Current Status ✅ COMPLETED
- ✅ Base 3DGS system operational
- ✅ Pipeline design documented
- ✅ **All 6 stage implementation scripts completed** (src/)
- ✅ **COLMAP filtering for strict isolation** (Step 1)
- ✅ **Custom rendering utilities** (Step 3)
- ✅ **Detection model implementation** (U-Net)
- ✅ **Interpolated view validation** (Step 6)
- ✅ Evaluation and visualization tools

### Next Steps: Hyperparameter Optimization
1. **Step 4 threshold tuning**: Find optimal threshold (0.15-0.25 range)
   - Goal: Training noise_ratio = 15-30%
   - Use `analyze_threshold.py` to visualize error distributions

2. **Step 4 min_size tuning**: Adjust small region filtering (20-50 pixels)
   - Smaller → more detailed masks
   - Larger → only major artifacts

3. **Step 5 architecture experiments**:
   - Try deeper U-Net
   - Adjust image_size (256 vs 512)
   - Experiment with loss weights

4. **Multi-scene validation**: Test on other Mip-NeRF 360 scenes
   - bicycle, garden, flowers, treehill, etc.

5. **Ablation studies**:
   - L1 only vs L1+LPIPS
   - Different train/test split ratios
   - Impact of COLMAP filtering on performance

---

## Notes for Claude Code

### Critical Knowledge for Future Sessions

1. **COLMAP Filtering is Essential** (Step 1):
   - DO NOT just copy sparse directory
   - MUST filter to only train cameras and points3D
   - This ensures strict experimental isolation
   - Without this, Step 2 will fail (tries to load missing images)

2. **Threshold is the Most Critical Hyperparameter** (Step 4):
   - Default threshold=0.03 in code is **WRONG** (too high)
   - ALWAYS use 0.15-0.2 for proper training data
   - Check noise_ratio after Step 4: should be 15-30%, not 1-3%
   - If detector predicts 50%+ noise, threshold was wrong in Step 4

3. **Two 3DGS Models Required**:
   - **93-image model**: `data/processed/stump/3dgs_output/` (for training)
   - **125-image model**: `output/stump/` (for validation in Step 6)
   - These serve different purposes in the pipeline

4. **File Naming Matters**:
   - Step 3: Use `step3_render_views_v2.py` (NOT step3_render_views.py)
   - Outputs: `render_*.png` and `gt_*.png`

5. **LPIPS Implementation Detail**:
   - This project's LPIPS returns scalar, not spatial map
   - Creates uniform map in `_compute_lpips` - this is intentional
   - Combined error = 0.5 * L1_norm + 0.5 * LPIPS_norm

6. **Batch Size Consideration**:
   - Dataset is small (~20 training pairs)
   - batch_size=4 is good default
   - Larger batch sizes won't help much

7. **Validation Strategy**:
   - Always validate each stage before proceeding
   - Use commands in "Best Practice: Validate Each Stage"
   - If Step 6 shows bad results, problem is likely in Step 4

8. **Data Paths**:
   - Input: `data/360_v2/{scene}/`
   - Processing: `data/processed/{scene}/`
   - Experiments: `experiments/{scene}/`
   - 125-model: `output/{scene}/`

9. **Common User Workflow**:
   ```bash
   # Initial run
   python src/step1_split.py --scene stump --skip_frames 3
   python src/step2_train_3dgs.py --scene stump --iterations 30000
   python src/step3_render_views_v2.py --scene stump
   python src/step4_generate_pairs.py --scene stump --threshold 0.15 --min_size 30
   python src/step5_train_detector.py --scene stump --num_epochs 50
   python src/step6_interpolate_detect.py --scene stump

   # If results bad, re-run from Step 4
   python src/step4_generate_pairs.py --scene stump --threshold 0.2 --min_size 25
   python src/step5_train_detector.py --scene stump --num_epochs 50
   python src/step6_interpolate_detect.py --scene stump
   ```

10. **Success Criteria**:
    - Step 4: noise_ratio in metadata.json = 15-30%
    - Step 6: mean_noise_ratio in summary.json = 20-40%
    - If Step 6 shows 50%+, retrain from Step 4 with different threshold
