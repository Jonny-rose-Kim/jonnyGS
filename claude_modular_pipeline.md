# 3DGS Noise Detection - 모듈화된 파이프라인

## 📋 프로젝트 개요

### 목표
3DGS novel view의 노이즈를 감지하는 모델 개발 - **단계별 실행 가능한 모듈화 설계**

### 핵심 특징
- ✅ **데이터셋 독립적**: 어떤 장면 데이터도 사용 가능
- ✅ **단계별 실행**: 각 단계를 독립적으로 실행
- ✅ **체크포인트**: 각 단계의 결과물 저장
- ✅ **재시작 가능**: 특정 단계부터 다시 시작 가능

---

## 🔄 전체 파이프라인 (5단계)

```
Step 1: Train/Test Split
  입력: 원본 이미지 (N장)
  출력: train_indices.json, test_indices.json

Step 2: 3DGS Training
  입력: train_indices.json
  출력: trained_model/ (Gaussian .ply 파일)

Step 3: Novel View Rendering
  입력: trained_model/, test_indices.json
  출력: rendered_views/ (렌더링된 이미지들)

Step 4: Error Map Generation
  입력: rendered_views/, test_indices.json (GT)
  출력: dataset_pairs/ (rendered, gt, mask 쌍)

Step 5: Model Training
  입력: dataset_pairs/
  출력: trained_detector/ (노이즈 감지 모델)
```

---

## 🏗️ 프로젝트 구조

```
noise-detector/
│
├── README.md
├── requirements.txt
├── config.yaml                   # 전체 설정 파일
│
├── data/
│   ├── raw/                      # 원본 데이터 (사용자 제공)
│   │   └── scene_001/
│   │       ├── images/           # 125장 예시
│   │       │   ├── 000.jpg
│   │       │   ├── 001.jpg
│   │       │   └── ...
│   │       └── sparse/           # COLMAP 카메라 정보 (선택)
│   │
│   └── processed/                # 처리된 데이터
│       └── scene_001/
│           ├── splits/
│           │   ├── train_indices.json
│           │   └── test_indices.json
│           ├── 3dgs_output/
│           │   └── point_cloud.ply
│           ├── rendered_views/
│           │   ├── view_000.png
│           │   ├── view_004.png
│           │   └── ...
│           └── dataset_pairs/
│               ├── train/
│               ├── val/
│               └── test/
│
├── src/
│   ├── step1_split.py            # Step 1: Train/Test Split
│   ├── step2_train_3dgs.py       # Step 2: 3DGS Training
│   ├── step3_render_views.py    # Step 3: Novel View Rendering
│   ├── step4_generate_pairs.py  # Step 4: Error Map & Pairs
│   ├── step5_train_detector.py  # Step 5: Detector Training
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── colmap_utils.py
│   │   └── visualization.py
│   │
│   └── models/
│       ├── simple_unet.py
│       ├── losses.py
│       └── metrics.py
│
├── scripts/
│   ├── run_all.py               # 전체 파이프라인 실행
│   └── run_step.py              # 특정 단계만 실행
│
└── experiments/
    └── scene_001_exp1/
        ├── checkpoints/
        ├── logs/
        └── results/
```

---

## 📝 상세 구현

### Step 1: Train/Test Split

**파일: `src/step1_split.py`**

```python
"""
Step 1: Train/Test Split

입력:
  - data/raw/{scene_name}/images/
  - config.yaml (skip_frames)

출력:
  - data/processed/{scene_name}/splits/train_indices.json
  - data/processed/{scene_name}/splits/test_indices.json
"""

import json
from pathlib import Path
import argparse

def create_split(image_dir, skip_frames=3):
    """
    이미지를 train/test로 split.
    
    Args:
        image_dir: 이미지 디렉토리 경로
        skip_frames: Test로 사용할 frame 간격
    
    Returns:
        train_indices: Train 이미지 인덱스 리스트
        test_indices: Test 이미지 인덱스 리스트
    """
    # 모든 이미지 파일 찾기
    image_files = sorted(Path(image_dir).glob("*.jpg")) + \
                  sorted(Path(image_dir).glob("*.JPG")) + \
                  sorted(Path(image_dir).glob("*.png"))
    
    total_frames = len(image_files)
    print(f"총 이미지 수: {total_frames}장")
    
    # Split 생성
    test_indices = list(range(0, total_frames, skip_frames + 1))
    train_indices = [i for i in range(total_frames) 
                    if i not in test_indices]
    
    print(f"Train: {len(train_indices)}장 ({len(train_indices)/total_frames*100:.1f}%)")
    print(f"Test:  {len(test_indices)}장 ({len(test_indices)/total_frames*100:.1f}%)")
    
    return train_indices, test_indices, [f.name for f in image_files]

def save_split(output_dir, train_indices, test_indices, image_names):
    """Split 정보를 JSON으로 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train indices 저장
    train_data = {
        'indices': train_indices,
        'count': len(train_indices),
        'images': [image_names[i] for i in train_indices]
    }
    with open(output_dir / 'train_indices.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Test indices 저장
    test_data = {
        'indices': test_indices,
        'count': len(test_indices),
        'images': [image_names[i] for i in test_indices]
    }
    with open(output_dir / 'test_indices.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n✓ Split 저장 완료: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Step 1: Train/Test Split')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--data_root', default='./data', help='Data root directory')
    parser.add_argument('--skip_frames', type=int, default=3, help='Skip frames for test split')
    args = parser.parse_args()
    
    print("="*60)
    print("Step 1: Train/Test Split")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Skip frames: {args.skip_frames}")
    print("="*60 + "\n")
    
    # 경로 설정
    image_dir = Path(args.data_root) / "raw" / args.scene / "images"
    output_dir = Path(args.data_root) / "processed" / args.scene / "splits"
    
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
    
    # Split 생성
    train_indices, test_indices, image_names = create_split(
        image_dir, 
        skip_frames=args.skip_frames
    )
    
    # 저장
    save_split(output_dir, train_indices, test_indices, image_names)
    
    print("\n" + "="*60)
    print("Step 1 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
python src/step1_split.py --scene scene_001 --skip_frames 3
```

**출력:**
```
data/processed/scene_001/splits/
  ├── train_indices.json  # {'indices': [1,2,3,5,6,7,...], 'count': 94, 'images': [...]}
  └── test_indices.json   # {'indices': [0,4,8,12,...], 'count': 31, 'images': [...]}
```

---

### Step 2: 3DGS Training

**파일: `src/step2_train_3dgs.py`**

```python
"""
Step 2: 3DGS Training

입력:
  - data/raw/{scene_name}/images/
  - data/processed/{scene_name}/splits/train_indices.json

출력:
  - data/processed/{scene_name}/3dgs_output/
    ├── point_cloud.ply
    ├── cameras.json
    └── cfg_args
"""

import json
import shutil
import subprocess
from pathlib import Path
import argparse

def prepare_3dgs_data(scene_path, train_indices_file, temp_dir):
    """
    3DGS 학습을 위한 데이터 준비.
    Train 이미지만 복사.
    
    Args:
        scene_path: 원본 장면 경로
        train_indices_file: Train indices JSON 파일
        temp_dir: 임시 디렉토리
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train indices 로드
    with open(train_indices_file, 'r') as f:
        train_data = json.load(f)
    
    train_images = train_data['images']
    print(f"Train 이미지 {len(train_images)}장을 복사 중...")
    
    # 이미지 복사
    images_src = Path(scene_path) / "images"
    images_dst = temp_dir / "input"
    images_dst.mkdir(parents=True, exist_ok=True)
    
    for img_name in train_images:
        src = images_src / img_name
        dst = images_dst / img_name
        shutil.copy2(src, dst)
    
    # COLMAP sparse 모델이 있으면 복사
    sparse_src = Path(scene_path) / "sparse"
    if sparse_src.exists():
        sparse_dst = temp_dir / "sparse"
        shutil.copytree(sparse_src, sparse_dst, dirs_exist_ok=True)
        print("✓ COLMAP sparse 모델 복사됨")
    
    print(f"✓ 데이터 준비 완료: {temp_dir}")
    return temp_dir

def train_3dgs(input_dir, output_dir, iterations=30000):
    """
    3DGS 학습 실행.
    
    Args:
        input_dir: 입력 데이터 디렉토리
        output_dir: 출력 디렉토리
        iterations: 학습 iteration 수
    """
    print(f"\n3DGS 학습 시작 ({iterations} iterations)...")
    
    # gaussian-splatting train.py 호출
    cmd = [
        "python", "train.py",
        "-s", str(input_dir),
        "-m", str(output_dir),
        "--iterations", str(iterations),
        "--test_iterations", "-1",  # Test evaluation 비활성화
        "--save_iterations", str(iterations)
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    
    try:
        # gaussian-splatting 디렉토리에서 실행한다고 가정
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="./gaussian-splatting"  # gaussian-splatting 설치 경로
        )
        print(result.stdout)
        print("\n✓ 3DGS 학습 완료")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 3DGS 학습 실패:")
        print(e.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description='Step 2: 3DGS Training')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--data_root', default='./data', help='Data root directory')
    parser.add_argument('--iterations', type=int, default=30000, help='Training iterations')
    parser.add_argument('--gaussian_splatting_path', default='./gaussian-splatting', 
                       help='Path to gaussian-splatting repository')
    args = parser.parse_args()
    
    print("="*60)
    print("Step 2: 3DGS Training")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Iterations: {args.iterations}")
    print("="*60 + "\n")
    
    # 경로 설정
    scene_path = Path(args.data_root) / "raw" / args.scene
    train_indices_file = (Path(args.data_root) / "processed" / args.scene / 
                         "splits" / "train_indices.json")
    temp_dir = Path(args.data_root) / "processed" / args.scene / "3dgs_temp"
    output_dir = Path(args.data_root) / "processed" / args.scene / "3dgs_output"
    
    # 검증
    if not train_indices_file.exists():
        raise FileNotFoundError(
            f"Train indices 파일을 찾을 수 없습니다: {train_indices_file}\n"
            "먼저 Step 1을 실행하세요."
        )
    
    # 1. 데이터 준비
    input_dir = prepare_3dgs_data(scene_path, train_indices_file, temp_dir)
    
    # 2. 3DGS 학습
    train_3dgs(input_dir, output_dir, iterations=args.iterations)
    
    # 3. 정리
    print(f"\n출력 저장 위치: {output_dir}")
    print(f"임시 파일은 유지됩니다: {temp_dir}")
    
    print("\n" + "="*60)
    print("Step 2 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
python src/step2_train_3dgs.py --scene scene_001 --iterations 30000
```

**출력:**
```
data/processed/scene_001/3dgs_output/
  ├── point_cloud/
  │   └── iteration_30000/
  │       └── point_cloud.ply
  ├── cameras.json
  └── cfg_args
```

---

### Step 3: Novel View Rendering

**파일: `src/step3_render_views.py`**

```python
"""
Step 3: Novel View Rendering

입력:
  - data/processed/{scene_name}/3dgs_output/
  - data/processed/{scene_name}/splits/test_indices.json
  - data/raw/{scene_name}/sparse/ (카메라 정보)

출력:
  - data/processed/{scene_name}/rendered_views/
    ├── view_0000.png
    ├── view_0004.png
    └── ...
"""

import json
import subprocess
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def load_camera_info(scene_path, test_indices_file):
    """
    Test view의 카메라 정보 로드.
    
    Args:
        scene_path: 장면 경로
        test_indices_file: Test indices JSON
    
    Returns:
        cameras: 카메라 정보 리스트
    """
    # Test indices 로드
    with open(test_indices_file, 'r') as f:
        test_data = json.load(f)
    
    test_indices = test_data['indices']
    
    # COLMAP 또는 transforms.json에서 카메라 로드
    # 여기서는 간단히 인덱스만 반환
    # 실제로는 COLMAP read_model이나 JSON 파싱 필요
    
    cameras = []
    for idx in test_indices:
        cameras.append({
            'index': idx,
            'image_name': test_data['images'][test_indices.index(idx)]
        })
    
    return cameras

def render_single_view(model_path, camera_info, output_path):
    """
    단일 view 렌더링.
    
    Args:
        model_path: 학습된 3DGS 모델 경로
        camera_info: 카메라 정보
        output_path: 출력 이미지 경로
    """
    # gaussian-splatting render.py 호출
    cmd = [
        "python", "render.py",
        "-m", str(model_path),
        "--skip_train",  # Train view는 렌더링 안 함
        "--skip_test"    # 우리가 직접 지정한 view만
    ]
    
    # 실제로는 카메라 파라미터를 전달해야 함
    # 여기서는 간단히 표현
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="./gaussian-splatting"
        )
        
        # 렌더링된 이미지를 output_path로 복사
        # (실제로는 gaussian-splatting의 출력 위치에서 가져옴)
        
    except subprocess.CalledProcessError as e:
        print(f"렌더링 실패: {e.stderr}")
        raise

def render_all_views(model_path, cameras, output_dir):
    """
    모든 test view 렌더링.
    
    Args:
        model_path: 3DGS 모델 경로
        cameras: 카메라 정보 리스트
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{len(cameras)}개 view 렌더링 시작...")
    
    for i, cam in enumerate(cameras):
        output_path = output_dir / f"view_{cam['index']:04d}.png"
        
        print(f"  [{i+1}/{len(cameras)}] Rendering view {cam['index']}...")
        
        # 렌더링 실행
        render_single_view(model_path, cam, output_path)
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ {i+1}/{len(cameras)} 완료")
    
    print(f"\n✓ 모든 view 렌더링 완료: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Step 3: Novel View Rendering')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--data_root', default='./data', help='Data root directory')
    args = parser.parse_args()
    
    print("="*60)
    print("Step 3: Novel View Rendering")
    print("="*60)
    print(f"Scene: {args.scene}")
    print("="*60 + "\n")
    
    # 경로 설정
    scene_path = Path(args.data_root) / "raw" / args.scene
    model_path = (Path(args.data_root) / "processed" / args.scene / 
                 "3dgs_output")
    test_indices_file = (Path(args.data_root) / "processed" / args.scene / 
                        "splits" / "test_indices.json")
    output_dir = (Path(args.data_root) / "processed" / args.scene / 
                 "rendered_views")
    
    # 검증
    if not model_path.exists():
        raise FileNotFoundError(
            f"3DGS 모델을 찾을 수 없습니다: {model_path}\n"
            "먼저 Step 2를 실행하세요."
        )
    
    if not test_indices_file.exists():
        raise FileNotFoundError(
            f"Test indices 파일을 찾을 수 없습니다: {test_indices_file}\n"
            "먼저 Step 1을 실행하세요."
        )
    
    # 1. 카메라 정보 로드
    cameras = load_camera_info(scene_path, test_indices_file)
    print(f"✓ {len(cameras)}개 test view 카메라 로드됨")
    
    # 2. 렌더링
    render_all_views(model_path, cameras, output_dir)
    
    print("\n" + "="*60)
    print("Step 3 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
python src/step3_render_views.py --scene scene_001
```

**출력:**
```
data/processed/scene_001/rendered_views/
  ├── view_0000.png  # Test index 0의 렌더링
  ├── view_0004.png  # Test index 4의 렌더링
  ├── view_0008.png
  └── ...
```

---

### Step 4: Error Map & Dataset Pairs Generation

**파일: `src/step4_generate_pairs.py`**

```python
"""
Step 4: Error Map & Dataset Pairs Generation

입력:
  - data/processed/{scene_name}/rendered_views/
  - data/raw/{scene_name}/images/ (GT)
  - data/processed/{scene_name}/splits/test_indices.json

출력:
  - data/processed/{scene_name}/dataset_pairs/
    ├── train/
    ├── val/
    └── test/
"""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import argparse
import lpips

class ErrorMapGenerator:
    """
    렌더링 이미지와 GT를 비교하여 error map (noise mask) 생성
    """
    
    def __init__(self, threshold=0.15, min_size=50):
        self.threshold = threshold
        self.min_size = min_size
        
        # LPIPS 모델 초기화
        self.lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            self.lpips_model = self.lpips_model.cuda()
        self.lpips_model.eval()
    
    def generate_mask(self, rendered, gt):
        """
        Error map 생성.
        
        Args:
            rendered: 렌더링 이미지 [H, W, 3], numpy, [0, 255]
            gt: Ground truth [H, W, 3], numpy, [0, 255]
        
        Returns:
            mask: Binary noise mask [H, W], float32, {0, 1}
        """
        # 1. Pixel-wise L1 error
        pixel_error = np.abs(rendered.astype(np.float32) - 
                           gt.astype(np.float32)).mean(axis=-1)
        pixel_error_norm = self._normalize(pixel_error)
        
        # 2. LPIPS error
        lpips_error = self._compute_lpips(rendered, gt)
        lpips_error_norm = self._normalize(lpips_error)
        
        # 3. 결합
        combined_error = 0.5 * pixel_error_norm + 0.5 * lpips_error_norm
        
        # 4. Binary mask
        mask = (combined_error > self.threshold).astype(np.float32)
        
        # 5. 후처리
        mask = self._remove_small_regions(mask)
        
        return mask
    
    def _compute_lpips(self, img1, img2):
        """LPIPS 계산"""
        def to_tensor(img):
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = (img / 127.5) - 1.0
            if torch.cuda.is_available():
                img = img.cuda()
            return img
        
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        
        with torch.no_grad():
            lpips_map = self.lpips_model(t1, t2, normalize=False)
            lpips_map = lpips_map.squeeze().cpu().numpy()
        
        return lpips_map
    
    def _normalize(self, arr):
        """정규화"""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min < 1e-7:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    def _remove_small_regions(self, mask):
        """작은 영역 제거"""
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        component_sizes = np.bincount(labeled.ravel())
        mask_sizes = component_sizes >= self.min_size
        mask_sizes[0] = 0
        cleaned_mask = mask_sizes[labeled]
        return cleaned_mask.astype(np.float32)

def generate_all_pairs(rendered_dir, images_dir, test_indices_file, 
                      output_dir, error_generator):
    """
    모든 데이터 쌍 생성.
    
    Args:
        rendered_dir: 렌더링된 이미지 디렉토리
        images_dir: 원본 이미지 디렉토리 (GT)
        test_indices_file: Test indices JSON
        output_dir: 출력 디렉토리
        error_generator: ErrorMapGenerator 인스턴스
    """
    # Test indices 로드
    with open(test_indices_file, 'r') as f:
        test_data = json.load(f)
    
    test_indices = test_data['indices']
    test_images = test_data['images']
    
    print(f"\n{len(test_indices)}개 데이터 쌍 생성 중...")
    
    pairs = []
    
    for i, (idx, img_name) in enumerate(zip(test_indices, test_images)):
        # 렌더링 이미지 로드
        rendered_path = Path(rendered_dir) / f"view_{idx:04d}.png"
        if not rendered_path.exists():
            print(f"  ⚠ 렌더링 이미지 없음: {rendered_path}")
            continue
        
        rendered = np.array(Image.open(rendered_path).convert('RGB'))
        
        # GT 이미지 로드
        gt_path = Path(images_dir) / img_name
        gt = np.array(Image.open(gt_path).convert('RGB'))
        
        # Error map 생성
        mask = error_generator.generate_mask(rendered, gt)
        
        # 저장할 정보
        pair = {
            'view_index': idx,
            'image_name': img_name,
            'rendered': rendered,
            'gt': gt,
            'mask': mask,
            'noise_ratio': float(mask.mean())
        }
        pairs.append(pair)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_indices)}] 완료")
    
    print(f"\n✓ {len(pairs)}개 쌍 생성 완료")
    
    # Train/Val/Test split (80/10/10)
    save_pairs_with_split(pairs, output_dir)

def save_pairs_with_split(pairs, output_dir):
    """
    생성된 쌍을 train/val/test로 나누어 저장.
    
    Args:
        pairs: 데이터 쌍 리스트
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    
    for i, pair in enumerate(pairs):
        # Split 결정 (80/10/10)
        mod = i % 10
        if mod < 8:
            split = 'train'
        elif mod == 8:
            split = 'val'
        else:
            split = 'test'
        
        # 저장 경로
        save_dir = output_dir / split / f"pair_{pair['view_index']:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 저장
        Image.fromarray(pair['rendered']).save(save_dir / "rendered.png")
        Image.fromarray(pair['gt']).save(save_dir / "gt.png")
        Image.fromarray((pair['mask'] * 255).astype(np.uint8)).save(
            save_dir / "mask.png"
        )
        
        # 메타데이터 저장
        metadata = {
            'view_index': pair['view_index'],
            'image_name': pair['image_name'],
            'noise_ratio': pair['noise_ratio']
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # 통계 출력
    stats = {
        'train': len([p for i, p in enumerate(pairs) if i % 10 < 8]),
        'val': len([p for i, p in enumerate(pairs) if i % 10 == 8]),
        'test': len([p for i, p in enumerate(pairs) if i % 10 == 9])
    }
    
    print(f"\n저장 완료:")
    print(f"  Train: {stats['train']}개")
    print(f"  Val:   {stats['val']}개")
    print(f"  Test:  {stats['test']}개")
    print(f"  위치: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Step 4: Generate Dataset Pairs')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--data_root', default='./data', help='Data root directory')
    parser.add_argument('--threshold', type=float, default=0.15, 
                       help='Error threshold for noise detection')
    args = parser.parse_args()
    
    print("="*60)
    print("Step 4: Error Map & Dataset Pairs Generation")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Threshold: {args.threshold}")
    print("="*60 + "\n")
    
    # 경로 설정
    rendered_dir = (Path(args.data_root) / "processed" / args.scene / 
                   "rendered_views")
    images_dir = Path(args.data_root) / "raw" / args.scene / "images"
    test_indices_file = (Path(args.data_root) / "processed" / args.scene / 
                        "splits" / "test_indices.json")
    output_dir = (Path(args.data_root) / "processed" / args.scene / 
                 "dataset_pairs")
    
    # 검증
    if not rendered_dir.exists():
        raise FileNotFoundError(
            f"렌더링 이미지를 찾을 수 없습니다: {rendered_dir}\n"
            "먼저 Step 3을 실행하세요."
        )
    
    # Error map generator 생성
    error_generator = ErrorMapGenerator(threshold=args.threshold)
    
    # 데이터 쌍 생성
    generate_all_pairs(
        rendered_dir,
        images_dir,
        test_indices_file,
        output_dir,
        error_generator
    )
    
    print("\n" + "="*60)
    print("Step 4 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
python src/step4_generate_pairs.py --scene scene_001 --threshold 0.15
```

**출력:**
```
data/processed/scene_001/dataset_pairs/
  ├── train/
  │   ├── pair_0000/
  │   │   ├── rendered.png
  │   │   ├── gt.png
  │   │   ├── mask.png
  │   │   └── metadata.json
  │   ├── pair_0008/
  │   └── ...
  ├── val/
  │   └── ...
  └── test/
      └── ...
```

---

### Step 5: Detector Training

**파일: `src/step5_train_detector.py`**

```python
"""
Step 5: Noise Detector Training

입력:
  - data/processed/{scene_name}/dataset_pairs/

출력:
  - experiments/{scene_name}/
    ├── checkpoints/
    │   └── best_model.pth
    ├── logs/
    └── results/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# 간단한 Dataset 클래스
class NoisePairDataset(Dataset):
    """데이터 쌍 로드용 Dataset"""
    
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root) / split
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for pair_dir in sorted(self.data_root.iterdir()):
            if not pair_dir.is_dir():
                continue
            
            rendered_path = pair_dir / "rendered.png"
            mask_path = pair_dir / "mask.png"
            
            if rendered_path.exists() and mask_path.exists():
                samples.append({
                    'rendered': rendered_path,
                    'mask': mask_path
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드 및 전처리
        rendered = Image.open(sample['rendered']).convert('RGB')
        rendered = rendered.resize((512, 512))
        rendered = np.array(rendered).astype(np.float32) / 255.0
        rendered = torch.from_numpy(rendered).permute(2, 0, 1)  # [3, H, W]
        
        # 마스크 로드
        mask = Image.open(sample['mask']).convert('L')
        mask = mask.resize((512, 512))
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return rendered, mask

# 간단한 U-Net (이전과 동일)
from src.models.simple_unet import SimpleUNet

# Loss와 Metrics
from src.models.losses import CombinedLoss
from src.models.metrics import compute_metrics

def train_model(train_loader, val_loader, num_epochs=50, device='cuda'):
    """
    모델 학습.
    
    Args:
        train_loader: Train DataLoader
        val_loader: Validation DataLoader
        num_epochs: 학습 epoch 수
        device: 학습 device
    
    Returns:
        model: 학습된 모델
        best_iou: 최고 IoU
    """
    # 모델 생성
    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    best_iou = 0.0
    
    print("\n학습 시작...")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            pred_masks = model(images)
            loss = criterion(pred_masks, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'f1': 0.0}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                pred_masks = model(images)
                loss = criterion(pred_masks, masks)
                
                val_loss += loss.item()
                
                metrics = compute_metrics(pred_masks, masks)
                val_metrics['iou'] += metrics['iou']
                val_metrics['f1'] += metrics['f1']
        
        val_loss /= len(val_loader)
        val_metrics['iou'] /= len(val_loader)
        val_metrics['f1'] /= len(val_loader)
        
        # LR Scheduling
        scheduler.step(val_loss)
        
        # 출력
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f}")
        
        # Best model 저장
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            # 저장 로직은 main()에서 처리
    
    return model, best_iou

def main():
    parser = argparse.ArgumentParser(description='Step 5: Train Noise Detector')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--data_root', default='./data', help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Step 5: Noise Detector Training")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print("="*60 + "\n")
    
    # 경로 설정
    data_root = (Path(args.data_root) / "processed" / args.scene / 
                "dataset_pairs")
    checkpoint_dir = Path("experiments") / args.scene / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 검증
    if not data_root.exists():
        raise FileNotFoundError(
            f"데이터 쌍을 찾을 수 없습니다: {data_root}\n"
            "먼저 Step 4를 실행하세요."
        )
    
    # 데이터셋 로드
    print("데이터 로드 중...")
    train_dataset = NoisePairDataset(data_root, split='train')
    val_dataset = NoisePairDataset(data_root, split='val')
    
    print(f"✓ Train: {len(train_dataset)} 샘플")
    print(f"✓ Val: {len(val_dataset)} 샘플")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 학습
    model, best_iou = train_model(
        train_loader, 
        val_loader, 
        num_epochs=args.num_epochs,
        device=device
    )
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'iou': best_iou
    }, checkpoint_dir / 'best_model.pth')
    
    print("\n" + "="*60)
    print(f"Step 5 완료! Best IoU: {best_iou:.4f}")
    print(f"모델 저장 위치: {checkpoint_dir / 'best_model.pth'}")
    print("="*60)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
python src/step5_train_detector.py --scene scene_001 --num_epochs 50
```

**출력:**
```
experiments/scene_001/
  └── checkpoints/
      └── best_model.pth
```

---

## 🚀 통합 실행 스크립트

### 전체 파이프라인 실행

**파일: `scripts/run_all.py`**

```python
"""
전체 파이프라인 한 번에 실행
"""

import subprocess
import argparse
from pathlib import Path

def run_step(step_name, script_path, args):
    """단일 스텝 실행"""
    print("\n" + "="*70)
    print(f"{'':^70}")
    print(f"{step_name:^70}")
    print(f"{'':^70}")
    print("="*70 + "\n")
    
    cmd = ["python", script_path] + args
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} 실패!")
    
    print(f"\n✓ {step_name} 완료\n")

def main():
    parser = argparse.ArgumentParser(description='Run Full Pipeline')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--skip_frames', type=int, default=3, help='Skip frames')
    parser.add_argument('--iterations', type=int, default=30000, help='3DGS iterations')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    args = parser.parse_args()
    
    print("="*70)
    print(f"{'3DGS Noise Detection - Full Pipeline':^70}")
    print("="*70)
    print(f"Scene: {args.scene}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"3DGS iterations: {args.iterations}")
    print(f"Training epochs: {args.epochs}")
    print("="*70)
    
    # Step 1: Split
    run_step(
        "Step 1: Train/Test Split",
        "src/step1_split.py",
        ["--scene", args.scene, "--skip_frames", str(args.skip_frames)]
    )
    
    # Step 2: 3DGS Training
    run_step(
        "Step 2: 3DGS Training",
        "src/step2_train_3dgs.py",
        ["--scene", args.scene, "--iterations", str(args.iterations)]
    )
    
    # Step 3: Rendering
    run_step(
        "Step 3: Novel View Rendering",
        "src/step3_render_views.py",
        ["--scene", args.scene]
    )
    
    # Step 4: Dataset Generation
    run_step(
        "Step 4: Dataset Pairs Generation",
        "src/step4_generate_pairs.py",
        ["--scene", args.scene]
    )
    
    # Step 5: Model Training
    run_step(
        "Step 5: Detector Training",
        "src/step5_train_detector.py",
        ["--scene", args.scene, "--num_epochs", str(args.epochs)]
    )
    
    print("\n" + "="*70)
    print(f"{'ALL STEPS COMPLETED!':^70}")
    print("="*70)

if __name__ == '__main__':
    main()
```

**실행:**
```bash
python scripts/run_all.py --scene scene_001 --skip_frames 3 --iterations 30000 --epochs 50
```

---

### 특정 단계만 실행

**파일: `scripts/run_step.py`**

```python
"""
특정 단계만 선택하여 실행
"""

import subprocess
import argparse

STEPS = {
    '1': ('src/step1_split.py', 'Train/Test Split'),
    '2': ('src/step2_train_3dgs.py', '3DGS Training'),
    '3': ('src/step3_render_views.py', 'Novel View Rendering'),
    '4': ('src/step4_generate_pairs.py', 'Dataset Pairs Generation'),
    '5': ('src/step5_train_detector.py', 'Detector Training')
}

def main():
    parser = argparse.ArgumentParser(description='Run Specific Step')
    parser.add_argument('--step', required=True, choices=['1','2','3','4','5'],
                       help='Step number to run')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--skip_frames', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    script_path, step_name = STEPS[args.step]
    
    print(f"\n실행: Step {args.step} - {step_name}")
    print(f"Scene: {args.scene}\n")
    
    # 공통 인자
    cmd = ["python", script_path, "--scene", args.scene]
    
    # 단계별 추가 인자
    if args.step == '1':
        cmd += ["--skip_frames", str(args.skip_frames)]
    elif args.step == '2':
        cmd += ["--iterations", str(args.iterations)]
    elif args.step == '5':
        cmd += ["--num_epochs", str(args.epochs)]
    
    # 실행
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
```

**실행 예시:**
```bash
# Step 1만 실행
python scripts/run_step.py --step 1 --scene scene_001 --skip_frames 3

# Step 4만 실행 (Step 1-3이 이미 완료된 경우)
python scripts/run_step.py --step 4 --scene scene_001

# Step 5만 실행 (재학습)
python scripts/run_step.py --step 5 --scene scene_001 --epochs 100
```

---

## 📊 사용 시나리오

### 시나리오 1: 처음부터 끝까지

```bash
# 한 번에 전체 실행
python scripts/run_all.py --scene my_scene --skip_frames 3 --iterations 30000 --epochs 50
```

### 시나리오 2: 단계별 실행 및 검증

```bash
# Step 1: Split 생성
python scripts/run_step.py --step 1 --scene my_scene --skip_frames 3

# 결과 확인
ls data/processed/my_scene/splits/

# Step 2: 3DGS 학습
python scripts/run_step.py --step 2 --scene my_scene --iterations 30000

# 결과 확인
ls data/processed/my_scene/3dgs_output/

# Step 3: 렌더링
python scripts/run_step.py --step 3 --scene my_scene

# 결과 확인
ls data/processed/my_scene/rendered_views/

# Step 4: 데이터 쌍 생성
python scripts/run_step.py --step 4 --scene my_scene

# 결과 확인
ls data/processed/my_scene/dataset_pairs/train/

# Step 5: 모델 학습
python scripts/run_step.py --step 5 --scene my_scene --epochs 50
```

### 시나리오 3: 재학습

```bash
# 데이터는 그대로, 다른 하이퍼파라미터로 재학습
python scripts/run_step.py --step 5 --scene my_scene --epochs 100
```

### 시나리오 4: 다른 skip_frames 실험

```bash
# skip_frames=5로 새로 생성
python scripts/run_step.py --step 1 --scene my_scene --skip_frames 5
python scripts/run_step.py --step 2 --scene my_scene
python scripts/run_step.py --step 3 --scene my_scene
python scripts/run_step.py --step 4 --scene my_scene
python scripts/run_step.py --step 5 --scene my_scene
```

---

## 🔍 디버깅 가이드

### 각 단계 출력 확인

```bash
# Step 1 출력 확인
cat data/processed/scene_001/splits/train_indices.json

# Step 2 출력 확인
ls -lh data/processed/scene_001/3dgs_output/point_cloud/iteration_30000/

# Step 3 출력 확인
ls data/processed/scene_001/rendered_views/ | wc -l

# Step 4 출력 확인
tree data/processed/scene_001/dataset_pairs/ -L 2

# Step 5 출력 확인
ls experiments/scene_001/checkpoints/
```

### 중간 단계 시각화

```python
# rendered vs GT 비교
from PIL import Image
import matplotlib.pyplot as plt

rendered = Image.open('data/processed/scene_001/rendered_views/view_0000.png')
gt = Image.open('data/raw/scene_001/images/000.jpg')
mask = Image.open('data/processed/scene_001/dataset_pairs/train/pair_0000/mask.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rendered); axes[0].set_title('Rendered')
axes[1].imshow(gt); axes[1].set_title('Ground Truth')
axes[2].imshow(mask, cmap='hot'); axes[2].set_title('Noise Mask')
plt.show()
```

---

## 📋 체크리스트

### 실행 전 준비
- [ ] gaussian-splatting 설치 및 경로 확인
- [ ] 원본 데이터 준비 (`data/raw/{scene}/images/`)
- [ ] 카메라 정보 확인 (`sparse/` 또는 `transforms.json`)
- [ ] 필요 패키지 설치 (`requirements.txt`)

### Step별 검증
- [ ] Step 1: `train_indices.json`, `test_indices.json` 생성 확인
- [ ] Step 2: `point_cloud.ply` 생성 확인
- [ ] Step 3: 렌더링 이미지 개수 = test 개수 확인
- [ ] Step 4: train/val/test 비율 확인 (80/10/10)
- [ ] Step 5: IoU > 0.6 달성 확인

---

## 🎯 핵심 개선사항

### 기존 문서 대비
1. ✅ **모듈화**: 5개 독립 스크립트
2. ✅ **일반화**: 특정 데이터셋 비의존
3. ✅ **체크포인트**: 각 단계 결과물 저장
4. ✅ **재시작 가능**: 특정 단계부터 재실행
5. ✅ **명확한 입출력**: 각 단계의 I/O 명시

### 실용성
- 각 단계 독립 실행으로 빠른 디버깅
- 중간 결과물 확인 가능
- 다양한 하이퍼파라미터 실험 용이
- 에러 발생 시 해당 단계만 재실행

---

이 문서로 Claude Code에서 단계별로 구현하시면 됩니다!
