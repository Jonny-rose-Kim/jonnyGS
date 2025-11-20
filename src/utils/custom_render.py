"""
Custom rendering utility for specific camera indices.
"""

import torch
from scene import Scene
from pathlib import Path
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from gaussian_renderer import GaussianModel


def render_specific_views(model_path, source_path, test_indices_file, output_dir,
                          sh_degree=3, white_background=False):
    """
    특정 인덱스의 카메라 뷰만 렌더링.

    Args:
        model_path: 학습된 3DGS 모델 경로
        source_path: 원본 데이터 경로 (COLMAP sparse가 있는 곳)
        test_indices_file: Test indices JSON 파일
        output_dir: 렌더링 결과 저장 디렉토리
        sh_degree: Spherical harmonics degree
        white_background: 흰 배경 사용 여부

    Returns:
        output_dir: 렌더링 결과 디렉토리
    """
    # Test indices 로드
    with open(test_indices_file, 'r') as f:
        test_data = json.load(f)

    test_indices = set(test_data['indices'])

    print(f"\n렌더링할 카메라: {len(test_indices)}개 (indices: {sorted(list(test_indices))[:10]}...)")

    # ModelParams와 PipelineParams를 딕셔너리로 생성
    class Args:
        def __init__(self):
            self.model_path = str(model_path)
            self.source_path = str(source_path)
            self.sh_degree = sh_degree
            self.white_background = white_background
            self.images = "images"
            self.resolution = -1
            self.data_device = "cuda"
            self.eval = False  # 우리가 직접 test indices를 지정하므로
            self.depths = ""
            self.train_test_exp = False

    args = Args()

    with torch.no_grad():
        # Gaussian 모델 로드
        gaussians = GaussianModel(sh_degree)

        # Scene 로드 (모든 카메라를 train으로 로드)
        scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

        # 배경 설정
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 모든 카메라 가져오기 (train_cameras에 모든 카메라가 있음)
        all_cameras = scene.getTrainCameras()

        print(f"전체 카메라 수: {len(all_cameras)}")

        # Test indices에 해당하는 카메라만 필터링
        test_cameras = []
        for idx, cam in enumerate(all_cameras):
            if idx in test_indices:
                test_cameras.append(cam)

        print(f"필터링된 test 카메라 수: {len(test_cameras)}")

        # 렌더링 시작
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class SimplePipeline:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
                self.antialiasing = False

        pipeline = SimplePipeline()

        print(f"\n렌더링 시작...")
        for idx, view in enumerate(tqdm(test_cameras, desc="Rendering progress")):
            # 원본 인덱스 찾기
            original_idx = None
            for i, cam in enumerate(all_cameras):
                if cam == view:
                    original_idx = i
                    break

            # 렌더링
            rendering = render(view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]

            # 저장 (원본 인덱스를 파일명에 사용)
            if original_idx is not None:
                render_filename = output_dir / f"render_{original_idx:04d}.png"
                gt_filename = output_dir / f"gt_{original_idx:04d}.png"
            else:
                render_filename = output_dir / f"render_{idx:04d}.png"
                gt_filename = output_dir / f"gt_{idx:04d}.png"

            torchvision.utils.save_image(rendering, str(render_filename))
            torchvision.utils.save_image(gt, str(gt_filename))

        print(f"✓ 렌더링 완료: {output_dir}")
        print(f"  - {len(test_cameras)}개 뷰 렌더링됨")

        return output_dir
