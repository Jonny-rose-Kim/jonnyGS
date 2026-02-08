"""
Noise data loader for noise-aware 3DGS retraining.

Loads noise detection results (indices, 2D masks, confidence scores)
and provides them to the training pipeline.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image


class NoiseDataLoader:
    """
    Loads noise detection results and provides them to the training pipeline.

    Expected directory structure (noise_data_dir):
        noise_gaussians.json   — noise indices, cluster results, metadata
        noise_gaussians.ply    — noise Gaussian points
        clean_gaussians.ply    — clean Gaussian points
        cameras.json           — camera info with id-to-name mapping

    Optional sibling directories:
        ../dataset_pairs/      — pre-computed 2D noise masks (Step 4)
        ../rendered_views/     — pre-rendered views (Step 3)
    """

    def __init__(self, noise_data_dir: str, detector_checkpoint: str = None, device: str = "cuda"):
        """
        Args:
            noise_data_dir: Path to noise_gaussians/ directory
            detector_checkpoint: Path to detector model checkpoint for on-the-fly mask generation
            device: torch device
        """
        self.noise_data_dir = Path(noise_data_dir)
        self.device = device
        self._detector = None
        self._detector_checkpoint = detector_checkpoint

        # Load noise_gaussians.json
        json_path = self.noise_data_dir / "noise_gaussians.json"
        if not json_path.exists():
            raise FileNotFoundError(f"noise_gaussians.json not found at {json_path}")

        with open(json_path) as f:
            self._noise_data = json.load(f)

        self._noise_indices = set(self._noise_data["noise_gaussian_indices"])
        self._total_gaussians = self._noise_data["total_gaussians"]
        self._clean_count = self._noise_data["clean_gaussians_count"]

        # Build camera name-to-id mapping from cameras.json
        self._cam_name_to_id = {}
        cameras_path = self.noise_data_dir / "cameras.json"
        if cameras_path.exists():
            with open(cameras_path) as f:
                cameras = json.load(f)
            for cam in cameras:
                self._cam_name_to_id[cam["img_name"]] = cam["id"]

        # Locate dataset_pairs directory (sibling)
        self._dataset_pairs_dir = self.noise_data_dir.parent / "dataset_pairs"
        self._mask_cache = {}

        # Build view_index to pair path mapping for pre-computed masks
        self._view_to_mask_path = {}
        if self._dataset_pairs_dir.exists():
            for split in ["train", "val", "test"]:
                split_dir = self._dataset_pairs_dir / split
                if not split_dir.exists():
                    continue
                for pair_dir in split_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    meta_path = pair_dir / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        view_idx = meta["view_index"]
                        mask_path = pair_dir / "mask.png"
                        if mask_path.exists():
                            self._view_to_mask_path[view_idx] = mask_path

        # Load noise Gaussian positions for spatial mask
        self._noise_positions = None

        # Derive per-cluster confidence from cluster_results
        self._cluster_vote_counts = self._compute_cluster_votes()

        # Log summary
        n_masks = len(self._view_to_mask_path)
        print(f"[NOISE] Loaded {len(self._noise_indices)} noise indices "
              f"(total: {self._total_gaussians}, clean: {self._clean_count})")
        print(f"[NOISE] Pre-computed 2D masks available for {n_masks} views")
        if detector_checkpoint:
            print(f"[NOISE] Detector checkpoint: {detector_checkpoint}")

    def _compute_cluster_votes(self) -> dict:
        """Count how many clusters voted each Gaussian as noise."""
        from collections import Counter
        vote_counts = Counter()
        cluster_results = self._noise_data.get("cluster_results", [])
        for cr in cluster_results:
            for idx in cr.get("noise_gaussian_indices", []):
                vote_counts[idx] += 1
        return vote_counts

    def load_noise_indices(self) -> set:
        """Return set of noise Gaussian indices."""
        return self._noise_indices

    def load_noise_mask_for_view(self, view_name: str, render_hw: tuple = None) -> torch.Tensor:
        """
        Return 2D noise mask for a specific view.

        Args:
            view_name: Camera image name (e.g., "_DSC9214.JPG")
            render_hw: (H, W) tuple for resizing mask to match render resolution

        Returns:
            mask: [H, W] float tensor, 0.0 (clean) ~ 1.0 (noise), or None if unavailable
        """
        # Check cache
        if view_name in self._mask_cache:
            mask = self._mask_cache[view_name]
            if render_hw and (mask.shape[0] != render_hw[0] or mask.shape[1] != render_hw[1]):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), size=render_hw, mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
            return mask

        # Try pre-computed mask from dataset_pairs
        view_id = self._cam_name_to_id.get(view_name)
        if view_id is not None and view_id in self._view_to_mask_path:
            mask_path = self._view_to_mask_path[view_id]
            mask = self._load_mask_from_file(mask_path)
            if render_hw and (mask.shape[0] != render_hw[0] or mask.shape[1] != render_hw[1]):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), size=render_hw, mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
            self._mask_cache[view_name] = mask
            return mask

        # No pre-computed mask available
        return None

    def generate_mask_from_rendered(self, rendered_image: torch.Tensor) -> torch.Tensor:
        """
        Generate noise mask on-the-fly using detector model.

        Args:
            rendered_image: [C, H, W] or [3, H, W] rendered image tensor, values in [0, 1]

        Returns:
            mask: [H, W] float tensor, 0~1
        """
        detector = self._get_detector()
        if detector is None:
            return None

        with torch.no_grad():
            mask = detector.predict_from_tensor(rendered_image)
            # [1, 1, H, W] → [H, W]
            mask = mask.squeeze(0).squeeze(0)
        return mask

    def load_confidence_scores(self, num_gaussians: int) -> torch.Tensor:
        """
        Return per-Gaussian noise confidence scores.

        Since the current data only has binary noise/clean classification,
        confidence is derived from multi-cluster voting:
            confidence = num_clusters_voted / total_clusters

        For Gaussians not in noise set: 0.0
        For noise Gaussians: vote_count / num_clusters (minimum 1/num_clusters)

        Args:
            num_gaussians: Current number of Gaussians (may differ from original
                          if Gaussians were removed or added)

        Returns:
            scores: [num_gaussians] float tensor, 0.0 ~ 1.0
        """
        num_clusters = len(self._noise_data.get("cluster_results", []))
        if num_clusters == 0:
            num_clusters = 1

        scores = torch.zeros(num_gaussians, dtype=torch.float32)

        for idx in self._noise_indices:
            if idx < num_gaussians:
                votes = self._cluster_vote_counts.get(idx, 1)
                scores[idx] = votes / num_clusters

        return scores

    def get_spatial_noise_mask(self, xyz: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Return 3D spatial mask indicating Gaussians near noise positions.

        Args:
            xyz: Current Gaussian positions [N, 3]
            radius: Suppression radius

        Returns:
            mask: [N] bool tensor, True = within radius of a noise Gaussian
        """
        noise_pos = self._get_noise_positions()
        if noise_pos is None or noise_pos.shape[0] == 0:
            return torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)

        noise_pos = noise_pos.to(xyz.device)

        # Chunk-based distance computation to avoid OOM for large point clouds
        N = xyz.shape[0]
        M = noise_pos.shape[0]
        mask = torch.zeros(N, dtype=torch.bool, device=xyz.device)

        chunk_size = 4096
        for i in range(0, M, chunk_size):
            chunk = noise_pos[i:i + chunk_size]  # [chunk, 3]
            # [N, 1, 3] - [1, chunk, 3] → [N, chunk]
            dists = torch.cdist(xyz.unsqueeze(0), chunk.unsqueeze(0)).squeeze(0)  # [N, chunk]
            mask |= (dists.min(dim=1).values < radius)

        return mask

    def _load_mask_from_file(self, mask_path) -> torch.Tensor:
        """Load a mask PNG file as float tensor [H, W] in range [0, 1]."""
        img = Image.open(mask_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def _get_noise_positions(self) -> torch.Tensor:
        """Load and cache noise Gaussian 3D positions from PLY."""
        if self._noise_positions is not None:
            return self._noise_positions

        ply_path = self.noise_data_dir / "noise_gaussians.ply"
        if not ply_path.exists():
            print(f"[NOISE] Warning: noise_gaussians.ply not found at {ply_path}")
            return None

        self._noise_positions = self._read_ply_positions(ply_path)
        print(f"[NOISE] Loaded {self._noise_positions.shape[0]} noise Gaussian positions")
        return self._noise_positions

    def _read_ply_positions(self, ply_path) -> torch.Tensor:
        """Read xyz positions from a PLY file."""
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        xyz = np.stack([
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"])
        ], axis=1)
        return torch.from_numpy(xyz.astype(np.float32))

    def _get_detector(self):
        """Lazy-load detector model."""
        if self._detector is not None:
            return self._detector

        if self._detector_checkpoint is None or not os.path.exists(self._detector_checkpoint):
            return None

        try:
            from src.utils.detector_inference import NoiseDetector
            self._detector = NoiseDetector(
                self._detector_checkpoint,
                device=self.device
            )
            return self._detector
        except Exception as e:
            print(f"[NOISE] Warning: Failed to load detector: {e}")
            return None

    def get_noise_count(self) -> int:
        """Return total number of noise Gaussians."""
        return len(self._noise_indices)

    def get_clean_count(self) -> int:
        """Return total number of clean Gaussians."""
        return self._clean_count
