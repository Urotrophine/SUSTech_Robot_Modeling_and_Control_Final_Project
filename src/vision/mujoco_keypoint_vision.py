# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import base64
import html
import io
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fovy_deg: float


@dataclass(frozen=True)
class KeypointVisionEstimate:
    mode: str
    source: str
    camera_name: str
    hole_world_est: np.ndarray
    peg_world_est: np.ndarray
    true_hole_world: np.ndarray
    true_peg_world: np.ndarray
    initial_offset_xy: np.ndarray
    hole_pixel: np.ndarray
    peg_pixel: np.ndarray
    true_hole_pixel: np.ndarray
    true_peg_pixel: np.ndarray
    hole_pixel_error: float
    peg_pixel_error: float
    confidence: float


def _id(model, objtype, name: str) -> int:
    oid = mujoco.mj_name2id(model, objtype, name)
    if oid < 0:
        raise ValueError(f"Object not found: {name}")
    return oid


def site_world_pos(model, data, site_name: str) -> np.ndarray:
    sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_forward(model, data)
    return data.site_xpos[sid].copy()


def camera_intrinsics(model, camera_name: str, width: int, height: int) -> CameraIntrinsics:
    camera_id = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy_deg = float(model.cam_fovy[camera_id])
    if fovy_deg <= 0.0:
        fovy_deg = 45.0
    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    fx = fy
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * (float(width) - 1.0),
        cy=0.5 * (float(height) - 1.0),
        fovy_deg=fovy_deg,
    )


def _camera_pose(model, data, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    camera_id = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    mujoco.mj_forward(model, data)
    pos = data.cam_xpos[camera_id].copy()
    rot = data.cam_xmat[camera_id].reshape(3, 3).copy()
    return pos, rot


def project_world_to_pixel(
    model,
    data,
    camera_name: str,
    point_world: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    """Project a world point into a MuJoCo camera.

    MuJoCo cameras look along local -Z, so the returned depth is positive
    forward distance, not Euclidean range.
    """
    intr = camera_intrinsics(model, camera_name, width, height)
    camera_pos, camera_rot = _camera_pose(model, data, camera_name)
    point_cam = camera_rot.T @ (np.asarray(point_world, dtype=float) - camera_pos)
    depth = -float(point_cam[2])
    if depth <= 1e-12:
        raise ValueError(f"Point is behind camera {camera_name}: {point_world}")
    pixel = np.array(
        [
            intr.cx + intr.fx * float(point_cam[0]) / depth,
            intr.cy - intr.fy * float(point_cam[1]) / depth,
        ],
        dtype=float,
    )
    return pixel, depth


def pixel_depth_to_world(
    model,
    data,
    camera_name: str,
    pixel: np.ndarray,
    depth: float,
    width: int,
    height: int,
) -> np.ndarray:
    intr = camera_intrinsics(model, camera_name, width, height)
    depth = float(depth)
    if depth <= 1e-12:
        raise ValueError(f"Depth must be positive for camera unprojection: {depth}")
    pixel = np.asarray(pixel, dtype=float)
    point_cam = np.array(
        [
            (float(pixel[0]) - intr.cx) * depth / intr.fx,
            -(float(pixel[1]) - intr.cy) * depth / intr.fy,
            -depth,
        ],
        dtype=float,
    )
    camera_pos, camera_rot = _camera_pose(model, data, camera_name)
    return camera_pos + camera_rot @ point_cam


def render_camera_rgbd(model, data, camera_name: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = renderer.render()
    finally:
        renderer.close()
    return rgb, depth


def _svg_point(pixel: np.ndarray, width: int, height: int) -> tuple[float, float, bool]:
    p = np.asarray(pixel, dtype=float)
    if p.shape[0] < 2 or not np.all(np.isfinite(p[:2])):
        return 0.0, 0.0, False
    inside = 0.0 <= p[0] <= width - 1 and 0.0 <= p[1] <= height - 1
    return float(np.clip(p[0], 0.0, width - 1.0)), float(np.clip(p[1], 0.0, height - 1.0)), bool(inside)


def write_keypoint_overlay_svg(
    path: Path,
    image_rgb: np.ndarray,
    estimate: KeypointVisionEstimate,
    title: str = "MuJoCo Vision Keypoints",
) -> None:
    """Write a visual proof image for the coarse visual localization step."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Vision debug SVG requires Pillow.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    items = [
        ("hole est", estimate.hole_pixel, "#ff4d2d", False),
        ("hole true", estimate.true_hole_pixel, "#34c759", True),
        ("peg est", estimate.peg_pixel, "#1f77ff", False),
        ("peg true", estimate.true_peg_pixel, "#00d5ff", True),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 86}" viewBox="0 0 {width} {height + 86}">',
        f'<rect width="{width}" height="{height + 86}" fill="#111"/>',
        f'<image x="0" y="0" width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>',
        f'<text x="12" y="{height + 24}" fill="#f5f5f5" font-family="Arial, sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<text x="12" y="{height + 48}" fill="#d8d8d8" font-family="Arial, sans-serif" font-size="13">'
        f'mode={html.escape(estimate.mode)}  camera={html.escape(estimate.camera_name)}  '
        f'hole_error={estimate.hole_pixel_error:.3f}px  peg_error={estimate.peg_pixel_error:.3f}px</text>',
    ]

    legend_x = 12
    for label, pixel, color, dashed in items:
        x, y, visible = _svg_point(pixel, width, height)
        dash = ' stroke-dasharray="5 4"' if dashed or not visible else ""
        if visible:
            lines.append(f'<line x1="{x - 10:.2f}" y1="{y:.2f}" x2="{x + 10:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"{dash}/>')
            lines.append(f'<line x1="{x:.2f}" y1="{y - 10:.2f}" x2="{x:.2f}" y2="{y + 10:.2f}" stroke="{color}" stroke-width="2"{dash}/>')
        else:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        lines.append(
            f'<text x="{min(width - 130, x + 8):.2f}" y="{max(14, y - 8):.2f}" '
            f'fill="{color}" font-family="Arial, sans-serif" font-size="13">{html.escape(label)}</text>'
        )

        lines.append(f'<circle cx="{legend_x}" cy="{height + 70}" r="5" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 10}" y="{height + 74}" fill="#e8e8e8" '
            f'font-family="Arial, sans-serif" font-size="12">{html.escape(label)}</text>'
        )
        legend_x += 94

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def roi_transform(
    hole_center: np.ndarray,
    hole_size: float,
    hole_normal: np.ndarray,
    crop_hole_scale: float = 5.0,
    crop_size: int = 224,
) -> np.ndarray:
    """Same full-image-to-crop transform as the retrained visual module."""
    hole_center = np.asarray(hole_center, dtype=float)
    hole_normal = np.asarray(hole_normal, dtype=float)
    if np.linalg.norm(hole_normal) < 1e-9:
        hole_normal = np.array([0.0, 1.0], dtype=float)
    angle = -math.atan2(float(hole_normal[1]), float(hole_normal[0])) + math.pi / 2.0
    s, c = math.sin(angle), math.cos(angle)
    transform = np.eye(3, dtype=float)
    transform[:2, :2] = np.array(((c, -s), (s, c)), dtype=float)
    size = max(1e-9, float(hole_size) * float(crop_hole_scale))
    transform[:2, 2] = (transform[:2, :2] @ -hole_center) + (size / 2.0, size / 3.0)
    transform[:2] *= float(crop_size) / size
    return transform


def warp_affine_rgb(image: np.ndarray, transform: np.ndarray, size: int) -> np.ndarray:
    """Warp RGB image using an input-pixel to output-pixel affine transform."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mujoco-unet ROI inference requires Pillow. Install it together with torch/torchvision."
        ) from exc

    image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    inv = np.linalg.inv(transform)
    coeffs = tuple(float(v) for v in inv[:2, :].reshape(-1))
    warped = image_pil.transform(
        (int(size), int(size)),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(warped, dtype=np.uint8)


def heatmap_argmax(heatmaps: np.ndarray) -> np.ndarray:
    """Return keypoint pixels as [x, y] from heatmaps shaped [N, H, W]."""
    arr = np.asarray(heatmaps)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected heatmaps shaped [N,H,W], got {arr.shape}")

    points = []
    for heatmap in arr:
        y, x = np.unravel_index(int(np.nanargmax(heatmap)), heatmap.shape)
        points.append([float(x), float(y)])
    return np.asarray(points, dtype=float)


def _point_to_homogeneous_xy(point_xy: np.ndarray) -> np.ndarray:
    return np.array([float(point_xy[0]), float(point_xy[1]), 1.0], dtype=float)


def _default_hole_tangent(axis_world: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_world, dtype=float)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    tangent = candidate - axis * float(np.dot(candidate, axis))
    if np.linalg.norm(tangent) < 1e-9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent = candidate - axis * float(np.dot(candidate, axis))
    return tangent / max(1e-12, float(np.linalg.norm(tangent)))


def make_retrain_roi_transform(
    *,
    model,
    data,
    camera_name: str,
    hole_world: np.ndarray,
    insertion_axis_world: np.ndarray,
    hole_size_world: float,
    width: int,
    height: int,
    crop_hole_scale: float,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Build the ROI transform expected by the MuJoCo retrained U-Net."""
    hole_pixel, _ = project_world_to_pixel(model, data, camera_name, hole_world, width, height)
    axis = np.asarray(insertion_axis_world, dtype=float)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    tangent = _default_hole_tangent(axis)
    half_size = 0.5 * max(1e-9, float(hole_size_world))
    edge_a, _ = project_world_to_pixel(model, data, camera_name, hole_world - tangent * half_size, width, height)
    edge_b, _ = project_world_to_pixel(model, data, camera_name, hole_world + tangent * half_size, width, height)
    hole_size_px = max(1.0, float(np.linalg.norm(edge_b - edge_a)))
    try:
        normal_px, _ = project_world_to_pixel(model, data, camera_name, hole_world + axis * 0.05, width, height)
        hole_normal = normal_px - hole_pixel
    except ValueError:
        hole_normal = np.array([0.0, 1.0], dtype=float)
    if np.linalg.norm(hole_normal) < 1e-6:
        hole_normal = np.array([0.0, 1.0], dtype=float)

    transform = roi_transform(
        hole_pixel,
        hole_size_px,
        hole_normal,
        crop_hole_scale=crop_hole_scale,
        crop_size=crop_size,
    )
    return transform, hole_pixel, hole_size_px, hole_normal


def _nan_pixel() -> np.ndarray:
    return np.array([np.nan, np.nan], dtype=float)


def estimate_random_vision(
    *,
    true_hole_world: np.ndarray,
    true_peg_world: np.ndarray,
    initial_offset_xy: np.ndarray,
    mode: str,
    seed: Optional[int],
) -> KeypointVisionEstimate:
    offset = np.asarray(initial_offset_xy, dtype=float)
    hole_est = np.asarray(true_hole_world, dtype=float).copy()
    hole_est[:2] += offset
    return KeypointVisionEstimate(
        mode=mode,
        source=f"configured {mode} XY coarse-vision offset"
        + (f" with seed {seed}" if seed is not None else ""),
        camera_name="none",
        hole_world_est=hole_est,
        peg_world_est=np.asarray(true_peg_world, dtype=float).copy(),
        true_hole_world=np.asarray(true_hole_world, dtype=float).copy(),
        true_peg_world=np.asarray(true_peg_world, dtype=float).copy(),
        initial_offset_xy=offset.copy(),
        hole_pixel=_nan_pixel(),
        peg_pixel=_nan_pixel(),
        true_hole_pixel=_nan_pixel(),
        true_peg_pixel=_nan_pixel(),
        hole_pixel_error=float("nan"),
        peg_pixel_error=float("nan"),
        confidence=1.0,
    )


def estimate_mujoco_oracle(
    *,
    model,
    data,
    camera_name: str,
    true_hole_world: np.ndarray,
    true_peg_world: np.ndarray,
    width: int,
    height: int,
    pixel_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> KeypointVisionEstimate:
    # Render once so this mode validates the MuJoCo camera/render path. The
    # oracle estimate still uses the known keypoint depths for deterministic
    # projection/unprojection.
    render_camera_rgbd(model, data, camera_name, width, height)
    generator = rng if rng is not None else np.random.default_rng()

    true_hole_pixel, hole_depth = project_world_to_pixel(
        model, data, camera_name, true_hole_world, width, height
    )
    true_peg_pixel, peg_depth = project_world_to_pixel(
        model, data, camera_name, true_peg_world, width, height
    )
    noise_std = max(0.0, float(pixel_noise_std))
    hole_noise = generator.normal(0.0, noise_std, size=2) if noise_std > 0.0 else np.zeros(2)
    peg_noise = generator.normal(0.0, noise_std, size=2) if noise_std > 0.0 else np.zeros(2)
    hole_pixel = true_hole_pixel + hole_noise
    peg_pixel = true_peg_pixel + peg_noise

    hole_est = pixel_depth_to_world(model, data, camera_name, hole_pixel, hole_depth, width, height)
    peg_est = pixel_depth_to_world(model, data, camera_name, peg_pixel, peg_depth, width, height)
    return KeypointVisionEstimate(
        mode="mujoco-oracle",
        source=f"MuJoCo camera oracle projection plus {noise_std:.3g}px Gaussian pixel noise",
        camera_name=camera_name,
        hole_world_est=hole_est,
        peg_world_est=peg_est,
        true_hole_world=np.asarray(true_hole_world, dtype=float).copy(),
        true_peg_world=np.asarray(true_peg_world, dtype=float).copy(),
        initial_offset_xy=hole_est[:2] - np.asarray(true_hole_world, dtype=float)[:2],
        hole_pixel=hole_pixel,
        peg_pixel=peg_pixel,
        true_hole_pixel=true_hole_pixel,
        true_peg_pixel=true_peg_pixel,
        hole_pixel_error=float(np.linalg.norm(hole_pixel - true_hole_pixel)),
        peg_pixel_error=float(np.linalg.norm(peg_pixel - true_peg_pixel)),
        confidence=1.0,
    )


def sample_depth_at_pixel(depth_image: np.ndarray, pixel: np.ndarray, radius: int = 2) -> float:
    depth = np.asarray(depth_image, dtype=float)
    x = int(np.clip(round(float(pixel[0])), 0, depth.shape[1] - 1))
    y = int(np.clip(round(float(pixel[1])), 0, depth.shape[0] - 1))
    y0 = max(0, y - radius)
    y1 = min(depth.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(depth.shape[1], x + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError(f"No valid depth near pixel {pixel}")
    return float(np.median(valid))


def estimate_mujoco_unet(
    *,
    model,
    data,
    camera_name: str,
    true_hole_world: np.ndarray,
    true_peg_world: np.ndarray,
    width: int,
    height: int,
    model_path: Path,
    device: str = "cpu",
    crop_size: int = 224,
    crop_hole_scale: float = 5.0,
    hole_size_world: float = 0.06,
    insertion_axis_world: Optional[np.ndarray] = None,
) -> KeypointVisionEstimate:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "mujoco-unet vision requires --vision-model pointing to a trained .pth file; "
            f"not found: {model_path}"
        )

    from vision.unet_keypoints import infer_unet_heatmaps

    rgb, depth = render_camera_rgbd(model, data, camera_name, width, height)
    axis_world = (
        np.asarray(insertion_axis_world, dtype=float)
        if insertion_axis_world is not None
        else np.array([0.0, 0.0, -1.0], dtype=float)
    )
    crop_transform, _, _, _ = make_retrain_roi_transform(
        model=model,
        data=data,
        camera_name=camera_name,
        hole_world=true_hole_world,
        insertion_axis_world=axis_world,
        hole_size_world=hole_size_world,
        width=width,
        height=height,
        crop_hole_scale=crop_hole_scale,
        crop_size=crop_size,
    )
    crop_rgb = warp_affine_rgb(rgb, crop_transform, crop_size)
    heatmaps = infer_unet_heatmaps(crop_rgb, model_path=model_path, device=device)
    crop_points = heatmap_argmax(heatmaps)
    inv_crop_transform = np.linalg.inv(crop_transform)
    points = np.asarray(
        [(inv_crop_transform @ _point_to_homogeneous_xy(p))[:2] for p in crop_points],
        dtype=float,
    )
    if points.shape[0] < 2:
        raise ValueError(f"Expected at least two U-Net heatmaps for hole and peg, got {points.shape[0]}")

    hole_pixel = points[0]
    peg_pixel = points[1]
    hole_depth = sample_depth_at_pixel(depth, hole_pixel)
    peg_depth = sample_depth_at_pixel(depth, peg_pixel)
    hole_est = pixel_depth_to_world(model, data, camera_name, hole_pixel, hole_depth, width, height)
    peg_est = pixel_depth_to_world(model, data, camera_name, peg_pixel, peg_depth, width, height)
    true_hole_pixel, _ = project_world_to_pixel(model, data, camera_name, true_hole_world, width, height)
    true_peg_pixel, _ = project_world_to_pixel(model, data, camera_name, true_peg_world, width, height)
    confidence = float(np.mean([np.nanmax(heatmaps[0]), np.nanmax(heatmaps[1])]))

    return KeypointVisionEstimate(
        mode="mujoco-unet",
        source=f"MuJoCo camera ROI crop plus U-Net heatmap inference from {model_path}",
        camera_name=camera_name,
        hole_world_est=hole_est,
        peg_world_est=peg_est,
        true_hole_world=np.asarray(true_hole_world, dtype=float).copy(),
        true_peg_world=np.asarray(true_peg_world, dtype=float).copy(),
        initial_offset_xy=hole_est[:2] - np.asarray(true_hole_world, dtype=float)[:2],
        hole_pixel=hole_pixel,
        peg_pixel=peg_pixel,
        true_hole_pixel=true_hole_pixel,
        true_peg_pixel=true_peg_pixel,
        hole_pixel_error=float(np.linalg.norm(hole_pixel - true_hole_pixel)),
        peg_pixel_error=float(np.linalg.norm(peg_pixel - true_peg_pixel)),
        confidence=confidence,
    )
