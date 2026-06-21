# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpiralTraceSample:
    time: float
    command_xy: np.ndarray
    actual_xy: np.ndarray
    spiral_offset_xy: np.ndarray
    radius: float


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_spiral_trace_svg(
    path: Path,
    samples: list[SpiralTraceSample],
    initial_offset_xy: np.ndarray,
    search_radius: float,
    alignment_tolerance: float,
) -> None:
    """Write a top-view SVG of the spiral rubbing/search path on the surface."""
    path.parent.mkdir(exist_ok=True)
    initial_offset_xy = np.asarray(initial_offset_xy, dtype=float)
    command = [np.asarray(s.command_xy, dtype=float) for s in samples]
    actual = [np.asarray(s.actual_xy, dtype=float) for s in samples]
    points = [np.zeros(2), initial_offset_xy]
    points.extend(command)
    points.extend(actual)

    search_radius = max(0.0, float(search_radius))
    alignment_tolerance = max(0.0, float(alignment_tolerance))
    circle_extents = [
        initial_offset_xy + np.array([search_radius, search_radius]),
        initial_offset_xy - np.array([search_radius, search_radius]),
        np.array([alignment_tolerance, alignment_tolerance]),
        -np.array([alignment_tolerance, alignment_tolerance]),
    ]
    points.extend(circle_extents)
    arr = np.vstack(points)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    pad = max(0.005, 0.12 * float(np.max(span)))
    min_xy -= pad
    max_xy += pad
    span = max_xy - min_xy

    width = 720
    height = 720
    margin = 70
    scale = min((width - 2 * margin) / float(span[0]), (height - 2 * margin) / float(span[1]))

    def to_px(point: np.ndarray) -> tuple[float, float]:
        x = margin + (float(point[0]) - float(min_xy[0])) * scale
        y = height - margin - (float(point[1]) - float(min_xy[1])) * scale
        return x, y

    command_px = [to_px(p) for p in command]
    actual_px = [to_px(p) for p in actual]
    hole_px = to_px(np.zeros(2))
    estimated_px = to_px(initial_offset_xy)
    search_radius_px = search_radius * scale
    tolerance_px = alignment_tolerance * scale

    if command_px:
        start_px = command_px[0]
        end_px = command_px[-1]
    else:
        start_px = estimated_px
        end_px = estimated_px

    title = "Archimedes spiral rubbing trajectory on the hole surface"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{escape(title)}</title>
  <rect width="100%" height="100%" fill="#f7f7f4"/>
  <text x="34" y="38" font-family="Arial, sans-serif" font-size="22" fill="#1f2933">Archimedes Spiral Rubbing Trace</text>
  <text x="34" y="62" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Top view on the contact surface, coordinates relative to the true hole center.</text>

  <line x1="{margin}" y1="{hole_px[1]:.2f}" x2="{width - margin}" y2="{hole_px[1]:.2f}" stroke="#d2d6dc" stroke-width="1"/>
  <line x1="{hole_px[0]:.2f}" y1="{margin}" x2="{hole_px[0]:.2f}" y2="{height - margin}" stroke="#d2d6dc" stroke-width="1"/>

  <circle cx="{estimated_px[0]:.2f}" cy="{estimated_px[1]:.2f}" r="{search_radius_px:.2f}" fill="none" stroke="#9aa6b2" stroke-width="2" stroke-dasharray="8 7"/>
  <circle cx="{hole_px[0]:.2f}" cy="{hole_px[1]:.2f}" r="{tolerance_px:.2f}" fill="#27ae60" fill-opacity="0.16" stroke="#229954" stroke-width="2"/>
  <circle cx="{hole_px[0]:.2f}" cy="{hole_px[1]:.2f}" r="5" fill="#229954"/>
  <circle cx="{estimated_px[0]:.2f}" cy="{estimated_px[1]:.2f}" r="5" fill="#d97706"/>
"""
    if command_px:
        svg += f'  <polyline points="{_polyline(command_px)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>\n'
    if actual_px:
        svg += f'  <polyline points="{_polyline(actual_px)}" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.72"/>\n'

    svg += f"""  <circle cx="{start_px[0]:.2f}" cy="{start_px[1]:.2f}" r="4" fill="#111827"/>
  <circle cx="{end_px[0]:.2f}" cy="{end_px[1]:.2f}" r="5" fill="#7c3aed"/>

  <g font-family="Arial, sans-serif" font-size="13" fill="#1f2933">
    <rect x="34" y="{height - 132}" width="330" height="96" rx="6" fill="#ffffff" stroke="#d2d6dc"/>
    <circle cx="54" cy="{height - 107}" r="5" fill="#229954"/><text x="70" y="{height - 102}">True hole center and alignment tolerance</text>
    <circle cx="54" cy="{height - 82}" r="5" fill="#d97706"/><text x="70" y="{height - 77}">Coarse vision estimate</text>
    <line x1="49" y1="{height - 56}" x2="64" y2="{height - 56}" stroke="#2563eb" stroke-width="3"/><text x="70" y="{height - 51}">Commanded Archimedes spiral</text>
  </g>
  <text x="{width - 260}" y="{height - 38}" font-family="Consolas, monospace" font-size="12" fill="#4b5563">offset=({initial_offset_xy[0]:+.4f}, {initial_offset_xy[1]:+.4f}) m</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
