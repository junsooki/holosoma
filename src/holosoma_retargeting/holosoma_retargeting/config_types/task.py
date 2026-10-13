"""Configuration types for retargeting task settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific configuration parameters.

    These parameters control task-specific behavior like ground meshgrid generation,
    object sampling, augmentation, and scaling. Can be overridden via CLI.
    """

    # Object name
    # Auto-determined based on task_type if None: "largebox" for object_interaction,
    # "multi_boxes" for climbing, "ground" for robot_only
    object_name: str | None = None

    # Ground meshgrid (robot_only task)
    ground_size: int = 15
    ground_range: tuple[float, float] = (-1.0, 1.0)

    # Climbing ground meshgrid (climbing task)
    climbing_ground_size: int = 8
    climbing_ground_range: tuple[float, float] = (-2.0, 2.0)

    # Surface weight parameters for climbing object sampling (climbing task)
    # Used in weighted_surface_sampling: points with z-coordinate > threshold get high weight
    # This biases sampling toward top surfaces (important for climbing contact points)
    surface_weight_threshold: float = 0.9  # z-coordinate threshold for high-weight points
    surface_weight_high: int = 20  # Weight for top surface points (z > threshold)
    surface_weight_low: int = 1  # Weight for other points

    # Object directory (for climbing tasks)
    # Auto-determined from data_path / task_name if None
    object_dir: Path | None = None

    # Opt-in object loader for object_interaction from our own data (default None: InterMimic .pt as upstream).
    # object_poses: .npz with `object_poses` [T, 7] = [qw, qx, qy, qz, x, y, z] of the object in the motion frame (z up,
    # metres, same frame and frames as global_joint_positions). object_mesh: the object mesh (OBJ/PLY/STL, metres, in the
    # object frame of those poses). object_urdf: optional URDF (only used by the viser visualizer). With object_poses set,
    # human joints load from `<data_path>/<task_name>.npz` (global_joint_positions, height) for any registered data_format
    # and the MuJoCo scene is generated from the robot MJCF plus a free-jointed `<object_name>_link` body.
    object_mesh: Path | None = None
    object_urdf: Path | None = None
    object_poses: Path | None = None
    # Opt-in static scene (default None; object_poses mode), OmniRetarget Sec. III-A environment. scene_obstacles_xml: MJCF
    # whose <asset> and <worldbody> children are merged into the generated scene; geoms named `obstacle_*` get the same
    # non-penetration constraints as the object (robot geoms only). scene_points: .npz `points` [N, 3] sampled on those
    # surfaces in the motion frame, added to the interaction mesh on every frame.
    scene_obstacles_xml: Path | None = None
    scene_points: Path | None = None
