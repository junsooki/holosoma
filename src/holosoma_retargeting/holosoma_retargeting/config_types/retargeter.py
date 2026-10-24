"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int] | tuple[int, int, float]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Each window is (start, end) or (start, end, z_floor).
    If z_floor is given per-window, it overrides the global z_floor for that window.
    Example: {"L_Toe": [(30, 60, 0.15)], "R_Toe": [(10, 20), (80, 95, 0.30)]}"""

    z_floor: float = 0.0
    """Default floor height used by Z pinning constraints (overridden by per-window z)."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""
    auto_windows: bool = False
    """Derive the windows from the source motion instead of ``windows`` (robot_retarget.py): frames where holosoma's
    velocity contact rule marks the toe as sticking AND the source toe is below ``auto_max_toe_height``, in runs of at
    least ``auto_min_frames`` frames."""
    auto_max_toe_height: float = 0.02
    """Source toe height (m, after preprocessing) below which a velocity-contact frame may be locked (auto_windows)."""
    auto_min_frames: int = 5
    """Minimum run length (frames) of an automatic foot-lock window (auto_windows)."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""

    skip_pairs: list[tuple[str, str]] = field(default_factory=list)
    """Subset of ``pairs`` that is not enforced on the frames in ``skip_windows`` (default: none).
    Example: [("left_ankle_roll_link", "right_ankle_roll_link")]"""

    skip_windows: list[tuple[int, int]] = field(default_factory=list)
    """Inclusive frame windows during which ``skip_pairs`` are not enforced; every other pair follows ``windows``.
    Example: [(76, 80)] skips ``skip_pairs`` on frames 76..80 only."""


@dataclass(frozen=True)
class FrameZeroStandConfig:
    """Double-support stand on source frame 0, released smoothly back to the unconstrained targets.

    Frame 0: every foot sticking link (G1: the sole spheres) at ``z_floor`` within ``sole_tolerance`` (both soles on the
    ground and flat) and the z axis of every ``upright_bodies`` body vertical within ``angle_tolerance`` (roll and pitch
    zero; yaw stays free). Frames 1..blend_frames-1 keep the same hard constraints but widen their bounds as the stand
    weight s falls from 1 to 0 along ``blend_profile``: z-axis components within angle_tolerance + (1 - s), link heights
    up to z_floor + sole_tolerance + (1 - s) * sole_release_height. From frame ``blend_frames`` on nothing is added."""

    enable: bool = False
    """Whether to enforce the frame-0 stand."""
    blend_frames: int = 20
    """Frames over which the stand weight falls from 1 (frame 0) to 0 (frame ``blend_frames`` and later)."""
    blend_profile: Literal["cosine", "min_jerk"] = "cosine"
    """Weight profile of the blend."""
    upright_bodies: list[str] = field(
        default_factory=lambda: ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    )
    """Bodies whose z axis is held vertical (roll/pitch zero)."""
    angle_tolerance: float = 0.005
    """Frame-0 bound on each horizontal component of a body z axis (~ roll/pitch in rad)."""
    z_floor: float = 0.005
    """Height of the foot sticking links on frame 0 (G1 sole sphere centres: their 5 mm radius above the ground)."""
    sole_tolerance: float = 1e-3
    """Frame-0 tolerance of the foot sticking link heights."""
    sole_release_height: float = 0.3
    """Growth of the upper foot-link height bound when the stand weight reaches 0 (m)."""
    init_from_stand: bool = True
    """Start frame 0 from the robot's default stand (zero joints) with the initial heading (yaw only), pelvis xy and
    the height that puts the foot sticking links at ``z_floor``."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    smooth_weight_joints: dict[str, float] = field(default_factory=dict)
    """Opt-in per-joint smoothness weights (joint name -> weight on the change from the previous frame); every other
    variable keeps the default 0.2."""

    contact_anchors: str = ""
    """Opt-in contact anchors (VICAR Eq. 1/7), an npz path; empty = off. Keys: `link` (K,) body names, `window` (K, 2)
    first and last frame (inclusive), `position` (K, T, 3) and `rotation` (K, T, 3, 3) world targets per frame,
    `pos_weight` and `rot_weight` (K,) or per frame (K, T). On its window frames, anchor k adds the linearized soft cost
    pos_weight ||p + J_p dq - p*||^2 + rot_weight ||J_r dq - log(R* R^T)||^2 on its body."""

    rotation_increments: str = ""
    """Opt-in rotation-increment term (VICAR L_R), an npz path; empty = off. Keys: `link` (L,) body names, `rotation`
    (L, T, 3, 3) source (human) rotations per frame, `weight` (L,). From frame 1 on, each body's world rotation change
    from the previous solved frame is matched to the source's: weight ||J_r dq - log(R_s(t) R_s(t-1)^T R(t-1) R^T)||^2
    (no rest-pose offset between the source and the robot body is needed)."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""
    frame0_stand: FrameZeroStandConfig = field(default_factory=FrameZeroStandConfig)
    """Configuration for the frame-0 double-support stand."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
