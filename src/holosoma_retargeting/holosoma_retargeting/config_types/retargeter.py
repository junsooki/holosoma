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
    pos_weight ||p + J_p dq - p*||^2 + rot_weight ||J_r dq - log(R* R^T)||^2 on its body. Optional `parent` (K,) body
    names ("" = world): the rotation target is then relative to that body, R* = R_parent rotation[k, t], and the
    rotation term moves the body relative to its parent (J_r - J_r,parent)."""

    rotation_increments: str = ""
    """Opt-in rotation-increment term (VICAR L_R), an npz path; empty = off. Keys: `link` (L,) body names, `rotation`
    (L, T, 3, 3) source (human) rotations per frame, `weight` (L,). From frame 1 on, each body's world rotation change
    from the previous solved frame is matched to the source's: weight ||J_r dq - log(R_s(t) R_s(t-1)^T R(t-1) R^T)||^2
    (no rest-pose offset between the source and the robot body is needed)."""
    held_object: str = ""
    """Opt-in held object, an npz path; empty = off. Keys: `window` (2,) first and last frame (inclusive), `link` (K,)
    marker body names, `position` (K, T, 3) world targets per frame, `weight` (scalar), optional `joint` (J,) joint names.
    Over the window the geoms named `held_*` (an object welded to the robot) collide like robot geoms and the task object's
    geoms do not; before it the reverse, after it neither (the released object is placed outside the solve; the task
    object's track there is no longer where it is). On window frames each marker adds weight ||p + J S dq - target||^2, S keeping
    the columns of the `joint` joints (all when absent): the markers move only those, the rest follows the other terms."""

    prior: str = ""
    """Opt-in stage 2 (VICAR Sec. III-C2), an npz path; empty = off. Key `qpos` (T, nq): the stage-1 motion of this run
    (same frames, same frame of reference). With it the solve starts from the prior's first row (every later frame from the
    last solved one, as stock) and every frame minimises, in tolerance
    units (residual / tolerance, squared; tolerances in `hand_contacts`): the stay-near-prior term, the change from the
    previous solved frame, optionally the mapped keypoints' displacement from the previous solved frame against the
    source's (L_p),
    the rotation increments (`rotation_increments`, L_R) and the soft wrist, finger and contact targets of
    `hand_contacts`, in place of the Laplacian, nominal and smoothness costs. The non-penetration, self-collision and
    foot-sticking rows stay hard; a frame infeasible with them is solved again with an L1 slack on them (weight
    `slack_weight_m`, logged)."""

    hand_contacts: str = ""
    """Opt-in whole-hand contacts and holding check for stage 2 (with `prior`, both or neither), an npz path; empty = off.
    Keys: `link` (K,) bodies, `r_link` (K, 3) each link's contact point (link frame), `c`, `n` (K, 3) and `d` (K,) its
    mapped point on the object, the outward normal there and n.c (object frame), `sigma_contact` (K,) the tolerance of
    each link's distance to c, `window` (2,) the hold's first and last row, `E_wrist` (T, 4, 4) the wrist link's target
    pose in the object frame and `wrist_weight` (T,) its strength (0 to 1), `wrist_link`, `finger_joints` (J,),
    `finger_target` (T, J) and `finger_weight` (T,), `wrench` (T, 6) the object-frame force and torque (about `com`) the
    contacts must apply, optional `contact_weight` (T,) the contact terms' strength per row (default 1 on the window,
    0 elsewhere; the holding check stays on the window), `com` (3,), `mu`, `f_max`, `tau_max` (J,), `force_gap` (m), tolerances `sigma_prior` (one per
    optimisation variable: root position, root quaternion, joints in qpos order), `sigma_step_rad`, `sigma_keypoint_m` (optional: without it no L_p), `sigma_wrist_m`, `sigma_wrist_rad`, `sigma_finger_rad`, and
    the L1 slack weights `slack_weight_m`, `slack_weight_n`. On window rows every link's point is pulled toward its c
    (Morphometric Imitation's closest contact; non-penetration stays hard) and the holding check: contact forces f (one
    per link, zero for a link whose gap to its face exceeds force_gap) with G f = wrench (L1 slack), each f in its
    friction cone about -n with 0 <= f_n <= f_max, and |J_c^T f| <= tau_max over the finger joints (L1 slack). Writes
    <result>_stage2_log.npz: per row the links' distances to c and gaps, which links may carry force, the slack of every
    constraint group, the forces, the wrist's distance to its target, the solve time."""

    collision_slack: float = 0.0
    """Opt-in L1 slack on the non-penetration and self-collision rows (weight per metre of slack); 0 = off (hard, stock).
    With it a frame the hard rows make infeasible solves, and <result>_collision_log.npz records per frame the largest
    slack (how deep the linearized solve lets a pair penetrate), the sum and that pair (body:geom / body:geom). Stage 2
    (`prior`) keeps its own handling: hard rows, slack only on a frame infeasible with them."""

    vicar_stage2: str = ""
    """Opt-in VICAR stage 2 (Sec. III-C2, Eqs. 6-9) over the whole trajectory, an npz path; empty = off. In place of
    the per-frame solve: from `qpos` (T, nq), the stage-1 motion of this run (its object columns stay), minimise
    L_p + lambda_R L_R + lambda_s L_s + lambda_col L_col + lambda_a L_anchor subject to the MJCF joint limits and
    |q_t - q_t-1| <= `dq_max` (one per optimisation variable, inf = unbounded) by Gauss-Newton over all frames at once,
    `iterations` at most, trust region `step` per frame and iteration. Keys: `human_rotation` (B, T, 3, 3) the source
    rotation of each mapped joint (joints_mapping order) for L_R, `anchor_link` (K,), `anchor_window` (K, 2),
    `anchor_position` (K, T, 3), `anchor_rotation` (K, T, 3, 3) world targets, and the scalars `lambda_R`, `lambda_s`,
    `lambda_col`, `lambda_a`, `step`, `iterations`, `outer_iterations` (augmented-Lagrangian outer iterations at most),
    `lambda_L` (holosoma's interaction-mesh Laplacian term, 0 = off) with `laplacian_mask_links` (mapped joints whose edges to
    the object's points are dropped on the anchor windows), and `beta_R` (scalar, or (K,) one per anchor). Writes
    <result>_vicar_log.npz (with each weighted term and its gradient at the start trajectory, diag_*)."""

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
