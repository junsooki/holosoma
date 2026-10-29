from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.retargeter import FootLockConfig, FrameZeroStandConfig, SelfCollisionConfig

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


V2_WORKERS = 32  # vicar_stage2: processes linearizing the frames
_V2_RETARGETER = None


def _v2_frame(args):
    return _V2_RETARGETER._v2_linearize_frame(*args)


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        frame0_stand: FrameZeroStandConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        smooth_weight_joints: dict[str, float] | None = None,
        contact_anchors: str = "",
        rotation_increments: str = "",
        held_object: str = "",
        prior: str = "",
        hand_contacts: str = "",
        collision_slack: float = 0.0,
        vicar_stage2: str = "",
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            frame0_stand: configuration for the frame-0 double-support stand (off by default).
            nominal_tracking_tau: the time constant for the nominal tracking cost.
            smooth_weight_joints: opt-in per-joint smoothness weights (joint name -> weight), 0.2 for the rest.
            contact_anchors: opt-in npz of windowed body pose targets (RetargeterConfig.contact_anchors).
            rotation_increments: opt-in npz of source body rotations whose frame-to-frame change the robot body matches
                (RetargeterConfig.rotation_increments).
            held_object: opt-in npz of an object welded to the robot over a window (RetargeterConfig.held_object).
            prior: opt-in stage 2, the stage-1 motion it starts from and stays near (RetargeterConfig.prior).
            hand_contacts: opt-in stage-2 whole-hand contacts and holding check (RetargeterConfig.hand_contacts).
            collision_slack: opt-in L1 slack weight on the collision rows (RetargeterConfig.collision_slack).
            vicar_stage2: opt-in VICAR stage 2 over the whole trajectory (RetargeterConfig.vicar_stage2).
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.scene_point_count = 0  # scene points appended after the object's (robot_retarget.py, task_config.scene_points)
        self.step_size = step_size
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.task_constants = task_constants

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        if self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        elif getattr(self.task_constants, "OBJECT_SCENE_XML", None):
            robot_xml_path = self.task_constants.OBJECT_SCENE_XML  # opt-in object loader (TaskConfig.object_poses)
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        self._init_self_collision(self._self_collision_config)
        self._init_frame0_stand(frame0_stand)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False
        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)
        if smooth_weight_joints:  # per-joint smoothness weights in place of the scalar for the named joints
            self.smooth_weight = np.full(self.nq_a, self.smooth_weight)
            for name, w in smooth_weight_joints.items():
                adr = self.robot_model.jnt_qposadr[self.robot_model.joint(name).id]
                self.smooth_weight[list(self.q_a_indices).index(adr)] = w

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]
        self._model_limits = (self.q_a_lb.copy(), self.q_a_ub.copy())  # the MJCF ranges (vicar_stage2), before the manual bounds

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.contact_anchors = dict(np.load(contact_anchors, allow_pickle=True)) if contact_anchors else None
        self.rotation_increments = dict(np.load(rotation_increments, allow_pickle=True)) if rotation_increments else None
        self.held = dict(np.load(held_object, allow_pickle=True)) if held_object else None
        self.prior = np.load(prior)["qpos"] if prior else None
        self.hc = dict(np.load(hand_contacts, allow_pickle=True)) if hand_contacts else None
        if (self.prior is None) != (self.hc is None):
            raise ValueError("stage 2 takes prior and hand_contacts together (the latter holds its tolerances)")
        if self.hc is not None:
            adr = [self.robot_model.jnt_qposadr[self.robot_model.joint(str(n)).id] for n in self.hc["finger_joints"]]
            self._hc_fingers = [list(self.q_a_indices).index(x) for x in adr]
        self.stage2_log: list[dict] = []
        self.collision_slack = float(collision_slack)
        self.collision_log: list[dict] = []
        self.v2 = dict(np.load(vicar_stage2, allow_pickle=True)) if vicar_stage2 else None
        if self.held is not None:  # collision masks to switch per frame: held geoms vs the task object's
            names = [mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(self.robot_model.ngeom)]
            self._held_geoms = [g for g, n in enumerate(names) if n.startswith("held_")]
            self._task_object_geoms = [g for g, n in enumerate(names) if self.object_name in n and not n.startswith("held_")]
            self._geom_masks = (self.robot_model.geom_contype.copy(), self.robot_model.geom_conaffinity.copy())
            adr = [self.robot_model.jnt_qposadr[self.robot_model.joint(str(n)).id] for n in self.held.get("joint", [])]
            self._held_cols = np.isin(self.q_a_indices, adr).astype(float) if adr else np.ones(self.nq_a)  # diagonal of S

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        self._foot_lock_z_floors: dict[str, tuple[float, ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            z_floors: list[float] = []
            for window in windows:
                if len(window) == 3:
                    start, end, z = int(window[0]), int(window[1]), float(window[2])
                elif len(window) == 2:
                    start, end = int(window[0]), int(window[1])
                    z = self.foot_lock.z_floor
                else:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
                z_floors.append(z)
            self._foot_lock_windows[side] = tuple(normalized_windows)
            self._foot_lock_z_floors[side] = tuple(z_floors)

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []
        self._self_collision_skip_windows: list[tuple[int, int]] = list(sc.skip_windows)
        self._self_collision_skip_geom_pairs: set[tuple[int, int]] = set()

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        declared = {frozenset(pair) for pair in sc.pairs}
        unknown = [pair for pair in sc.skip_pairs if frozenset(pair) not in declared]
        if unknown:
            raise ValueError(f"[SelfCollision] skip_pairs not in pairs: {unknown}")
        skip = {frozenset(pair) for pair in sc.skip_pairs}

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))
                    if frozenset((body_a, body_b)) in skip:
                        self._self_collision_skip_geom_pairs.add((ga, gb))
        if skip:
            print(
                f"[SelfCollision] {len(self._self_collision_skip_geom_pairs)} geom pairs from {len(skip)} skip_pairs "
                f"not enforced in skip_windows {self._self_collision_skip_windows}"
            )

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _init_frame0_stand(self, frame0_stand: FrameZeroStandConfig | None) -> None:
        """Initialize the frame-0 stand configuration and resolve its bodies."""
        self.frame0_stand = frame0_stand or FrameZeroStandConfig()
        self._upright_body_ids: list[int] = []
        if not self.frame0_stand.enable:
            return
        for name in self.frame0_stand.upright_bodies:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"[Frame0Stand] body '{name}' not in the robot model")
            self._upright_body_ids.append(body_id)
        print(
            f"[Frame0Stand] Enabled: blend {self.frame0_stand.blend_frames} frames ({self.frame0_stand.blend_profile}), "
            f"upright {list(self.frame0_stand.upright_bodies)}, foot links {list(self.foot_links)} at "
            f"z={self.frame0_stand.z_floor}+-{self.frame0_stand.sole_tolerance}m"
        )

    def _frame0_stand_weight(self, frame_idx: int) -> float:
        """Stand weight: 1 on frame 0, falling to 0 at frame ``blend_frames`` along the blend profile."""
        cfg = self.frame0_stand
        if not cfg.enable or frame_idx < 0:
            return 0.0
        if frame_idx == 0:
            return 1.0
        if frame_idx >= cfg.blend_frames:
            return 0.0
        t = frame_idx / cfg.blend_frames
        if cfg.blend_profile == "min_jerk":
            return float(1.0 - (10 * t**3 - 15 * t**4 + 6 * t**5))
        return float(0.5 * (1.0 + np.cos(np.pi * t)))

    def _frame0_stand_pose(self, q: np.ndarray) -> np.ndarray:
        """Default stand (zero joints) with the root yaw and xy of ``q`` and the foot sticking links at z_floor."""
        q = np.copy(q)
        w, x, y, z = q[3:7] / np.linalg.norm(q[3:7])
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        q[7 : 7 + self.task_constants.ROBOT_DOF] = 0.0
        q[2] = 0.0
        foot_z = self._get_robot_link_positions(q, self.foot_links.values())[:, 2]
        q[2] = self.frame0_stand.z_floor - float(np.mean(foot_z))
        return q

    def _frame0_stand_residuals(self, q: np.ndarray):
        """Linearized stand residuals r0 + A @ dqa: foot sticking link heights minus z_floor (A_sole) and the
        horizontal components of each upright body's z axis (A_up)."""
        J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)
        r_sole = np.array([p_WF_dict[k][2] - self.frame0_stand.z_floor for k in J_WF_dict])
        A_sole = np.array([J_WF_dict[k][2] for k in J_WF_dict])

        # robot_data holds q after _calc_manipulator_jacobians; d(R e_z) = omega x (R e_z) = -[R e_z]_x omega
        m, d = self.robot_model, self.robot_data
        T = self._build_transform_qdot_to_qvel_fast()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        r_up, A_up = [], []
        for body_id in self._upright_body_ids:
            mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
            v = d.xmat[body_id].reshape(3, 3)[:, 2]
            skew_v = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            A_up.append((-skew_v @ jacr @ T)[:2, self.q_a_indices])
            r_up.append(v[:2])
        return r_sole, A_sole, np.concatenate(r_up), np.vstack(A_up)

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray | list[np.ndarray]): Demo object points in local frame.
                Single array for static points, or list of num_frames arrays for per-frame points.
            object_points_local (np.ndarray | list[np.ndarray]): Current object points in local frame.
                Single array for static points, or list of num_frames arrays for per-frame points.
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        num_frames = human_joint_motions.shape[0]
        if self.v2 is not None:
            return self._vicar_stage2(human_joint_motions, dest_res_path, object_poses, object_points_local_demo, object_points_local), [], [], []
        if isinstance(object_points_local_demo, list):
            assert len(object_points_local_demo) == num_frames, (
                f"object_points_local_demo length {len(object_points_local_demo)} != num_frames {num_frames}"
            )
        if isinstance(object_points_local, list):
            assert len(object_points_local) == num_frames, (
                f"object_points_local length {len(object_points_local)} != num_frames {num_frames}"
            )
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        if self.frame0_stand.enable and self.frame0_stand.init_from_stand:
            q = self._frame0_stand_pose(q)
        retargeted_motions = [q]

        tetrahedra = []
        obj_pts_demo_list = []  # scaled object pts
        obj_pts_list = []  # original size object pts

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                # Per-frame or static object points
                obj_pts_demo_i = (
                    object_points_local_demo[i]
                    if isinstance(object_points_local_demo, list)
                    else object_points_local_demo
                )
                obj_pts_i = object_points_local[i] if isinstance(object_points_local, list) else object_points_local

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, obj_pts_demo_i])
                )
                tetrahedra.append(source_tetrahedra)

                if self.debug:
                    # Only for visualization
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(object_quat_demo, object_trans_demo, obj_pts_demo_i)
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, obj_pts_i)

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                if self.held is not None:
                    self._switch_held(i)
                if self.prior is not None:  # stage 2: frame 0 starts from the prior's row, later frames from the last solved
                    q = np.copy(self.prior[i] if i == 0 else q)  # frame, the object where this frame has it
                    q[-7:] = q_locked_list[i][-7:]
                    self._kp_source = (human_mapped_joints, human_joint_motions[max(i - 1, 0), self.smplh_mapped_joint_indices])
                    t_solve = time.time()
                try:  # infeasible frame: keep the frames solved so far (<dest>_partial.npz, failed_frame) for review
                    q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=obj_pts_i,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                    init_t=i == 0,
                    n_iter=50 if i == 0 else 10,
                    frame_idx=i,
                )
                except RuntimeError:
                    np.savez(str(dest_res_path).replace(".npz", "_partial.npz"), qpos=np.array(retargeted_motions)[1:],
                             failed_frame=i, fps=30)
                    raise
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        q, self.laplacian_match_links.values()
                    )  # 15 X 3
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )

                if self.prior is not None:
                    self._stage2_record(q, i, time.time() - t_solve)
                elif self.collision_slack > 0:  # the last iteration's slack: how deep the solve lets it penetrate, which pair
                    s_col, names = getattr(self, "_s1_slack", (None, []))
                    v = np.asarray(s_col.value, float) if s_col is not None and s_col.value is not None else np.zeros(0)
                    self.collision_log.append({"frame": i, "slack_max_m": float(v.max(initial=0.0)), "slack_sum_m": float(v.sum()),
                                               "pair": names[int(v.argmax())] if v.size and v.max() > 0 else ""})
                    self._s1_slack = (None, [])
                retargeted_motions.append(q)
                if self.visualize and self.debug:
                    self.draw_q(q)

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

        # Save results
        np.savez(
            dest_res_path,
            qpos=np.array(retargeted_motions)[1:],
            human_joints=human_joint_motions,
            fps=30,
            cost=cost,
        )
        print("Saving results to path:", dest_res_path)
        if self.collision_log:
            np.savez(str(dest_res_path).replace(".npz", "_collision_log.npz"),
                     **{k: np.array([r[k] for r in self.collision_log]) for k in self.collision_log[0]})
        if self.prior is not None:
            log = {k: np.array([r[k] for r in self.stage2_log]) for k in self.stage2_log[0]}
            np.savez(str(dest_res_path).replace(".npz", "_stage2_log.npz"), **log)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]
        labels = ["laplacian"]  # one per constraint: the binding rows of an infeasible frame (_infeasible_rows)

        foot_rows = []  # stage 2: the foot rows get an L1 slack (_stage2)
        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        if self.prior is None:
                            constraints += [
                                Jxy @ dqa >= p_lb[:2],
                                Jxy @ dqa <= p_ub[:2],
                            ]
                        else:
                            foot_rows += [(Jxy[i], p_lb[i]) for i in range(2)] + [(-Jxy[i], -p_ub[i]) for i in range(2)]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    z_anchor = self._is_foot_locked_in_window(key, frame_idx)
                    if z_anchor is None:
                        continue

                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        labels += ["foot"] * (len(constraints) - len(labels))
        # Frame-0 stand: tight bounds on frame 0, widening with the falling stand weight over the blend
        w_stand = self._frame0_stand_weight(frame_idx)
        if w_stand > 0:
            r_sole, A_sole, r_up, A_up = self._frame0_stand_residuals(q)
            cfg = self.frame0_stand
            tol_up = cfg.angle_tolerance + (1.0 - w_stand)
            tol_sole_up = cfg.sole_tolerance + (1.0 - w_stand) * cfg.sole_release_height
            constraints += [
                A_sole @ dqa >= -r_sole - cfg.sole_tolerance,
                A_sole @ dqa <= -r_sole + tol_sole_up,
                A_up @ dqa >= -r_up - tol_up,
                A_up @ dqa <= -r_up + tol_up,
            ]

        labels += ["frame0_stand"] * (len(constraints) - len(labels))
        # Non-penetration constraints
        if self.activate_obj_non_penetration:
            Js, phis = self._update_jacobians_and_phis_from_q(q)
        else:  # no object / scene / ground rows; the geom names the self-collision rows use
            Js, phis = {}, {}
            m = self.robot_model
            self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        col_rows, col_names = [], []  # stage 2 or collision_slack: these rows get an L1 slack
        slack_rows = self.prior is not None or self.collision_slack > 0
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = -phi - self.penetration_tolerance
            if not slack_rows:
                constraints += [Ja_n @ dqa >= rhs]
                labels.append(f"non-penetration {self._pair_name(*key)} (phi {phi * 1000:.1f} mm)")
            else:
                col_rows.append((Ja_n, rhs))
                col_names.append(self._pair_name(*key))

        # Self-collision constraints
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
            rhs = self._self_collision_tolerance - phi
            if not slack_rows:
                constraints += [Ja_n @ dqa >= rhs]
                labels.append(f"self-collision {self._pair_name(key[1], key[2])} (phi {phi * 1000:.1f} mm)")
            else:
                col_rows.append((Ja_n, rhs))
                col_names.append(self._pair_name(key[1], key[2]))

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        labels += ["joint lower limit", "joint upper limit"][: len(constraints) - len(labels)]
        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]
        labels.append("step size")

        # Objective
        obj_terms = []

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        obj_terms += self._link_target_terms(q, q_t_last, frame_idx, dqa)
        if self.held is not None and self._held_on(frame_idx):  # held object: its markers onto their targets
            H = self.held
            J_H, p_H, _ = self._calc_manipulator_jacobians(q, links={str(n): str(n) for n in H["link"]}, obj_frame=False)
            for k, n in enumerate(H["link"]):
                obj_terms.append(float(H["weight"]) * cp.sum_squares((J_H[str(n)] * self._held_cols) @ dqa - (H["position"][k, frame_idx] - p_H[str(n)])))
        if self.prior is None and col_rows:  # collision_slack: the collision rows with an L1 slack, logged per pair
            s_col = cp.Variable(len(col_rows), nonneg=True)
            constraints += [np.array([r for r, _ in col_rows]) @ dqa + s_col >= np.array([b for _, b in col_rows])]
            obj_terms.append(self.collision_slack * cp.sum(s_col))
            self._s1_slack = (s_col, col_names)
        if self.prior is not None:  # stage 2: its own objective in place of the above, the collision rows hard, the hand
            base = constraints
            s2_cons, s2_terms = self._stage2(q, q_a_n_last, q_t_last, frame_idx, dqa, {"collision": col_rows, "foot": foot_rows}, hard=True)
            constraints = base + s2_cons
            obj_terms = s2_terms + self._link_target_terms(q, q_t_last, frame_idx, dqa)

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self.prior is not None:
            # stage 2 infeasible with the collision and foot rows hard: the same rows with an L1 slack, logged
            s2_cons, s2_terms = self._stage2(q, q_a_n_last, q_t_last, frame_idx, dqa, {"collision": col_rows, "foot": foot_rows}, hard=False)
            constraints = base + s2_cons
            obj_terms = s2_terms + self._link_target_terms(q, q_t_last, frame_idx, dqa)
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            if self.prior is None and not slack_rows:
                self._infeasible_rows(constraints[: len(labels)], labels, frame_idx)
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        dqa_star = dqa.value
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def _infeasible_rows(self, constraints, labels, frame_idx: int) -> None:
        """Diagnosis of an infeasible frame (stock rows, nothing is solved with it): every inequality row gets a
        non-negative slack and the total slack is minimised; prints each row that needs one (its label, the slack, and
        for joint limits the joint), with and without the step-size cone."""
        names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        adr = {int(self.robot_model.jnt_qposadr[i]): n for i, n in enumerate(names)}
        for keep_step in (True, False):
            relaxed, slacks = [], []
            for c, lab in zip(constraints, labels):
                if isinstance(c, cp.constraints.Inequality):
                    s_ = cp.Variable(c.shape, nonneg=True)
                    relaxed.append(c.args[0] <= c.args[1] + s_)
                    slacks.append((lab, s_))
                elif lab != "step size" or keep_step:
                    relaxed.append(c)
            prob = cp.Problem(cp.Minimize(sum(cp.sum(s_) for _, s_ in slacks)), relaxed)
            prob.solve(solver=cp.CLARABEL)
            rows = []
            for lab, s_ in slacks:
                v = np.atleast_1d(np.asarray(s_.value, float)) if s_.value is not None else np.zeros(0)
                for k in np.flatnonzero(v > 1e-6):
                    j = adr.get(int(self.q_a_indices[k]), f"q_a[{k}]") if "joint" in lab else ""
                    rows.append(f"{lab} {j} slack {v[k] * 1000:.2f} mm/mrad".replace("  ", " "))
            print(f"[infeasible] frame {frame_idx} ({'with' if keep_step else 'without'} the step-size cone {self.step_size}): "
                  f"phase-1 status {prob.status}, rows needing slack: {rows}", file=sys.stderr, flush=True)

    def _vicar_stage2(self, human_joint_motions, dest_res_path, object_poses, obj_pts_demo, obj_pts):
        """VICAR stage 2 (Sec. III-C2) over the whole trajectory, from `vicar_stage2`'s qpos (stage 1; the object columns
        stay), one soft objective: min sum_k w_k sum_rows (r / tol_k)^2 (each residual over its tolerance in its own unit,
        so a weighted row of 1 is a row at tolerance) subject to the MJCF joint limits and |q_t - q_t-1| <= dq_max (hard).
        Terms: L_p (|dp_human| - |dp_robot|, m/frame) and L_R (angle(dR_human) - angle(dR_robot), rad/frame) per mapped
        link frame to frame; L_s q_t - q_t-1 (rad/frame, m/frame for the root); L_anchor on each anchor's window frames
        p - p* (m) and rotvec(R* R^T) (rad) of its link; L_lap holosoma's interaction-mesh Laplacian (stage 1's: human
        joints, object and scene points in the object frame, Delaunay of the source, uniform weights), L_t V_t(q) - L_t S_t
        (m), the edges between `laplacian_mask_links` and the object's points removed on the anchor windows; L_col_<group>
        the clearance hinge max(0, -phi) (m, weight w_col_<group>) of every robot geom against the carton, table, tray, floor and itself (soft:
        penetration is a reported metric). On the anchor windows the anchor link's own geoms (the palm; the fingers stay)
        against the object are left out of L_col. Gauss-Newton: each iteration linearizes every term at the current trajectory and
        solves the convex QP over all frames at once (Clarabel; the hinge linearized exactly with a slack), holosoma's
        trust region `step` per frame, backtracking on the objective; converged when an accepted step lowers the objective
        by less than `rtol` of its value (or no step along the QP direction lowers it); `iterations` reached without that
        raises after saving. The weighted terms and their gradients in dq at the start and the solution go to the log
        (diag_*, diag_final_*). Saves the motion and <result>_vicar_log.npz."""
        V, idx, n = self.v2, self.q_a_indices, self.nq_a
        Q = np.array(V["qpos"], float)
        T = len(Q)
        links = list(self.laplacian_match_links.values())
        hum = human_joint_motions[:, self.smplh_mapped_joint_indices]
        dH = np.linalg.norm(np.diff(hum, axis=0), axis=2)  # (T-1, B)
        Rh = np.asarray(V["human_rotation"])  # (B, T, 3, 3), links order
        thH = np.stack([np.linalg.norm(Rotation.from_matrix(Rh[b, 1:] @ np.swapaxes(Rh[b, :-1], 1, 2)).as_rotvec(), axis=1) for b in range(len(links))], 1)
        KEYS = ("p", "R", "s", "anchor_m", "anchor_rad", "L", "col")
        tol = {k: float(V[f"tol_{k}"]) for k in KEYS}
        GROUPS = ("carton", "table", "tray", "floor", "self")
        w = {k: float(V[f"w_{k}"]) for k in ("p", "R", "s", "anchor", "L") + tuple(f"col_{g}" for g in GROUPS)}
        c2 = {"L_p": w["p"] / tol["p"] ** 2, "L_R": w["R"] / tol["R"] ** 2, "L_s": w["s"] / tol["s"] ** 2,
              "L_anchor_pos": w["anchor"] / tol["anchor_m"] ** 2, "L_anchor_rot": w["anchor"] / tol["anchor_rad"] ** 2,
              "L_lap": w["L"] / tol["L"] ** 2, **{f"L_col_{g}": w[f"col_{g}"] / tol["col"] ** 2 for g in GROUPS}}  # weight per squared raw residual
        step, rtol, iters = float(V["step"]), float(V["rtol"]), int(V["iterations"])
        A_link, A_win, A_p, A_R = (V[k] for k in ("anchor_link", "anchor_window", "anchor_position", "anchor_rotation"))
        lb, ub = self._model_limits
        dq_max = np.asarray(V["dq_max"], float)
        hinge = np.flatnonzero(np.isfinite(dq_max))

        lap = None
        if w["L"] > 0:  # per frame: the source Laplacian matrix (masked) and its target coordinates
            keys = list(self.laplacian_match_links)
            Vr, masked = len(keys), [keys.index(str(k)) for k in V["laplacian_mask_links"]]
            lap, n_masked = [], 0
            for t in range(T):
                od = obj_pts_demo[t] if isinstance(obj_pts_demo, list) else obj_pts_demo
                o = obj_pts[t] if isinstance(obj_pts, list) else obj_pts
                n_obj = len(od) - self.scene_point_count
                src = np.vstack([transform_points_world_to_local(object_poses[t, 3:], object_poses[t, :3], hum[t]), od])
                adj = [set(a) for a in get_adjacency_list(create_interaction_mesh(src)[1], len(src))]
                if any(w_[0] <= t <= w_[1] for w_ in A_win):
                    for i in masked:
                        for j in [j for j in adj[i] if Vr <= j < Vr + n_obj]:
                            adj[i].discard(j)
                            adj[j].discard(i)
                            n_masked += 1
                adj = [list(a) for a in adj]
                L = sp.csr_matrix(calculate_laplacian_matrix(src, adj))
                lap.append((L, L @ src, o))
            print(f"[vicar stage 2] Laplacian: {len(src)} vertices, {n_masked} hand-object edges masked", flush=True)

        global _V2_RETARGETER
        _V2_RETARGETER = self  # the frames are linearized in forked worker processes (their own MjData)
        pool = multiprocessing.get_context("fork").Pool(min(T, V2_WORKERS))

        def linearize(Q):
            """Per frame: link poses and Jacobians, collision rows (J, phi, pair, group), anchors, Laplacian vertices."""
            return pool.map(_v2_frame, [(t, Q[t]) for t in range(T)])

        def costs(Q, F):
            """Each weighted term's value, the anchor errors, and the deepest penetration per group per frame (m)."""
            P = np.array([[p[2] for p in f[0]] for f in F])
            R = np.array([[p[3] for p in f[0]] for f in F])
            dR = np.linalg.norm(np.diff(P, axis=0), axis=2)
            thR = np.array([[np.linalg.norm(Rotation.from_matrix(R[t, b] @ R[t - 1, b].T).as_rotvec()) for b in range(len(links))] for t in range(1, T)])
            e_p, e_r = np.full((T, len(A_link)), np.nan), np.full((T, len(A_link)), np.nan)
            for t, f in enumerate(F):
                for k, (_, _, p, Rl) in f[2]:
                    e_p[t, k] = np.linalg.norm(p - A_p[k, t])
                    e_r[t, k] = np.linalg.norm(Rotation.from_matrix(A_R[k, t] @ Rl.T).as_rotvec())
            pen = {g: np.array([max([max(0.0, -r[1]) for r in f[1] if r[3] == g], default=0.0) for f in F]) for g in GROUPS}
            c = {"L_p": c2["L_p"] * float(((dH - dR) ** 2).sum()), "L_R": c2["L_R"] * float(((thH - thR) ** 2).sum()),
                 "L_s": c2["L_s"] * float((np.diff(Q[:, idx], axis=0) ** 2).sum()),
                 "L_anchor": c2["L_anchor_pos"] * float(np.nansum(e_p ** 2)) + c2["L_anchor_rot"] * float(np.nansum(e_r ** 2)),
                 "L_lap": c2["L_lap"] * float(sum(np.sum((lap[t][0] @ np.vstack([f[4][0], lap[t][2]]) - lap[t][1]) ** 2) for t, f in enumerate(F))) if lap else 0.0}
            for g in GROUPS:
                c[f"L_col_{g}"] = c2[f"L_col_{g}"] * float(sum(max(0.0, -r[1]) ** 2 for f in F for r in f[1] if r[3] == g))
            c["total"] = float(sum(c.values()))
            return c, e_p, e_r, pen

        def gn_system(Q, F):
            """The least-squares terms linearized at Q: weighted A, b (residual A dq - b) and each row's term."""
            rows, cols, vals, rhs, wt, term = [], [], [], [], [], []

            def row(entries, b, weight, name):  # one residual sqrt(weight) (sum_t J_t dq_t - b)
                r = len(rhs)
                term.append(name)
                for t, J in entries:
                    nz = np.flatnonzero(J)
                    rows.extend([r] * len(nz))
                    cols.extend((t * n + nz).tolist())
                    vals.extend(J[nz].tolist())
                rhs.append(b)
                wt.append(np.sqrt(weight))
            for t in range(1, T):
                for b in range(len(links)):
                    Jp1, Jr1, p1, R1 = F[t][0][b]
                    Jp0, Jr0, p0, R0 = F[t - 1][0][b]
                    u = p1 - p0
                    nu = np.sqrt(u @ u + 1e-8)
                    row([(t, u @ Jp1 / nu), (t - 1, -(u @ Jp0) / nu)], dH[t - 1, b] - nu, c2["L_p"], "L_p")
                    rv = Rotation.from_matrix(R1 @ R0.T).as_rotvec()
                    th = np.linalg.norm(rv)
                    ax = rv / th if th > 1e-6 else np.zeros(3)
                    row([(t, ax @ Jr1), (t - 1, -(ax @ Jr0))], thH[t - 1, b] - th, c2["L_R"], "L_R")
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = 1.0
                    row([(t, e), (t - 1, -e)], -(Q[t, idx[j]] - Q[t - 1, idx[j]]), c2["L_s"], "L_s")
            for t, f in enumerate(F):
                for k, (Jp, Jr, p, Rl) in f[2]:
                    e_rot = Rotation.from_matrix(A_R[k, t] @ Rl.T).as_rotvec()
                    for i in range(3):
                        row([(t, Jp[i])], A_p[k, t, i] - p[i], c2["L_anchor_pos"], "L_anchor")
                        row([(t, Jr[i])], e_rot[i], c2["L_anchor_rot"], "L_anchor")
            for t, f in enumerate(F if lap else []):  # the robot links (object frame) and the object / scene points
                L, target, o = lap[t]
                Vt = np.vstack([f[4][0], o])
                JL = np.asarray(sp.kron(L[:, : len(f[4][0])], sp.eye(3)).tocsr() @ f[4][1])
                res = (target - L @ Vt).reshape(-1)
                for i in np.flatnonzero(np.abs(JL).sum(1) > 0):  # rows without a robot vertex are constant
                    row([(t, JL[i])], res[i], c2["L_lap"], "L_lap")
            W = sp.diags(wt)
            A = W @ sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), T * n))
            return A, W @ np.array(rhs), np.array(term)

        def col_rows(F):
            C_rows, C_cols, C_vals, phi, wr = [], [], [], [], []
            for t, f in enumerate(F):
                for r in f[1]:
                    nz = np.flatnonzero(r[0])
                    C_rows.extend([len(phi)] * len(nz))
                    C_cols.extend((t * n + nz).tolist())
                    C_vals.extend(r[0][nz].tolist())
                    phi.append(r[1])
                    wr.append(c2[f"L_col_{r[3]}"])
            return sp.csr_matrix((C_vals, (C_rows, C_cols)), shape=(len(phi), T * n)), np.array(phi), np.array(wr)

        def diag(Q, F):
            """Each weighted term's value and gradient in dq (T, n) at Q; L_col per group."""
            A, b, term = gn_system(Q, F)
            c = costs(Q, F)[0]
            names, val, grad = [], [], []
            for k in ("L_p", "L_R", "L_s", "L_anchor", "L_lap"):
                sel = term == k
                names.append(k)
                val.append(c[k])
                grad.append((-2.0 * (A[sel].T @ b[sel])).reshape(T, n))
            for g in GROUPS:
                gr = np.zeros((T, n))
                for t, f in enumerate(F):
                    for r in f[1]:
                        if r[3] == g:
                            gr[t] -= 2.0 * c2[f"L_col_{g}"] * max(0.0, -r[1]) * r[0]
                names.append(f"L_col_{g}"); val.append(c[f"L_col_{g}"]); grad.append(gr)
            return np.array(names), np.array(val), np.array(grad)

        history = []
        t_start = time.time()
        F = linearize(Q)
        c, e_p0, e_r0, pen = costs(Q, F)
        pen0 = pen
        diag0 = diag(Q, F)
        print("[vicar stage 2] start terms: " + ", ".join(f"{k} {v:.5g} |g| {np.linalg.norm(g):.4g}" for k, v, g in zip(*diag0)), flush=True)
        if "priority" in V:  # the priority hierarchy (_v2_priority) in place of the weighted sum
            return self._v2_priority(V, Q, F, linearize, gn_system, costs, pool, dest_res_path, human_joint_motions, tol, GROUPS, t_start)
        status = "not converged"
        lim = np.flatnonzero(lb > -1e5)
        for it in range(iters):
            history.append(c | {f"pen_{g}_mm": 1000 * float(pen[g].max()) for g in GROUPS} | {"time_s": time.time() - t_start})
            print(f"[vicar stage 2] iteration {it}: " + ", ".join(f"{k} {v:.5g}" for k, v in history[-1].items()), flush=True)
            A, b, _ = gn_system(Q, F)
            C, phi, wr = col_rows(F)
            x = cp.Variable(T * n)
            X = cp.reshape(x, (T, n), order="C")
            Qa = Q[:, idx]
            P = (A.T @ A).tocsc()
            obj = cp.quad_form(x, cp.psd_wrap(P)) - 2.0 * (A.T @ b) @ x
            cons = [cp.norm(X, 2, axis=1) <= step]
            cons += [X[:, lim] >= lb[lim] - Qa[:, lim], X[:, lim] <= ub[lim] - Qa[:, lim]]
            dQ = np.diff(Qa, axis=0)[:, hinge]
            cons += [cp.abs(X[1:, hinge] - X[:-1, hinge] + dQ) <= dq_max[hinge]]
            if len(phi):  # the clearance hinge: s >= max(0, -(phi + C dq))
                s = cp.Variable(len(phi), nonneg=True)
                cons.append(C @ x + phi + s >= 0)
                obj = obj + cp.sum_squares(cp.multiply(np.sqrt(wr), s))
            prob = cp.Problem(cp.Minimize(obj), cons)
            t0 = time.time()
            prob.solve(solver=cp.CLARABEL)
            print(f"[vicar stage 2] QP {time.time() - t0:.1f} s, {len(phi)} collision rows", flush=True)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"vicar stage 2 iteration {it}: QP {prob.status}")
            dx = np.asarray(x.value).reshape(T, n)
            alpha = 1.0
            for _ in range(10):  # backtracking on the objective
                Q_new = Q.copy()
                Q_new[:, idx] += alpha * dx
                Q_new[:, 3:7] /= np.linalg.norm(Q_new[:, 3:7], axis=1, keepdims=True)
                F_new = linearize(Q_new)
                c_new, _, _, pen_new = costs(Q_new, F_new)
                if c_new["total"] < c["total"]:
                    break
                alpha /= 2
            history[-1]["alpha"] = alpha
            if c_new["total"] >= c["total"]:
                status = f"converged: no decrease along the QP step at iteration {it}"
                break
            decrease = c["total"] - c_new["total"]
            Q, F, c, pen = Q_new, F_new, c_new, pen_new
            if decrease < rtol * c["total"]:
                status = f"converged: relative decrease {decrease / c['total']:.2e} < rtol {rtol:g} at iteration {it + 1}"
                break
        c, e_p, e_r, pen = costs(Q, F)
        history.append(c | {f"pen_{g}_mm": 1000 * float(pen[g].max()) for g in GROUPS} | {"time_s": time.time() - t_start, "alpha": np.nan})
        diag1 = diag(Q, F)
        pool.close()
        print(f"[vicar stage 2] {status}; final: " + ", ".join(f"{k} {v:.5g}" for k, v in history[-1].items()), flush=True)
        np.savez(dest_res_path, qpos=Q, human_joints=human_joint_motions, fps=30, cost=c["total"])
        keys = list(history[-1])
        np.savez(str(dest_res_path).replace(".npz", "_vicar_log.npz"), cost_names=np.array(keys),
                 cost=np.array([[h.get(k, np.nan) for k in keys] for h in history]), status=np.array(status),
                 converged=np.array(status.startswith("converged")), anchor_pos_m=e_p, anchor_rot_rad=e_r,
                 anchor_pos_m_start=e_p0, anchor_rot_rad_start=e_r0, **{f"pen_{g}_m": pen[g] for g in GROUPS},
                 **{f"pen_{g}_m_start": pen0[g] for g in GROUPS},
                 tol_names=np.array(KEYS), tol=np.array([tol[k] for k in KEYS]), weight_names=np.array(list(w)),
                 weight=np.array(list(w.values())), diag_names=diag0[0], diag_value=diag0[1], diag_grad=diag0[2],
                 diag_final_value=diag1[1], diag_final_grad=diag1[2], q_a_indices=idx)
        if iters and not status.startswith("converged"):
            raise RuntimeError(f"vicar stage 2: {iters} iterations without convergence (rtol {rtol:g}); saved {dest_res_path}")
        return Q

    def _v2_linearize_frame(self, t: int, q: np.ndarray):
        """vicar_stage2 at frame t, configuration q: the mapped links' poses and Jacobians, the collision rows
        (J, phi, pair, group) without the anchor link's own geoms against the object on its window, the anchors' (k, pose
        and Jacobians) on their window, and the Laplacian vertices."""
        idx, A, m = self.q_a_indices, self.v2, self.robot_model
        pose = [self._link_pose_jacobians(q, link) for link in self.laplacian_match_links.values()]
        self._row_ids = {}
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        Jsc, phisc = self._compute_self_collision_constraints(t)
        col = [(Jsc[k][idx], phisc[k], self._pair_name(k[1], k[2]), "self") for k in phisc]
        palm = {m.body(str(A["anchor_link"][k])).id for k in range(len(A["anchor_link"])) if A["anchor_window"][k][0] <= t <= A["anchor_window"][k][1]}
        for k in phis:
            name = self._pair_name(*k)
            group = ("floor" if "ground" in name else "table" if "obstacle_table" in name else "tray" if "obstacle_" in name
                     else "carton" if self.object_name in name else None)
            if group is None:
                raise RuntimeError(f"vicar stage 2: collision pair {name} has no group")
            if group == "carton" and {int(m.geom_bodyid[k[0]]), int(m.geom_bodyid[k[1]])} & palm:
                continue  # the anchored palm (the anchor link's own geoms) against the object on the anchor window
            for J, ph in zip(np.atleast_2d(Js[k][..., idx]), np.atleast_1d(phis[k])):
                col.append((J, float(ph), name, group))
        anc = [(k, self._link_pose_jacobians(q, str(A["anchor_link"][k]))) for k in range(len(A["anchor_link"]))
               if A["anchor_window"][k][0] <= t <= A["anchor_window"][k][1]]
        lap = None
        if float(A["w_L"]) > 0:  # the mapped links in the object frame and their Jacobians (3B, n)
            J_OC, p_OC, _ = self._calc_manipulator_jacobians(q, links=self.laplacian_match_links, obj_frame=True)
            lap = (np.array([p_OC[k] for k in self.laplacian_match_links]), np.vstack([J_OC[k] for k in self.laplacian_match_links]))
        ext = None  # priority: the foot and torso links' poses and Jacobians
        if "priority" in A:
            ext = [self._link_pose_jacobians(q, str(link)) for link in list(A["foot_links"]) + list(A["torso_links"])]
        return pose, col, anc, ext, lap

    def _v2_priority(self, V, Q, F, linearize, gn_system, costs, pool, dest_res_path, human_joint_motions, tol, GROUPS, t_start):
        """vicar_stage2 with `priority` set: a strict priority hierarchy over the stage-2 terms in place of the weighted sum
        (tools/hs2_priority.md). Levels, highest first: P1 the palm anchor (L_anchor), the planted feet (L_foot: on each run
        of `planted` the foot link (`foot_links`) at the run's stage-1 mean xy and yaw, flat as with its hinges at 0, its sole on the
        floor) and the clearance hinge (L_col, every group); P2 the torso direction, pelvis to the shoulder midpoint
        (`torso_links`), vs the human's (`torso_up_human`, L_torso); P3 L_lap, L_p, L_R; P4 L_s. Every row is its residual
        over its tolerance; inside a level the rows' squares add. Gauss-Newton: each iteration linearizes all terms and
        solves one QP per level, highest first (Clarabel): level k minimizes its rows subject to the hard joint limits, joint
        speeds and trust region (collision rows by exact constraint generation from the rows the previous step could
        reach), and every higher level's rows held at |achieved r| + 1, one tolerance of slack (the clearance rows:
        penetration at most achieved + tol_col); the step is the last level's. Collision rows that cannot activate inside the trust
        region (phi > |C_row| step) are left out (exact). Backtracking on the lexicographic merit (levels 1-3: the sum over
        their rows of max(0, |r| - 1)^2, the part beyond tolerance; level 4: L_s's sum of squares), a level counting as
        unchanged within rtol of its value; converged when no step along the direction is lexicographically better;
        `iterations` reached raises after saving. Logs each level's residuals per term (levels_json)."""
        import json

        import clarabel

        idx, n, T, m, d = self.q_a_indices, self.nq_a, len(Q), self.robot_model, self.robot_data
        step, rtol, iters = float(V["step"]), float(V["rtol"]), int(V["iterations"])
        lb, ub = self._model_limits
        lim = np.flatnonzero(lb > -1e5)
        dq_max = np.asarray(V["dq_max"], float)
        hinge = np.flatnonzero(np.isfinite(dq_max))
        foot_links, torso_links = [str(x) for x in V["foot_links"]], [str(x) for x in V["torso_links"]]
        planted, up_h = np.asarray(V["planted"], bool), np.asarray(V["torso_up_human"], float)
        tf_m, tf_r, tt, tc = float(V["tol_foot_m"]), float(V["tol_foot_rad"]), float(V["tol_torso_rad"]), tol["col"]
        if planted.shape != (len(foot_links), T) or up_h.shape != (T, 3):
            raise RuntimeError(f"vicar stage 2 priority: planted {planted.shape}, torso_up_human {up_h.shape}, {T} frames")
        LEVELS = (("L_anchor", "L_foot"), ("L_torso",), ("L_lap", "L_p", "L_R"), ("L_s",))

        # planted-foot targets: the sole height of each foot link with every hinge at 0 (legs straight, foot flat), each stance run's
        # stage-1 mean xy and yaw (about z, from the qpos0 orientation)
        ground = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "ground"]
        if len(ground) != 1:
            raise RuntimeError(f"vicar stage 2 priority: {len(ground)} geoms named ground")
        d0 = mujoco.MjData(m)  # every hinge at 0 (legs straight, feet flat), each free body upright, clear of the floor
        for k, j in enumerate(np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)):
            d0.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7] = [0.0, 3.0 * k, 2.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d0)
        ft = np.zeros(6)
        foot_P, foot_R = np.full((len(foot_links), T, 3), np.nan), np.full((len(foot_links), T, 3, 3), np.nan)
        for k, l in enumerate(foot_links):
            b0 = m.body(l).id
            gs = [g for g in range(m.ngeom) if g != ground[0] and (int(m.geom_bodyid[g]) == b0 or b0 in self._ancestors(int(m.geom_bodyid[g])))]
            dist = {g: mujoco.mj_geomDistance(m, d0, g, ground[0], 10.0, ft) for g in gs}
            g_low = min(dist, key=dist.get)
            z_sole = float(d0.xpos[b0][2] - dist[g_low])
            print(f"[vicar stage 2 priority] {l}: z {d0.xpos[b0][2]:.4f}, lowest geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_low)} of {len(gs)}", flush=True)
            R0 = d0.xmat[b0].reshape(3, 3).copy()
            edges = np.flatnonzero(np.diff(np.r_[0, planted[k].astype(int), 0]))
            for s, e in zip(edges[::2], edges[1::2]):
                poses = [self._link_pose_jacobians(Q[t], l)[2:] for t in range(s, e)]
                yaw = [np.arctan2(*(R @ R0.T)[[1, 0], 0]) for _, R in poses]
                foot_P[k, s:e] = [*np.mean([p[:2] for p, _ in poses], 0), z_sole]
                foot_R[k, s:e] = Rotation.from_euler("z", np.arctan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw)))).as_matrix() @ R0
            print(f"[vicar stage 2 priority] {l}: sole height {z_sole:.4f} m, {len(edges) // 2} stance runs, {int(planted[k].sum())} frames", flush=True)

        def extra_rows(F):
            """L_foot and L_torso rows, residual over tolerance: weighted A, b (residual A dq - b) and each row's term."""
            rows, cols, vals, rhs, term = [], [], [], [], []

            def row(t, J, b, s, name):
                nz = np.flatnonzero(J)
                rows.extend([len(rhs)] * len(nz))
                cols.extend((t * n + nz).tolist())
                vals.extend((s * J[nz]).tolist())
                rhs.append(s * b)
                term.append(name)
            nf = len(foot_links)
            for t, f in enumerate(F):
                for k in range(nf):
                    if planted[k, t]:
                        Jp, Jr, p, R = f[3][k]
                        er = Rotation.from_matrix(foot_R[k, t] @ R.T).as_rotvec()
                        for i in range(3):
                            row(t, Jp[i], foot_P[k, t, i] - p[i], 1 / tf_m, "L_foot")
                        for i in range(3):
                            row(t, Jr[i], er[i], 1 / tf_r, "L_foot")
                (JP, _, pP, _), (JL, _, pL, _), (JR, _, pR, _) = f[3][nf:nf + 3]
                v = 0.5 * (pL + pR) - pP
                u = v / np.linalg.norm(v)
                Ju = (np.eye(3) - np.outer(u, u)) / np.linalg.norm(v) @ (0.5 * (JL + JR) - JP)
                for i in range(3):
                    row(t, Ju[i], up_h[t, i] - u[i], 1 / tt, "L_torso")
            return sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), T * n)), np.array(rhs), np.array(term)

        def system(Q, F):
            A, b, term = gn_system(Q, F)
            A2, b2, t2 = extra_rows(F)
            C_r, C_c, C_v, phi, grp, frm, cn, n_all = [], [], [], [], [], [], [], 0
            for t, f in enumerate(F):
                for r in f[1]:
                    n_all += 1
                    if r[1] > np.linalg.norm(r[0]) * step:
                        continue  # cannot reach phi < 0 inside the trust region
                    nz = np.flatnonzero(r[0])
                    C_r.extend([len(phi)] * len(nz))
                    C_c.extend((t * n + nz).tolist())
                    C_v.extend(r[0][nz].tolist())
                    phi.append(r[1])
                    grp.append(r[3])
                    frm.append(t)
                    cn.append(np.linalg.norm(r[0]))
            C = sp.csr_matrix((C_v, (C_r, C_c)), shape=(len(phi), T * n))
            pen = {g: np.array([max([max(0.0, -r[1]) for r in f[1] if r[3] == g], default=0.0) for f in F]) for g in GROUPS}
            return dict(A=sp.vstack([A, A2]).tocsr(), b=np.r_[b, b2], term=np.r_[term, t2], C=C, phi=np.array(phi),
                        grp=np.array(grp), frm=np.array(frm, int), cn=np.array(cn), n_col=n_all, pen=pen)

        def level_r(S, k):
            r = -S["b"][np.isin(S["term"], LEVELS[k])]
            return np.r_[r, np.maximum(0.0, -S["phi"]) / tc] if k == 0 else r

        def merit(S):
            return np.array([float(np.sum(np.maximum(0.0, np.abs(level_r(S, k)) - 1.0) ** 2)) for k in range(3)]
                            + [float(np.sum(level_r(S, 3) ** 2))])

        def lex(new, old):
            for a, b in zip(new, old):
                if a > b + rtol * b:
                    return 1
                if a < b - rtol * b:
                    return -1
            return 0

        def report(S):
            r = -S["b"]
            out = {}
            for k, L in enumerate(LEVELS):
                for name in L:
                    x = r[S["term"] == name]
                    out[f"P{k + 1} {name}"] = {"rows": int(len(x)), "rms": float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0,
                                               "max": float(np.abs(x).max()) if len(x) else 0.0,
                                               "beyond_tol": float(np.mean(np.abs(x) > 1)) if len(x) else 0.0,
                                               "excess": float(np.sum(np.maximum(0.0, np.abs(x) - 1) ** 2))}
            for g in GROUPS:
                x = np.maximum(0.0, -S["phi"][S["grp"] == g]) / tc
                out[f"P1 L_col_{g}"] = {"rows": int(len(x)), "max": float(x.max()) if len(x) else 0.0,
                                        "beyond_tol": float(np.mean(x > 1)) if len(x) else 0.0,
                                        "excess": float(np.sum(np.maximum(0.0, x - 1) ** 2)), "max_pen_mm": float(1000 * S["pen"][g].max())}
            return out

        def hard_rows(Q):
            N, Qa, I = T * n, Q[:, idx], sp.eye(T * n, format="csr")
            sl = (np.arange(T)[:, None] * n + lim[None]).ravel()
            sv = (np.arange(T - 1)[:, None] * n + hinge[None]).ravel()
            D = I[sv + n] - I[sv]
            dQ = np.diff(Qa, axis=0)[:, hinge].ravel()
            vm = np.tile(dq_max[hinge], T - 1)
            S_rows = sp.block_diag([sp.vstack([sp.csr_matrix((1, n)), -sp.eye(n)])] * T)  # (step, dq_t) in the SOC per frame
            s_b = np.zeros((n + 1) * T)
            s_b[::n + 1] = step
            return (sp.vstack([I[sl], -I[sl], D, -D]).tocsr(), np.r_[(ub[lim] - Qa[:, lim]).ravel(), (Qa[:, lim] - lb[lim]).ravel(), vm - dQ, vm + dQ],
                    S_rows.tocsr(), s_b)

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        def solve_levels(Q, S, reach):
            """One QP per level, highest first, each holding the higher levels; the last level's step and each level's
            linearized sum of squares at its own QP and at the step. Collision rows by constraint generation (exact):
            each QP starts from the penetrating rows and adds every left-out row its solution violates, until none."""
            N = T * n
            Gh, hh, Ssoc, sb = hard_rows(Q)
            A, b, term, C, phi = S["A"], S["b"], S["term"], S["C"], S["phi"]
            act = phi <= 0.0
            s_star = None
            bG, bh, own, times, xs = [], [], [], [], []
            for k, L in enumerate(LEVELS):
                sel = np.isin(term, L)
                Ak, bk = A[sel], b[sel]
                P0, q0 = 2.0 * (Ak.T @ Ak), -2.0 * (Ak.T @ bk)
                t0, rounds = time.time(), 0
                while True:
                    rounds += 1
                    ia = np.flatnonzero(act)
                    Ca, pa = C[ia], phi[ia]
                    G, h = sp.vstack([Gh] + bG).tocsr(), np.concatenate([hh] + bh)
                    if k == 0:  # the clearance hinge: s >= 0, C dq + phi + s >= 0, cost (s / tol_col)^2
                        ms = len(ia)
                        P = sp.block_diag([P0, sp.eye(ms) * 2.0 / tc ** 2])
                        q = np.r_[np.asarray(q0).ravel(), np.zeros(ms)]
                        G = sp.bmat([[G, None], [-Ca, -sp.eye(ms)], [None, -sp.eye(ms)]])
                        h = np.r_[h, pa, np.zeros(ms)]
                        Ssk = sp.hstack([Ssoc, sp.csr_matrix((Ssoc.shape[0], ms))])
                    else:  # held clearance: penetration at most P1's + tol_col
                        ms, P, q, Ssk = 0, P0, np.asarray(q0).ravel(), Ssoc
                        G, h = sp.vstack([G, -Ca]), np.r_[h, pa + s_star[ia] + tc]
                    sol = clarabel.DefaultSolver(sp.triu(P).tocsc(), q, sp.vstack([G, Ssk]).tocsc(), np.r_[h, sb],
                                                 [clarabel.NonnegativeConeT(G.shape[0])] + [clarabel.SecondOrderConeT(n + 1)] * T, settings).solve()
                    if str(sol.status) not in ("Solved", "AlmostSolved", "InsufficientProgress"):  # cvxpy's OPTIMAL(_INACCURATE), as the weighted path
                        raise RuntimeError(f"vicar stage 2 priority: level P{k + 1} QP {sol.status}")
                    z = np.array(sol.x)
                    x = z[:N]
                    g = C @ x + phi
                    bad = ~act & (g < (0.0 if k == 0 else -(s_star + tc)))
                    if not bad.any():
                        break
                    act |= bad
                times.append(time.time() - t0)
                print(f"[vicar stage 2 priority] P{k + 1} QP {times[-1]:.1f} s, {G.shape[0]} rows, {int(act.sum())} / {len(act)} collision rows, "
                      f"{rounds} rounds, {sol.status}", flush=True)
                rk = Ak @ x - bk
                if k == 0:
                    s_star = np.maximum(0.0, -g)  # each row's penetration at P1's step (its slack at the optimum)
                own.append(float(rk @ rk) + (float(np.sum(s_star ** 2)) / tc ** 2 if k == 0 else 0.0))
                xs.append((Ak, bk))
                u = np.abs(rk) + 1.0  # held: no row more than one tolerance worse than this level's QP left it
                bG += [Ak, -Ak]
                bh += [bk + u, u - bk]
            at_step = [float(np.sum((Ak @ x - bk) ** 2)) for Ak, bk in xs]
            at_step[0] += float(np.sum(np.maximum(0.0, -(C @ x + phi)) ** 2)) / tc ** 2
            return x.reshape(T, n), own, at_step, times, G.shape[0]

        S = system(Q, F)
        mer = merit(S)
        rep0 = report(S)
        history, status = [], "not converged"
        print("[vicar stage 2 priority] start merit " + ", ".join(f"P{k + 1} {v:.5g}" for k, v in enumerate(mer)), flush=True)
        reach = np.full(T, step)  # per frame: how far the previous step went (the trust region at first)
        for it in range(iters):
            dx, own, at_step, times, n_rows = solve_levels(Q, S, reach)
            alpha = 1.0
            for _ in range(10):  # backtracking on the lexicographic merit
                Q_new = Q.copy()
                Q_new[:, idx] += alpha * dx
                Q_new[:, 3:7] /= np.linalg.norm(Q_new[:, 3:7], axis=1, keepdims=True)
                F_new = linearize(Q_new)
                S_new = system(Q_new, F_new)
                mer_new = merit(S_new)
                if lex(mer_new, mer) < 0:
                    break
                alpha /= 2
            h = {"it": it, **{f"merit_P{k + 1}": float(v) for k, v in enumerate(mer)}, **{f"qp_P{k + 1}": o for k, o in enumerate(own)},
                 **{f"step_P{k + 1}": o for k, o in enumerate(at_step)}, **{f"qp_s_P{k + 1}": s for k, s in enumerate(times)},
                 "alpha": alpha, "collision_rows": len(S["phi"]), "collision_rows_all": S["n_col"], "time_s": time.time() - t_start}
            history.append(h)
            print("[vicar stage 2 priority] iteration " + ", ".join(f"{k} {v:.5g}" for k, v in h.items()), flush=True)
            if lex(mer_new, mer) >= 0:
                status = f"converged: no lexicographic decrease along the step at iteration {it}"
                break
            Q, F, S, mer = Q_new, F_new, S_new, mer_new
            reach = alpha * np.linalg.norm(dx, axis=1)
            np.savez(str(dest_res_path).replace(".npz", "_checkpoint.npz"), qpos=Q, human_joints=human_joint_motions, fps=30, iteration=it + 1)
        rep = report(S)
        c, e_p, e_r, pen = costs(Q, F)
        pool.close()
        print(f"[vicar stage 2 priority] {status}; merit " + ", ".join(f"P{k + 1} {v:.5g}" for k, v in enumerate(mer)), flush=True)
        for k, v in rep.items():
            print(f"[vicar stage 2 priority] {k}: " + ", ".join(f"{a} {b:.4g}" for a, b in v.items()), flush=True)
        np.savez(dest_res_path, qpos=Q, human_joints=human_joint_motions, fps=30, cost=float(mer.sum()))
        keys = sorted({k for h in history for k in h})
        np.savez(str(dest_res_path).replace(".npz", "_vicar_log.npz"), cost_names=np.array(keys),
                 cost=np.array([[h.get(k, np.nan) for k in keys] for h in history] + [[np.nan] * len(keys)]).reshape(len(history) + 1, len(keys)),
                 status=np.array(status), converged=np.array(status.startswith("converged")), anchor_pos_m=e_p, anchor_rot_rad=e_r,
                 **{f"pen_{g}_m": pen[g] for g in GROUPS}, merit=mer, levels_json=np.array(json.dumps(rep)),
                 levels_start_json=np.array(json.dumps(rep0)), level_terms=np.array(json.dumps(LEVELS)),
                 foot_target_position=foot_P, foot_target_rotation=foot_R, planted=planted, q_a_indices=idx)
        if iters and not status.startswith("converged"):
            raise RuntimeError(f"vicar stage 2 priority: {iters} iterations without convergence; saved {dest_res_path}")
        return Q

    def _ancestors(self, b: int) -> list[int]:
        out = []
        while b:
            b = int(self.robot_model.body_parentid[b])
            out.append(b)
        return out

    def _link_pose_jacobians(self, q: np.ndarray, link: str):
        """World position and rotation of a body at q, and their Jacobians in dq_a (3 x nq_a each)."""
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        body = self.robot_model.body(link).id
        Jp = np.zeros((3, self.robot_model.nv))
        Jr = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacBody(self.robot_model, self.robot_data, Jp, Jr, body)
        T = self._build_transform_qdot_to_qvel_fast()
        R = self.robot_data.xmat[body].reshape(3, 3).copy()
        return (Jp @ T)[:, self.q_a_indices], (Jr @ T)[:, self.q_a_indices], self.robot_data.xpos[body].copy(), R

    def _link_target_terms(self, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int, dqa):
        """Cost terms of the opt-in contact anchors and rotation increments on this frame (linearized at q)."""
        terms = []
        A = self.contact_anchors
        if A is not None:
            for k, link in enumerate(A["link"]):
                if A["window"][k][0] <= frame_idx <= A["window"][k][1]:
                    Jp, Jr, p, R = self._link_pose_jacobians(q, str(link))
                    R_t = A["rotation"][k, frame_idx]
                    if "parent" in A and str(A["parent"][k]):  # a rotation target relative to the parent body
                        _, Jr_p, _, R_p = self._link_pose_jacobians(q, str(A["parent"][k]))
                        R_t, Jr = R_p @ R_t, Jr - Jr_p
                    e_rot = Rotation.from_matrix(R_t @ R.T).as_rotvec()
                    w_p, w_r = (A[key][k][frame_idx] if np.ndim(A[key][k]) else A[key][k] for key in ("pos_weight", "rot_weight"))
                    terms.append(w_p * cp.sum_squares(Jp @ dqa - (A["position"][k, frame_idx] - p)))
                    terms.append(w_r * cp.sum_squares(Jr @ dqa - e_rot))
        S = self.rotation_increments
        if S is not None and frame_idx > 0:
            for k, link in enumerate(S["link"]):
                R_last = self._link_pose_jacobians(q_t_last, str(link))[3]
                _, Jr, _, R = self._link_pose_jacobians(q, str(link))
                R_s = S["rotation"][k]
                e_rot = Rotation.from_matrix(R_s[frame_idx] @ R_s[frame_idx - 1].T @ R_last @ R.T).as_rotvec()
                terms.append(S["weight"][k] * cp.sum_squares(Jr @ dqa - e_rot))
        return terms
    def _pair_name(self, g1: int, g2: int) -> str:
        """`body:geom` of the two geoms of a collision row."""
        m = self.robot_model
        return " / ".join(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''}:{self._geom_names[g]}" for g in (g1, g2))

    def _stage2(self, q, q_a_n_last, q_t_last, frame_idx: int, dqa, rows: dict, hard: bool):
        """Stage 2 (RetargeterConfig.prior, hand_contacts) on this frame, linearized at q: (constraints, cost terms).
        Costs, each residual over its tolerance and squared: near the prior row, the change from the previous solved
        frame, every mapped keypoint's displacement from the previous solved frame against the source's (L_p), the soft
        wrist and finger targets at their per-row strength and, on the hold rows, every hand link's point onto its mapped
        point (Morphometric Imitation's closest contact, tolerance `sigma_contact` per link). The collision and
        foot-sticking rows (`rows`: name -> [(row, rhs)] for row @ dq >= rhs) are hard, or with `hard` False (the
        frame was infeasible) carry an L1 slack, logged. On the hold rows the holding check (C_hold): forces only on the
        links whose gap to their face is at most `force_gap` at q, the others' fixed at 0."""
        H, tol = self.hc, (lambda k: float(self.hc[k]))
        cons, terms, v = [], [], {}
        terms.append(cp.sum_squares(cp.multiply(1.0 / H["sigma_prior"], dqa + q_a_n_last - self.prior[frame_idx, self.q_a_indices])))
        if frame_idx > 0:
            terms.append(cp.sum_squares(dqa - (q_t_last[self.q_a_indices] - q_a_n_last)) / tol("sigma_step_rad") ** 2)
        if frame_idx > 0 and "sigma_keypoint_m" in H:  # optional L_p
            J, p, _ = self._calc_manipulator_jacobians(q, links=self.laplacian_match_links, obj_frame=False)
            _, p_last, _ = self._calc_manipulator_jacobians(q_t_last, links=self.laplacian_match_links, obj_frame=False)
            h_now, h_prev = self._kp_source
            for i, k in enumerate(self.laplacian_match_links):
                terms.append(cp.sum_squares(J[k] @ dqa + p[k] - p_last[k] - (h_now[i] - h_prev[i])) / tol("sigma_keypoint_m") ** 2)
        for name, rr in rows.items():
            if rr and hard:
                cons.append(np.array([r for r, _ in rr]) @ dqa >= np.array([b for _, b in rr]))
            elif rr:
                v[name] = cp.Variable(len(rr), nonneg=True)
                cons.append(np.array([r for r, _ in rr]) @ dqa + v[name] >= np.array([b for _, b in rr]))
                terms.append(tol("slack_weight_m") * cp.sum(v[name]))
        R_o, o = Rotation.from_quat(q[[-3, -2, -1, -4]]).as_matrix(), q[-7:-4]
        w = float(H["wrist_weight"][frame_idx])
        if w > 0:  # the wrist onto T_o E_wrist
            Jp, Jr, p, R = self._link_pose_jacobians(q, str(H["wrist_link"]))
            E = H["E_wrist"][frame_idx]
            terms.append(w * cp.sum_squares(Jp @ dqa - (R_o @ E[:3, 3] + o - p)) / tol("sigma_wrist_m") ** 2)
            terms.append(w * cp.sum_squares(Jr @ dqa - Rotation.from_matrix(R_o @ E[:3, :3] @ R.T).as_rotvec()) / tol("sigma_wrist_rad") ** 2)
        w = float(H["finger_weight"][frame_idx])
        if w > 0:
            f_idx = self._hc_fingers
            terms.append(w * cp.sum_squares(dqa[f_idx] + q_a_n_last[f_idx] - H["finger_target"][frame_idx]) / tol("sigma_finger_rad") ** 2)
        cw = float(H["contact_weight"][frame_idx]) if "contact_weight" in H else float(int(H["window"][0]) <= frame_idx <= int(H["window"][1]))
        hold = int(H["window"][0]) <= frame_idx <= int(H["window"][1])
        if cw > 0:
            K, nf = len(H["link"]), len(self._hc_fingers)
            s_w, s_t = (cp.Variable(n, nonneg=True) for n in (6, nf))
            f = cp.Variable((K, 3))
            Jc, touching = [], []
            for k, link in enumerate(H["link"]):  # closest contact: every link's point toward its mapped point c
                _, _, p_l, R_l = self._link_pose_jacobians(q, str(link))
                p = p_l + R_l @ H["r_link"][k]
                Jx = R_o.T @ self._calc_contact_jacobian_from_point(self.robot_model.body(str(link)).id, p, input_world=True)[:, self.q_a_indices]
                x, n = R_o.T @ (p - o), H["n"][k]
                terms.append(cw * cp.sum_squares(x + Jx @ dqa - H["c"][k]) / float(H["sigma_contact"][k]) ** 2)
                Jc.append(Jx[:, self._hc_fingers])
                touching.append(float(n @ x - H["d"][k]) <= tol("force_gap"))
            cons += [f[k] == 0 for k in range(K) if not touching[k] or not hold]  # a hovering link carries no load
            # C_hold: the forces that carry the object (G f = wrench) in friction cones, normal force and finger torque limits
            fn = cp.hstack([-H["n"][k] @ f[k] for k in range(K)])
            G = np.zeros((6, 3 * K))
            for k in range(K):
                r = H["c"][k] - H["com"]
                G[:3, 3 * k:3 * k + 3] = np.eye(3)
                G[3:, 3 * k:3 * k + 3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            cons += [fn >= 0, fn <= tol("f_max")] + [cp.norm(f[k] + fn[k] * H["n"][k]) <= tol("mu") * fn[k] for k in range(K)]
            cons += [cp.abs(sum(Jc[k].T @ f[k] for k in range(K))) <= H["tau_max"] + s_t,
                     cp.abs(G @ cp.vec(f.T, order="F") - (H["wrench"][frame_idx] if hold else 0)) <= s_w]
            terms += [tol("slack_weight_n") * (cp.sum(s_w) + cp.sum(s_t)), cp.sum_squares(f) / tol("f_max") ** 2]
            v |= {"wrench": s_w, "torque": s_t, "force": f, "touching": np.array(touching)}
        self._s2_vars = v
        return cons, terms

    def _stage2_record(self, q, frame_idx: int, solve_s: float) -> None:
        """One row of the stage-2 log: the last iteration's slack per constraint group and forces, and at the solution
        each contact link's distance to its c and gap to its face, and the wrist's distance to its target."""
        H, v = self.hc, self._s2_vars
        val = lambda k, shape: np.asarray(v[k].value, float) if k in v and getattr(v[k], "value", None) is not None else np.full(shape, np.nan)  # noqa: E731
        K, nf = len(H["link"]), len(self._hc_fingers)
        R_o, o = Rotation.from_quat(q[[-3, -2, -1, -4]]).as_matrix(), q[-7:-4]
        dist, gap = [], []
        for k, link in enumerate(H["link"]):
            _, _, p_l, R_l = self._link_pose_jacobians(q, str(link))
            x = R_o.T @ (p_l + R_l @ H["r_link"][k] - o)
            dist.append(float(np.linalg.norm(x - H["c"][k])))
            gap.append(float(H["n"][k] @ x - H["d"][k]))
        _, _, p, R = self._link_pose_jacobians(q, str(H["wrist_link"]))
        E = H["E_wrist"][frame_idx]
        self.stage2_log.append({
            "frame": frame_idx, "solve_s": solve_s, "in_hold": int(H["window"][0]) <= frame_idx <= int(H["window"][1]),
            "collision_slack_m": float(val("collision", 1).sum()) if "collision" in v else 0.0,
            "foot_slack_m": float(val("foot", 1).sum()) if "foot" in v else 0.0,
            "anchor_dist_m": np.array(dist), "anchor_gap_m": np.array(gap), "force_allowed": v.get("touching", np.zeros(K, bool)),
            "wrench_slack": val("wrench", 6), "torque_slack": val("torque", nf),
            "force": val("force", (K, 3)), "wrist_err_m": float(np.linalg.norm(R_o @ E[:3, 3] + o - p)),
            "wrist_err_rad": float(np.linalg.norm(Rotation.from_matrix(R_o @ E[:3, :3] @ R.T).as_rotvec()))})

    def _held_on(self, frame_idx: int) -> bool:
        return int(self.held["window"][0]) <= frame_idx <= int(self.held["window"][1])

    def _switch_held(self, frame_idx: int) -> None:
        """Held geoms collide on window frames, the task object's geoms before the window, neither after it."""
        contype, conaff = self._geom_masks
        on, before = self._held_on(frame_idx), frame_idx < int(self.held["window"][0])
        for g in self._held_geoms:
            self.robot_model.geom_contype[g], self.robot_model.geom_conaffinity[g] = (contype[g], conaff[g]) if on else (0, 0)
        for g in self._task_object_geoms:
            self.robot_model.geom_contype[g], self.robot_model.geom_conaffinity[g] = (contype[g], conaff[g]) if before else (0, 0)

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> float | None:
        """Return z_floor if foot is locked at this frame, else None."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return None

        for i, (start, end) in enumerate(self._foot_lock_windows.get(side, ())):
            if start <= frame_idx <= end:
                return self._foot_lock_z_floors[side][i]
        return None

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        skipped = (
            self._self_collision_skip_geom_pairs
            if any(start <= frame_idx <= end for start, end in self._self_collision_skip_windows)
            else set()
        )
        for geom_a, geom_b in self._self_collision_geom_pairs:
            if (geom_a, geom_b) in skipped:
                continue
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        for _ in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
            )
            if np.isclose(cost, last_cost):
                break
            last_cost = cost
        return q_n, cost

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        # Opt-in static scene: a box obstacle uses the outward normal of its face nearest the contact (exact for touching
        # and penetrating points; MuJoCo's closest points coincide at zero distance, which zeroes the generic normal);
        # other obstacle geoms use the direction from the geom centre when the closest points coincide.
        obstacle_normal = None
        for geom, sign in ((geom2, 1.0), (geom1, -1.0)):
            if not self._geom_names[geom.id].startswith("obstacle_"):
                continue
            c, Rg = self.robot_data.geom_xpos[geom.id], self.robot_data.geom_xmat[geom.id].reshape(3, 3)
            p = 0.5 * (pos1 + pos2)
            if geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
                local = Rg.T @ (p - c)
                axis = int(np.argmin(geom.size[:3] - np.abs(local)))
                n_local = np.zeros(3)
                n_local[axis] = 1.0 if local[axis] >= 0 else -1.0
                obstacle_normal = sign * (Rg @ n_local)
            elif norm_v <= 1e-12 and np.linalg.norm(p - c) > 1e-9:
                obstacle_normal = sign * (p - c) / np.linalg.norm(p - c)
            break

        if obstacle_normal is not None:
            nhat_BA_W = obstacle_normal
        elif norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            n1, n2 = self._geom_names[g1], self._geom_names[g2]
            o1, o2 = n1.startswith("obstacle_"), n2.startswith("obstacle_")
            if o1 or o2:  # opt-in static scene (TaskConfig.scene_obstacles_xml): against robot geoms only
                other = n2 if o1 else n1
                return not (o1 and o2) and "ground" not in other and self.object_name not in other
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold and self.v2 is not None and self._mesh_plane_rows(g1, g2, fromto, threshold, Js, phis):
                continue
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _hull_vertices(self, mesh_id: int) -> np.ndarray:
        """A mesh's convex hull vertices in its geom frame (cached)."""
        cache = self.__dict__.setdefault("_hull_cache", {})
        if mesh_id not in cache:
            m = self.robot_model
            v = m.mesh_vert[m.mesh_vertadr[mesh_id] : m.mesh_vertadr[mesh_id] + m.mesh_vertnum[mesh_id]].astype(float)
            cache[mesh_id] = v[ConvexHull(v).vertices]
        return cache[mesh_id]

    def _mesh_plane_rows(self, g1: int, g2: int, fromto: np.ndarray, threshold: float, Js: dict, phis: dict) -> bool:
        """vicar_stage2: a robot mesh against a flat surface (the ground plane, or a box obstacle whose nearest face is its
        top): in place of the single closest-point row, one row per convex-hull vertex within `threshold` (a box top: the
        vertices over the face): Js[(g1, g2)] (V, nv), phis[(g1, g2)] (V,) with phi_v = n . (v - c),
        J_v = n . (J_mesh(v) - J_flat(v)); self._row_ids[(g1, g2)] the hull vertex ids. False for any other pair."""
        m, d = self.robot_model, self.robot_data
        for f, r in ((g1, g2), (g2, g1)):
            if m.geom_type[r] != mujoco.mjtGeom.mjGEOM_MESH or self._geom_names[r].startswith("obstacle_"):
                continue
            R, size = d.geom_xmat[f].reshape(3, 3), m.geom_size[f]
            if m.geom_type[f] == mujoco.mjtGeom.mjGEOM_PLANE:
                box = False
                break
            if m.geom_type[f] == mujoco.mjtGeom.mjGEOM_BOX and self._geom_names[f].startswith("obstacle_"):
                local = R.T @ (0.5 * (fromto[:3] + fromto[3:]) - d.geom_xpos[f])
                if int(np.argmin(size - np.abs(local))) == 2 and local[2] >= 0:
                    box = True
                    break
        else:
            return False
        n = R[:, 2]
        c = d.geom_xpos[f] + (size[2] * n if box else 0.0)
        V = self._hull_vertices(int(m.geom_dataid[r]))
        Vw = d.geom_xpos[r] + V @ d.geom_xmat[r].reshape(3, 3).T
        phi = (Vw - c) @ n
        ok = phi <= threshold
        if box:
            loc = (Vw - d.geom_xpos[f]) @ R
            ok &= (np.abs(loc[:, 0]) <= size[0]) & (np.abs(loc[:, 1]) <= size[1]) & (loc[:, 2] >= -size[2])
        keep = np.flatnonzero(ok)
        if keep.size == 0:
            return False
        T = self._build_transform_qdot_to_qvel_fast()
        Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        rows = np.zeros((keep.size, m.nv))
        for j, i in enumerate(keep):
            mujoco.mj_jac(m, d, Jp, Jr, Vw[i], int(m.geom_bodyid[r]))
            rows[j] = n @ Jp
            mujoco.mj_jac(m, d, Jp, Jr, Vw[i], int(m.geom_bodyid[f]))
            rows[j] -= n @ Jp
        Js[(g1, g2)] = rows @ T
        phis[(g1, g2)] = phi[keep]
        self.__dict__.setdefault("_row_ids", {})[(g1, g2)] = keep
        return True

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
