"""RxP2 (rx_p2_27dof_0721) sim2sim controller parameters and observation layout.

RxP2's MJCF (compiled from the URDF, see assets/robots/rx_p2/rx_p2_box.xml)
lays out joints as waist -> right arm -> left arm -> right leg -> left leg.
Isaac Lab's PhysX articulation parses the *same* URDF into a different DOF
order (interleaved left/right, grouped by joint depth) — this is the order
the HAIC policy was actually trained on. The two orders are captured below as
RX_P2_JOINT_NAMES_MJC / RX_P2_JOINT_NAMES_ISAAC; the isaac order and the
default_pos/stiffness/damping values were read verbatim off
`asset_meta.json` dumped by an actual RxP2TrackMyBox training run
(HAIC/outputs/2026-08-27/15-18-48-RxP2TrackMyBox-ppo_haic/asset_meta.json),
not re-derived by hand. All policy I/O (actions, obs history, PD targets
before mapping onto the sim) uses the isaac order; MuJoCo qpos/qvel/ctrl use
the mjc order. `_MJC_TO_ISAAC` / `_ISAAC_TO_MJC` convert between them.
"""

from pathlib import Path

import numpy as np
import onnxruntime

from .intermimic import PolicyOutput
from .math_utils import quat_rotate_inverse_xyzw
from .motion import MotionReference


# Isaac Lab / policy training order (verbatim from a real RxP2 training run's
# asset_meta.json `joint_names_isaac`). This is the order the ONNX policy's
# actions and proprioceptive history are indexed by.
RX_P2_JOINT_NAMES_ISAAC = [
    "l_hip_pitch_joint", "r_hip_pitch_joint", "waist_yaw_joint",
    "l_hip_roll_joint", "r_hip_roll_joint", "waist_roll_joint",
    "l_hip_yaw_joint", "r_hip_yaw_joint", "waist_pitch_joint",
    "l_knee_joint", "r_knee_joint",
    "l_shoulder_pitch_joint", "r_shoulder_pitch_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
    "l_shoulder_roll_joint", "r_shoulder_roll_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
    "l_shoulder_yaw_joint", "r_shoulder_yaw_joint",
    "l_elbow_joint", "r_elbow_joint",
    "l_wrist_roll_joint", "r_wrist_roll_joint",
    "l_wrist_pitch_joint", "r_wrist_pitch_joint",
]

# MuJoCo actuator/qpos order used by assets/robots/rx_p2/rx_p2_box.xml
# (same order as the source URDF: waist, r_arm, l_arm, r_leg, l_leg).
RX_P2_JOINT_NAMES_MJC = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_shoulder_yaw_joint",
    "r_elbow_joint", "r_wrist_roll_joint", "r_wrist_pitch_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_shoulder_yaw_joint",
    "l_elbow_joint", "l_wrist_roll_joint", "l_wrist_pitch_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint",
    "r_knee_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint",
    "l_knee_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
]

# qj_isaac = qj_mjc[_MJC_TO_ISAAC]; target_mjc = target_isaac[_ISAAC_TO_MJC].
_MJC_TO_ISAAC = np.array(
    [RX_P2_JOINT_NAMES_MJC.index(name) for name in RX_P2_JOINT_NAMES_ISAAC]
)
_ISAAC_TO_MJC = np.array(
    [RX_P2_JOINT_NAMES_ISAAC.index(name) for name in RX_P2_JOINT_NAMES_MJC]
)

# Tracking-keypoint body order: cfg/task/base/hdmi-rx_p2-base.yaml's
# command.tracking_keypoint_names regexes, resolved (preserve_order=False,
# i.e. filtered against the asset's own body order) against the isaac
# body_names_isaac from the same training run's asset_meta.json.
RX_P2_BODY_NAMES = [
    "base_link",
    "l_hip_pitch_link", "r_hip_pitch_link",
    "l_hip_yaw_link", "r_hip_yaw_link",
    "waist_pitch_link",
    "l_knee_link", "r_knee_link",
    "l_shoulder_pitch_link", "r_shoulder_pitch_link",
    "l_ankle_roll_link", "r_ankle_roll_link",
    "l_elbow_link", "r_elbow_link",
    "l_wrist_pitch_link", "r_wrist_pitch_link",
]

# Hard joint ranges (isaac order), transcribed verbatim from the <joint range=...>
# entries in assets/robots/rx_p2/rx_p2_box.xml (itself compiled from the same
# rx_p2_27dof_0721 URDF used to build the Isaac Lab USD asset).
RX_P2_JOINT_LIMITS_ISAAC = np.array([
    [-2.094, 0.524], [-2.094, 0.524], [-1.57, 1.57],
    [-0.785, 0.785], [-0.785, 0.785], [-0.436, 0.436],
    [-1.047, 1.047], [-1.047, 1.047], [-0.262, 0.524],
    [0.0, 1.798], [0.0, 1.798],
    [-3.229, 1.92], [-3.229, 1.92],
    [-0.61, 0.61], [-0.61, 0.61],
    [-0.262, 1.833], [-1.833, 0.262],
    [-0.52, 0.52], [-0.52, 0.52],
    [-2.007, 3.054], [-3.054, 2.007],
    [-0.524, 2.094], [-0.524, 2.094],
    [-3.054, 2.007], [-2.007, 3.054],
    [-1.623, 1.623], [-1.623, 1.623],
], dtype=np.float32)

# RX_P2_CFG.soft_joint_pos_limit_factor. Isaac Lab's JointPosition action
# (action.joint_target_clamp=true) clamps every commanded target to this
# factor of the hard range around its midpoint before it ever reaches the PD
# controller — see Articulation._process_cfg's soft_joint_pos_limits formula.
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
_joint_mid = RX_P2_JOINT_LIMITS_ISAAC.mean(axis=1)
_joint_half_range = (RX_P2_JOINT_LIMITS_ISAAC[:, 1] - RX_P2_JOINT_LIMITS_ISAAC[:, 0]) / 2.0
SOFT_JOINT_POS_LIMITS_ISAAC = np.stack([
    _joint_mid - _joint_half_range * SOFT_JOINT_POS_LIMIT_FACTOR,
    _joint_mid + _joint_half_range * SOFT_JOINT_POS_LIMIT_FACTOR,
], axis=1).astype(np.float32)

# action.joint_target_rate_limit (isaac order, rad/s) from
# cfg/task/base/hdmi-rx_p2-base.yaml. JointPosition.__call__ applies this as
# a per-physics-substep bound on how far the *applied* target may move away
# from the previous substep's applied target — a much tighter constraint than
# the once-per-control-step target jump this sim2sim adapter used before.
RX_P2_JOINT_TARGET_RATE_LIMIT_ISAAC = np.array([
    10.0, 10.0, 8.0,
    10.0, 10.0, 5.0,
    10.0, 10.0, 5.0,
    10.0, 10.0,
    8.0, 8.0,
    8.0, 8.0,
    6.0, 6.0,
    5.0, 5.0,
    8.0, 8.0,
    8.0, 8.0,
    6.0, 6.0,
    6.0, 6.0,
], dtype=np.float32)


class RxP2InterMimic:
    """Run an exported HAIC RxP2 policy in MuJoCo."""

    num_actions = 27
    num_dof = 27

    def __init__(self, policy_path, motion_path):
        policy_path = Path(policy_path)
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"RxP2 policy not found: {policy_path}. Export the HAIC "
                "RxP2 policy to ONNX or pass --policy."
            )
        self.policy = onnxruntime.InferenceSession(str(policy_path))
        self.policy_inputs = self.policy.get_inputs()
        self.input_names = [item.name for item in self.policy_inputs]
        self.output_names = [item.name for item in self.policy.get_outputs()]

        self.refmotion = MotionReference(
            motion_path=motion_path,
            future_steps=[1, 2, 8, 16, 32],
            body_names=RX_P2_BODY_NAMES,
            joint_names=RX_P2_JOINT_NAMES_ISAAC,
            root_body_name="base_link",
        )

        # Default pose (isaac order), verbatim from asset_meta.json
        # `default_joint_pos` of a real RxP2TrackMyBox training run.
        self.default_pos_isaac = np.array([
            -0.3, -0.3, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.6, 0.6,
            0.0, 0.0,
            -0.3, -0.3,
            0.1, -0.1,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ], dtype=np.float32)
        # Action scaling (isaac order) from
        # cfg/task/base/hdmi-rx_p2-base.yaml action.action_scaling.
        self.action_scale_isaac = np.array([
            0.55, 0.55, 0.55,
            0.35, 0.35, 0.44,
            0.55, 0.55, 0.44,
            0.35, 0.35,
            0.44, 0.44,
            0.25, 0.25,
            0.44, 0.44,
            0.20, 0.20,
            0.44, 0.44,
            0.44, 0.44,
            0.20, 0.20,
            0.15, 0.15,
        ], dtype=np.float32)
        # PD stiffness (isaac order), verbatim from asset_meta.json `stiffness`.
        dof_kps_isaac = np.array([
            100.0, 100.0, 100.0,
            100.0, 100.0, 100.0,
            50.0, 50.0, 100.0,
            100.0, 100.0,
            100.0, 100.0,
            20.0, 20.0,
            100.0, 100.0,
            20.0, 20.0,
            60.0, 60.0,
            80.0, 80.0,
            50.0, 50.0,
            50.0, 50.0,
        ], dtype=np.float32)
        # PD damping (isaac order), verbatim from asset_meta.json `damping`.
        dof_kds_isaac = np.array([
            2.0, 2.0, 3.0,
            2.0, 2.0, 2.5,
            1.0, 1.0, 2.5,
            2.0, 2.0,
            3.0, 3.0,
            1.5, 1.5,
            3.0, 3.0,
            1.5, 1.5,
            2.0, 2.0,
            2.5, 2.5,
            1.5, 1.5,
            1.5, 1.5,
        ], dtype=np.float32)
        # effort_limit_sim (isaac order) from RX_P2_CFG actuator groups.
        torque_limits_isaac = np.array([
            40.6, 40.6, 40.6,
            40.6, 40.6, 36.8,
            18.4, 18.4, 36.8,
            40.6, 40.6,
            40.6, 40.6,
            36.8, 36.8,
            18.4, 18.4,
            36.8, 36.8,
            18.4, 18.4,
            18.4, 18.4,
            18.4, 18.4,
            18.4, 18.4,
        ], dtype=np.float32)

        # PD control, torque clipping, and target shaping happen against
        # mjc-ordered qpos/qvel/ctrl in scripts/deploy_haic.py, so these stay
        # permuted into mjc order once at init.
        self.default_pos = self.default_pos_isaac[_ISAAC_TO_MJC]
        self.dof_kps = dof_kps_isaac[_ISAAC_TO_MJC]
        self.dof_kds = dof_kds_isaac[_ISAAC_TO_MJC]
        self.torque_limits = torque_limits_isaac[_ISAAC_TO_MJC]
        self.soft_joint_pos_limits = SOFT_JOINT_POS_LIMITS_ISAAC[_ISAAC_TO_MJC]
        self.joint_target_rate_limit = RX_P2_JOINT_TARGET_RATE_LIMIT_ISAAC[_ISAAC_TO_MJC]
        self.applied_joint_target = self.default_pos.copy()

        self.dof_pos_buf = np.tile(self.default_pos_isaac, (9, 1))
        self.action_buf = np.zeros((self.num_actions, 3), dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

    def reset_history(self):
        self.dof_pos_buf[:, :] = self.default_pos_isaac
        self.action_buf[:, :] = 0.0
        self.action[:] = 0.0
        self.refmotion.t = 0
        # Mirrors JointPosition.reset(): seed the applied target with the
        # reference motion's own joint_pos at the reset frame (clamped to the
        # soft limits), not the neutral default pose — otherwise the first
        # control step's rate limit ramps from the wrong starting point.
        seed_isaac = np.clip(
            self.refmotion.joint_pos[0],
            SOFT_JOINT_POS_LIMITS_ISAAC[:, 0],
            SOFT_JOINT_POS_LIMITS_ISAAC[:, 1],
        )
        self.applied_joint_target = seed_isaac[_ISAAC_TO_MJC].astype(np.float32)

    def step(self, state):
        self.refmotion.update()
        policy_feed = self._build_policy_feed(self._compute_obs(state))
        outputs = self.policy.run(None, policy_feed)
        action = self._get_action(outputs)
        self.action = action.copy()
        target_dof_pos_isaac = action * self.action_scale_isaac + self.default_pos_isaac
        target_dof_pos_mjc = target_dof_pos_isaac[_ISAAC_TO_MJC]
        return PolicyOutput(
            target_dof_pos=target_dof_pos_mjc.astype(np.float32),
            dof_kps=self.dof_kps,
            dof_kds=self.dof_kds,
            tau_ff=np.zeros(self.num_dof, dtype=np.float32),
        )

    def shape_target(self, raw_target_mjc, sim_dt):
        """Reproduce JointPosition.__call__'s per-substep clamp + rate limit
        (active_adaptation/envs/mdp/action.py), called once per MuJoCo
        physics step instead of once per Isaac physics step — the rate limit
        is linear in time so an equal total ramp results as long as
        rate_limit * sim_dt is applied at every substep, regardless of substep
        count. `raw_target_mjc` is the fresh policy output, held fixed for the
        whole control period; this method mutates and returns the actually
        applied, ramped target for the current substep.
        """
        lo = self.soft_joint_pos_limits[:, 0]
        hi = self.soft_joint_pos_limits[:, 1]
        clamped = np.clip(raw_target_mjc, lo, hi)
        max_delta = self.joint_target_rate_limit * sim_dt
        ramped = np.clip(
            clamped, self.applied_joint_target - max_delta, self.applied_joint_target + max_delta
        )
        ramped = np.clip(ramped, lo, hi)
        self.applied_joint_target = ramped.astype(np.float32)
        return self.applied_joint_target

    def _compute_obs(self, state):
        # observation.policy / observation.command in
        # cfg/task/base/hdmi-base.yaml (rx_p2 does not override either)
        # carry joint_pos_history + prev_actions and
        # ref_body_pos_future_local + ref_joint_pos_future + ref_motion_phase
        # only — no joint-velocity terms. Confirmed against a real export's
        # in_shapes: command=376, policy=249, object_geo_=384
        # (HAIC/scripts/exports/RxP2TrackMyBox/policy-m8dufod2-final.json).
        projected_gravity = quat_rotate_inverse_xyzw(
            state["quat"], np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        # state["qj"] comes from mujoco qpos, i.e. mjc order.
        qj_isaac = state["qj"][_MJC_TO_ISAAC]
        self.dof_pos_buf[1:] = self.dof_pos_buf[:-1]
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.dof_pos_buf[0] = qj_isaac
        self.action_buf[:, 0] = self.action
        history = [0, 1, 2, 3, 4, 8]
        obs_policy = np.concatenate([
            state["ang_vel"], projected_gravity,
            self.dof_pos_buf[history].reshape(-1),
            self.action_buf.reshape(-1),
        ])
        obs_command = np.concatenate([
            self.refmotion.get_ref_body_pos_future_local(),
            self.refmotion.get_ref_joint_pos_future(),
            self.refmotion.get_ref_motion_phase(),
        ])
        return {
            "command": obs_command[None].astype(np.float32),
            "policy": obs_policy[None].astype(np.float32),
            "object_geo": self.refmotion.get_object_geo()[None].astype(np.float32),
        }

    def _get_action(self, outputs):
        for name, output in zip(self.output_names, outputs):
            array = np.asarray(output)
            if array.ndim == 2 and array.shape[-1] == self.num_actions:
                return array[0].astype(np.float32)
        raise RuntimeError(
            f"Could not find a {self.num_actions}-DoF action output in "
            f"ONNX outputs {[(n, np.asarray(v).shape) for n, v in zip(self.output_names, outputs)]}"
        )

    def _build_policy_feed(self, observations):
        if len(self.policy_inputs) == 1 and self.policy_inputs[0].name not in observations:
            value = np.concatenate(
                [observations["command"], observations["policy"], observations["object_geo"]],
                axis=1,
            )
            return {self.policy_inputs[0].name: self._match_input_shape(value, self.policy_inputs[0])}
        feed = {}
        for input_info in self.policy_inputs:
            name = input_info.name
            if name not in observations:
                if name.endswith("_") and name[:-1] in observations:
                    name = name[:-1]
                else:
                    raise KeyError(
                        f"Policy expects input '{input_info.name}', "
                        f"available inputs: {sorted(observations)}"
                    )
            feed[input_info.name] = self._match_input_shape(observations[name], input_info)
        return feed

    @staticmethod
    def _match_input_shape(value, input_info):
        value = np.asarray(value, dtype=np.float32)
        shape = input_info.shape
        if len(shape) < 2 or not isinstance(shape[1], int) or value.shape[1] == shape[1]:
            return value
        if value.shape[1] > shape[1]:
            return value[:, :shape[1]]
        padded = np.zeros((value.shape[0], shape[1]), dtype=np.float32)
        padded[:, :value.shape[1]] = value
        return padded
