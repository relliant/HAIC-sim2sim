"""TienKung sim2sim controller parameters and observation layout.

The HAIC TienKung task has a 20-DoF actuator set (no wrist joints).  Keeping
these values in a separate controller prevents the G1 29-DoF assumptions from
silently reordering or truncating TienKung actions.
"""

from pathlib import Path

import numpy as np
import onnxruntime

from .intermimic import PolicyOutput
from .math_utils import quat_rotate_inverse_xyzw
from .motion import MotionReference


TIENKUNG_JOINT_NAMES = [
    "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "shoulder_pitch_r_joint", "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
]

TIENKUNG_BODY_NAMES = [
    "pelvis", "hip_pitch_l_link", "hip_pitch_r_link", "hip_yaw_l_link",
    "hip_yaw_r_link", "knee_pitch_l_link", "knee_pitch_r_link",
    "shoulder_pitch_l_link", "shoulder_pitch_r_link", "ankle_roll_l_link",
    "ankle_roll_r_link", "elbow_pitch_l_link", "elbow_pitch_r_link",
    "left_hand", "right_hand",
]


class TienKungInterMimic:
    """Run an exported HAIC TienKung policy in MuJoCo."""

    num_actions = 20
    num_dof = 20

    def __init__(self, policy_path, motion_path):
        policy_path = Path(policy_path)
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"TienKung policy not found: {policy_path}. Export the HAIC "
                "TienKung policy to ONNX or pass --policy."
            )
        self.policy = onnxruntime.InferenceSession(str(policy_path))
        self.policy_inputs = self.policy.get_inputs()
        self.input_names = [item.name for item in self.policy_inputs]
        self.output_names = [item.name for item in self.policy.get_outputs()]

        self.refmotion = MotionReference(
            motion_path=motion_path,
            future_steps=[1, 2, 8, 16, 32],
            body_names=TIENKUNG_BODY_NAMES,
            joint_names=TIENKUNG_JOINT_NAMES,
        )

        self.default_pos = np.array([
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
            0.0, 0.1, 0.0, -0.3,
            0.0, -0.1, 0.0, -0.3,
        ], dtype=np.float32)
        self.action_scale = np.array([
            0.35, 0.55, 0.55, 0.35, 0.25, 0.20,
            0.35, 0.55, 0.55, 0.35, 0.25, 0.20,
            0.44, 0.44, 0.44, 0.44,
            0.44, 0.44, 0.44, 0.44,
        ], dtype=np.float32)
        self.dof_kps = np.array([
            700.0, 700.0, 500.0, 700.0, 30.0, 16.8,
            700.0, 700.0, 500.0, 700.0, 30.0, 16.8,
            60.0, 20.0, 10.0, 10.0, 60.0, 20.0, 10.0, 10.0,
        ], dtype=np.float32)
        self.dof_kds = np.array([
            10.0, 10.0, 5.0, 10.0, 2.5, 1.4,
            10.0, 10.0, 5.0, 10.0, 2.5, 1.4,
            3.0, 1.5, 1.0, 1.0, 3.0, 1.5, 1.0, 1.0,
        ], dtype=np.float32)
        self.torque_limits = np.array([
            100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
            100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ], dtype=np.float32)

        self.dof_pos_buf = np.tile(self.default_pos, (9, 1))
        self.dof_vel_buf = np.zeros((9, self.num_dof), dtype=np.float32)
        self.action_buf = np.zeros((self.num_actions, 3), dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

    def reset_history(self):
        self.dof_pos_buf[:, :] = self.default_pos
        self.dof_vel_buf[:, :] = 0.0
        self.action_buf[:, :] = 0.0
        self.action[:] = 0.0
        self.refmotion.t = 0

    def step(self, state):
        self.refmotion.update()
        policy_feed = self._build_policy_feed(self._compute_obs(state))
        outputs = self.policy.run(None, policy_feed)
        action = self._get_action(outputs)
        self.action = action.copy()
        target_dof_pos = action * self.action_scale + self.default_pos
        return PolicyOutput(
            target_dof_pos=target_dof_pos.astype(np.float32),
            dof_kps=self.dof_kps,
            dof_kds=self.dof_kds,
            tau_ff=np.zeros(self.num_dof, dtype=np.float32),
        )

    def _compute_obs(self, state):
        projected_gravity = quat_rotate_inverse_xyzw(
            state["quat"], np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        self.dof_pos_buf[1:] = self.dof_pos_buf[:-1]
        self.dof_vel_buf[1:] = self.dof_vel_buf[:-1]
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.dof_pos_buf[0] = state["qj"]
        self.dof_vel_buf[0] = state["dqj"]
        self.action_buf[:, 0] = self.action
        history = [0, 1, 2, 3, 4, 8]
        obs_policy = np.concatenate([
            state["ang_vel"], projected_gravity,
            self.dof_pos_buf[history].reshape(-1),
            self.dof_vel_buf[history].reshape(-1),
            self.action_buf.reshape(-1),
        ])
        obs_command = np.concatenate([
            self.refmotion.get_ref_body_pos_future_local(),
            self.refmotion.get_ref_joint_pos_future(),
            self.refmotion.get_ref_joint_vel_future(),
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
