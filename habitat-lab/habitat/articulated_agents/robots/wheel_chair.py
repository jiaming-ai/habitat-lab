import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)


class WheelChair(MobileManipulator):
    cls_uuid: str = "wheel_chair"

    def _get_fetch_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(8, 14)),
            # no gripper
            gripper_joints=[],
            # arm_init_params=np.zeros((6,)),
            gripper_init_params=np.zeros((2,)),
            ee_offset=[mn.Vector3()],  # zeroed
            ee_links=[13],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]),
            wheel_joints=[4,5,6,7],
            arm_init_params=np.array(
                [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573],
                dtype=np.float32,
            ),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.11, 0, 1.269),
                    cam_look_at_pos=mn.Vector3(0, 0, 1.269),
                    attached_link_id=-1,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },
            gripper_closed_state=np.array([0.0, 0.0], dtype=np.float32),
            gripper_open_state=np.array([0.04, 0.04], dtype=np.float32),
            gripper_state_eps=0.001,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            wheel_mtr_pos_gain=0.0,
            wheel_mtr_vel_gain=1.3,
            wheel_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0, 0, 0),
            base_link_names={
                "base_link",
                "r_wheel_link",
                "l_wheel_link",
                "r_wheel_link",
                "bellows_link",
                "bellows_link2",
                "estop_link",
                "laser_link",
                "torso_fixed_link",
            },
        )

    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_fetch_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
        )
        self.back_joint_id = 6
        self.head_rot_jid = 8
        self.head_tilt_jid = 9

    def reconfigure(self) -> None:
        super().reconfigure()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    def reset(self) -> None:
        super().reset()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def update(self):
        super().update()
        # Fix the head.
        self._set_joint_pos(self.head_rot_jid, 0)
        self._set_motor_pos(self.head_rot_jid, 0)
        self._set_joint_pos(self.head_tilt_jid, np.pi / 2)
        self._set_motor_pos(self.head_tilt_jid, np.pi / 2)
        # Fix the back
        fix_back_val = 0.15
        self._set_joint_pos(self.back_joint_id, fix_back_val)
        self._set_motor_pos(self.back_joint_id, fix_back_val)


