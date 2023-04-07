#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import argparse
from typing import Any

import magnum as mn
import numpy as np
from controllers import ControllerHelper
from magnum.platform.glfw import Application

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    OracleNavActionConfig,
    PddlApplyActionConfig,
)
from habitat.gui.gui_application import GuiAppDriver, GuiApplication
from habitat.gui.gui_input import GuiInput
from habitat.gui.replay_gui_app_renderer import ReplayGuiAppRenderer

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"

DEFAULT_CFG = "benchmark/rearrange/rearrange_easy_human_and_fetch.yaml"


class SandboxDriver(GuiAppDriver):
    def __init__(self, args, config, gui_input):
        with habitat.config.read_write(config):
            # needed so we can provide keyframes to GuiApplication
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
        self.env = habitat.Env(config=config)
        self.obs = self.env.reset()

        self.ctrl_helper = ControllerHelper(self.env, args, gui_input)

        self.gui_humanoid_ctrl = self.ctrl_helper.get_gui_humanoid_controller()

        self.ctrl_helper.on_environment_reset()

        self.cam_zoom_dist = 1.0
        self.gui_input = gui_input

        self._debug_line_render = None

    def set_debug_line_render(self, debug_line_render):
        self._debug_line_render = debug_line_render
        self._debug_line_render.set_line_width(3)

    # trying to get around mypy complaints about missing sim attributes
    def get_sim(self) -> Any:
        return self.env.task._sim

    def visualize_task(self):
        sim = self.get_sim()
        idxs, goal_pos = sim.get_targets()
        scene_pos = sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        for i in range(len(idxs)):
            radius = 0.25
            goal_color = mn.Color3(0, 153 / 255, 51 / 255)  # green
            target_color = mn.Color4(1, 1, 1, 0.1)  # white, almost transparent
            self._debug_line_render.draw_circle(
                goal_pos[i], radius, goal_color
            )
            self._debug_line_render.draw_transformed_line(
                target_pos[i], goal_pos[i], target_color, goal_color
            )
            # self._debug_line_render.draw_circle(target_pos[i], radius, goal_color)

    def viz_and_get_grasp_drop_hints(self):
        object_color = mn.Color3(255 / 255, 255 / 255, 0)
        object_highlight_radius = 0.1
        object_drop_height = 0.15

        ray = self.gui_input.mouse_ray

        if not ray or ray.direction.y >= 0:
            return None, None

        # hack move ray below ceiling (todo: base this on humanoid agent base y, so that it works in multi-floor homes)
        raycast_start_y = 2.0
        if ray.origin.y < raycast_start_y:
            return None, None

        dist_to_raycast_start_y = (
            ray.origin.y - raycast_start_y
        ) / -ray.direction.y
        assert dist_to_raycast_start_y >= 0
        adjusted_origin = ray.origin + ray.direction * dist_to_raycast_start_y
        ray.origin = adjusted_origin

        # reference code for casting a ray into the scene
        raycast_results = self.get_sim().cast_ray(ray=ray)
        if not raycast_results.has_hits():
            return None, None

        hit_info = raycast_results.hits[0]
        # self._debug_line_render.draw_circle(hit_info.point, 0.03, mn.Color3(1, 0, 0))

        if self.gui_humanoid_ctrl.is_grasped:
            self._debug_line_render.draw_circle(
                hit_info.point, object_highlight_radius, object_color
            )
            if self.gui_input.get_mouse_button_down(GuiInput.MouseNS.LEFT):
                drop_pos = hit_info.point + mn.Vector3(
                    0, object_drop_height, 0
                )
                return None, drop_pos
        else:
            # Currently, it's too hard to select objects that are very small
            # on-screen. Todo: use hit_info.point and search for nearest rigid object within
            # X cm.
            if hit_info.object_id != -1:
                sim = self.get_sim()
                rigid_obj_mgr = sim.get_rigid_object_manager()
                is_rigid_obj = rigid_obj_mgr.get_library_has_id(
                    hit_info.object_id
                )
                if is_rigid_obj:
                    rigid_obj = (
                        sim.get_rigid_object_manager().get_object_by_id(
                            hit_info.object_id
                        )
                    )
                    assert rigid_obj
                    if (
                        rigid_obj.motion_type
                        == habitat_sim.physics.MotionType.DYNAMIC
                    ):
                        self._debug_line_render.draw_circle(
                            rigid_obj.translation,
                            object_highlight_radius,
                            object_color,
                        )
                        if self.gui_input.get_mouse_button_down(
                            GuiInput.MouseNS.LEFT
                        ):
                            grasp_object_id = hit_info.object_id
                            return grasp_object_id, None

        return None, None

    def viz_and_get_humanoid_walk_dir(self):
        path_color = mn.Color3(0, 153 / 255, 255 / 255)
        path_endpoint_radius = 0.12

        ray = self.gui_input.mouse_ray

        floor_y = 0.15  # hardcoded to ReplicaCAD

        if not ray or ray.direction.y >= 0 or ray.origin.y <= floor_y:
            return None

        dist_to_floor_y = (ray.origin.y - floor_y) / -ray.direction.y
        target_on_floor = ray.origin + ray.direction * dist_to_floor_y

        agent_idx = 0
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        robot_root = art_obj.transformation

        pathfinder = self.get_sim().pathfinder
        snapped_pos = pathfinder.snap_point(target_on_floor)
        snapped_start_pos = robot_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        if not found_path:
            return None

        path_points = []
        for path_i in range(0, len(path.points)):
            adjusted_point = mn.Vector3(path.points[path_i])
            # first point in path is at wrong height
            if path_i == 0:
                adjusted_point.y = mn.Vector3(path.points[path_i + 1]).y
            path_points.append(adjusted_point)

        self._debug_line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )

        if (
            self.gui_input.get_mouse_button(GuiInput.MouseNS.RIGHT)
            or self.gui_input.get_key(GuiInput.KeyNS.SPACE)
        ) and len(path.points) >= 2:
            walk_dir = mn.Vector3(path.points[1]) - mn.Vector3(path.points[0])
            return walk_dir

        return None

    def viz_and_get_humanoid_hints(self):
        grasp_object_id, drop_pos = self.viz_and_get_grasp_drop_hints()
        walk_dir = self.viz_and_get_humanoid_walk_dir()

        return walk_dir, grasp_object_id, drop_pos

    def sim_update(self, dt):
        # todo: pipe end_play somewhere

        (
            walk_dir,
            grasp_object_id,
            drop_pos,
        ) = self.viz_and_get_humanoid_hints()
        self.gui_humanoid_ctrl.set_act_hints(
            walk_dir, grasp_object_id, drop_pos
        )

        action, end_play, reset_ep = self.ctrl_helper.update(self.obs)

        self.obs = self.env.step(action)

        if reset_ep:
            self.obs = self.env.reset()
            self.ctrl_helper.on_environment_reset()

        self.visualize_task()

        post_sim_update_dict = {}

        if self.gui_input.mouse_scroll_offset != 0:
            zoom_sensitivity = 0.07
            if self.gui_input.mouse_scroll_offset < 0:
                self.cam_zoom_dist *= (
                    1.0
                    + -self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            else:
                self.cam_zoom_dist /= (
                    1.0 + self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            max_zoom_dist = 50.0
            min_zoom_dist = 0.1
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist, min_zoom_dist, max_zoom_dist
            )

        agent_idx = 0
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        robot_root = art_obj.transformation
        lookat = robot_root.translation + mn.Vector3(0, 1, 0)
        cam_transform = mn.Matrix4.look_at(
            lookat + mn.Vector3(0.5, 1, 0.5).normalized() * self.cam_zoom_dist,
            lookat,
            mn.Vector3(0, 1, 0),
        )
        post_sim_update_dict["cam_transform"] = cam_transform

        post_sim_update_dict[
            "keyframes"
        ] = (
            self.get_sim().gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        )

        def flip_vertical(obs):
            converted_obs = np.empty_like(obs)
            for row in range(obs.shape[0]):
                converted_obs[row, :] = obs[obs.shape[0] - row - 1, :]
            return converted_obs

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # reference code for visualizing a camera sensor in the app GUI
        # post_sim_update_dict["debug_images"] = [
        #     flip_vertical(depth_to_rgb(self.obs["agent_1_robot_head_depth"]))
        # ]

        return post_sim_update_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sps",
        type=int,
        default=30,
        help="Target rate to step the environment (steps per second); actual SPS may be lower depending on your hardware",
    )
    parser.add_argument(
        "--width",
        default=1280,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=720,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--humanoid-user-agent",
        action="store_true",
        default=False,
        help="Set to true if the user-controlled agent is a humanoid. Set to false if the user-controlled agent is a robot.",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--use-batch-renderer",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )

    args = parser.parse_args()

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        task_config.actions[
            "agent_1_oracle_nav_action"
        ] = OracleNavActionConfig(agent_index=1)

        if True:
            # Code below is ported from interactive_play.py. I'm not sure what it is for.
            agent_config = get_agent_config(sim_config=sim_config)
            if "composite_success" in task_config.measurements:
                task_config.measurements.composite_success.must_call_stop = (
                    False
                )
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    driver = SandboxDriver(args, config, gui_app_wrapper.get_sim_input())
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()
    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size.x, framebuffer_size.y, args.use_batch_renderer
    )
    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    # sloppy: provide replay app renderer's debug_line_render to our driver
    driver.set_debug_line_render(
        app_renderer._replay_renderer.debug_line_render(0)
    )

    gui_app_wrapper.exec()