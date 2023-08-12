#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from hablab_utils import get_agent_art_obj_transform

import habitat_sim
import math
import numpy as np

class GuiThrowHelper:
    def __init__(self, gui_service, agent_idx):
        self._sandbox_service = gui_service
        self._agent_idx = agent_idx
        self._largest_island_idx = None

    def _get_sim(self):
        return self._sandbox_service.sim


    def compute_velocity_throw(self, start_point, end_point, gravity=-9.8):
        displacement = end_point - start_point
        dx, dy, dz = displacement
        time_of_flight = math.sqrt((2 * dy) / gravity)
        horizontal_distance = math.sqrt(dx**2 + dz**2)
        vx = dx / time_of_flight
        vz = dz / time_of_flight

        vy = (end_point[1] - start_point[1] - 0.5 * gravity * time_of_flight**2) / time_of_flight
        vel = mn.Vector3(vx, vy, vz)

        times = np.linspace(0, time_of_flight, 10)        
        x_pos = start_point[0] + vel[0] * times     
        z_pos = start_point[2] + vel[2] * times
        y_pos = start_point[1] + vel[1] * times + 0.5 * gravity * (times ** 2)
        path_points = []
        for i in range(10):
            path_points.append(mn.Vector3(x_pos[i], y_pos[i], z_pos[i]))
        return vel, path_points
    
    def viz_and_get_humanoid_throw(self):
        path_color = mn.Color3(153 / 255, 0, 255 / 255)
        path_endpoint_radius = 0.12

        ray = self._sandbox_service.gui_input.mouse_ray

        floor_y = 0.15  # hardcoded to ReplicaCAD

        if not ray or ray.direction.y >= 0 or ray.origin.y <= floor_y:
            return None

        dist_to_floor_y = (ray.origin.y - floor_y) / -ray.direction.y
        target_on_floor = ray.origin + ray.direction * dist_to_floor_y

        art_obj = (
            self._get_sim().agents_mgr[self._agent_idx].articulated_agent.sim_obj
        )
        robot_root = art_obj.transformation.translation
        path_points = [robot_root, target_on_floor]
        vel_vector, path_points = self.compute_velocity_throw(robot_root, target_on_floor)
        self._sandbox_service.line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )
        self._sandbox_service.line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )


        return vel_vector  

    @staticmethod
    def _evaluate_cubic_bezier(ctrl_pts, t):
        assert len(ctrl_pts) == 4
        weights = (
            pow(1 - t, 3),
            3 * t * pow(1 - t, 2),
            3 * pow(t, 2) * (1 - t),
            pow(t, 3),
        )

        result = weights[0] * ctrl_pts[0]
        for i in range(1, 4):
            result += weights[i] * ctrl_pts[i]

        return result
