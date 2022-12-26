# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSuccess,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl import (  # noqa: F401.
    FixedHighLevelPolicy,
    HighLevelPolicy,
    NeuralHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    NoopSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy, PolicyAction
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class HierarchicalPolicy(nn.Module, Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()

        self._action_space = action_space
        self._num_envs: int = num_envs

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}

        task_spec_file = osp.join(
            full_config.habitat.task.task_spec_base_path,
            full_config.habitat.task.task_spec + ".yaml",
        )
        domain_file = full_config.habitat.task.pddl_domain_def

        self._pddl_problem = PddlProblem(
            domain_file,
            task_spec_file,
            config,
        )

        for i, (skill_id, use_skill_name) in enumerate(
            config.hierarchical_policy.use_skills.items()
        ):
            if use_skill_name == "":
                # Skip loading this skill if no name is provided
                continue
            skill_config = config.hierarchical_policy.defined_skills[
                use_skill_name
            ]

            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            skill_policy.set_pddl_problem(self._pddl_problem)
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i
            self._idx_to_name[i] = skill_id

        self._call_high_level: torch.Tensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )
        self._cur_skills: torch.Tensor = torch.full(
            (self._num_envs,), -1, dtype=torch.long
        )

        high_level_cls = eval(
            config.hierarchical_policy.high_level_policy.name
        )
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.hierarchical_policy.high_level_policy,
            self._pddl_problem,
            num_envs,
            self._name_to_idx,
            observation_space,
            action_space,
        )
        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        )

    def eval(self):
        pass

    def extract_policy_info(
        self, action_data, infos, dones
    ) -> List[Dict[str, float]]:
        ret_policy_infos = []
        for i, (info, policy_info) in enumerate(
            zip(infos, action_data.policy_info)
        ):
            cur_skill_idx = self._cur_skills[i].item()
            ret_policy_info: Dict[str, Any] = {
                "cur_skill": self._idx_to_name[cur_skill_idx],
                **policy_info,
            }

            did_skill_fail = dones[i] and not info[CompositeSuccess.cls_uuid]
            for skill_name, idx in self._name_to_idx.items():
                ret_policy_info[f"failed_skill_{skill_name}"] = (
                    did_skill_fail if idx == cur_skill_idx else 0.0
                )
            ret_policy_infos.append(ret_policy_info)

        return ret_policy_infos

    @property
    def num_recurrent_layers(self):
        return self._high_level_policy.num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._high_level_policy.parameters()

    def to(self, device):
        self._high_level_policy.to(device)
        for skill in self._skills.values():
            skill.to(device)

    def _broadcast_skill_ids(
        self,
        skill_ids: torch.Tensor,
        sel_dat: Dict[str, Any],
        should_adds: Optional[torch.Tensor] = None,
    ) -> Dict[int, Tuple[List[int], Dict[str, Any]]]:
        """
        Groups the information per skill. Specifically, this will return a map
        from the skill ID to the indices of the batch and the observations at
        these indices the skill is currently running for. This is used to batch
        observations per skill.
        """

        skill_to_batch: Dict[int, List[int]] = defaultdict(list)
        if should_adds is None:
            should_adds = [True for _ in range(len(skill_ids))]
        for i, (cur_skill, should_add) in enumerate(
            zip(skill_ids, should_adds)
        ):
            if should_add:
                cur_skill = cur_skill.item()
                skill_to_batch[cur_skill].append(i)
        grouped_skills = {}
        for k, v in skill_to_batch.items():
            grouped_skills[k] = (
                v,
                {dat_k: dat[v] for dat_k, dat in sel_dat.items()},
            )
        return grouped_skills

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        log_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._high_level_policy.apply_mask(masks)  # type: ignore[attr-defined]

        should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )

        hl_says_term = self._high_level_policy.get_termination(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )
        # Compute the actions from the current skills
        actions = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)),
            device=masks.device,
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_says_term": hl_says_term,
            },
            # Only decide on skill termination if the episode is active.
            should_adds=masks,
        )

        # Check if skills should terminate.
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            if skill_id == -1:
                # Policy has not prediced a skill yet.
                should_terminate[batch_ids] = 1.0
                continue
            (
                should_terminate[batch_ids],
                bad_should_terminate[batch_ids],
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._idx_to_name[self._cur_skills[i].item()]
                    for i in batch_ids
                ],
            )
        self._call_high_level = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = self._call_high_level | (~masks).view(-1).cpu()

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = {}
        if self._call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate,
                hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
                deterministic,
                log_info,
            )

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                should_adds=self._call_high_level,
            )

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                rnn_hidden_states[batch_ids] *= 0.0
                prev_actions[batch_ids] *= 0
            self._cur_skills = (
                (~self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills)

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
        )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            action_data = self._skills[skill_id].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
            )

            # LL skills are not allowed to terminate the overall episode.
            actions[batch_ids] = action_data.actions
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states
        actions[:, self._stop_action_idx] = 0.0

        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0

        ac_info = {
            "values": None,
            "action_log_probs": None,
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        ac_info.update(hl_info)

        return PolicyAction(
            take_actions=actions, policy_info=log_info, **ac_info
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.habitat_baselines.rl.policy,
            config,
            observation_space,
            orig_action_space,
            config.habitat_baselines.num_environments,
        )
