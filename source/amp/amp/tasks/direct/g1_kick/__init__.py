# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""宇树G1踢球任务环境注册"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1-Kick-Direct-v0",
    entry_point=f"{__name__}.g1_kick_env:G1KickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_kick_env_cfg:G1KickEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

