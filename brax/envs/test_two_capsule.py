# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains 8 robotis ax12a servos to walk like a human

Based on the OpenAI Gym MuJoCo Reacher environment.
"""

from typing import Tuple

import dataclasses
import jax
import jax.numpy as jnp
import brax
from brax.envs import env
from brax.physics import math
from brax.physics.base import take
from brax.tests import debug_outputs

from google.protobuf import text_format


class SkeletonEnv(env.Env):
  """Trains a reacher arm to touch a sequence of random targets."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.target_idx = self.sys.body_idx['Target']
    #self.target_idx = self.sys.body_idx['servo_bracket_fp04_f2']

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, steps, zero = jnp.zeros(4)
    metrics = {
        'rewardDist': zero,
        'rewardCtrl': zero,
    }
    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    rng = state.rng
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    reward = 1.0

    steps = state.steps + self.action_repeat
    done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    metrics = {
        'rewardDist': 1.0,
        'rewardCtrl': 1.0,
    }

    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Egocentric observation of target and arm body."""
    
    return qp.pos[self.target_idx]

  def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a target location in a random circle slightly above xy plane."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    dist = .2 * jax.random.uniform(rng1)
    ang = jnp.pi * 2. * jax.random.uniform(rng2)
    target_x = 1.0 + dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = 3.01
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target



_SYSTEM_CONFIG = """

bodies {
  name: "Shoulders"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.75 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}

bodies {
  name: "Front Right Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}

bodies {
  name: "Ground"
  colliders { plane {} }
  frozen { all: true }
}
bodies {
  name: "Target"
  colliders { sphere { radius: 0.02 }}
  frozen { all: true }
}

joints {
  name: "Shoulders_Front Right Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.875 }
  child_offset { z: 0.375 }
  parent: "Shoulders"
  child: "Front Right Upper"
  stiffness: 5000.0
  angular_damping: 35
}


actuators {
  name: "Shoulders_Front Right Upper"
  torque {}
  joint: "Shoulders_Front Right Upper"
  strength: 300.0
}

friction: 0.6
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "Front Right Upper"
  second: "Ground"
}
collide_include {
  first: "Shoulders"
  second: "Ground"
}


dt: 0.02
substeps: 4
"""