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

"""Trains 2 robotis ax12a servos to crawl forward

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
    self.target_idx = self.sys.body_idx['target']
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
    target_x = dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = .01
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target



_SYSTEM_CONFIG = """
bodies {
  name: "Ground"
  colliders {
    plane{}
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}


bodies {
  name: "body"
  colliders {
    box {
      halfsize {
          x: 0.04
          y: 0.04
          z: 0.02
      }
    }
    position { 
        x: -0.07
        z: .0
    }
  }
  colliders {
    box {
      halfsize {
          x: 0.02
          y: 0.002
          z: 0.01
      }
    }
    position { 
        x: -0.015
        y: 0.01
        z: .01
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}

bodies {
  name: "servo_0"
  colliders {
    box {
      halfsize {
          x: 0.02
          y: 0.002
          z: 0.01
      }
    }
    position { 
        x: .0
        z: .0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}



bodies {
  name: "target"
  colliders {
    position {
        x: 1.0
    }
    sphere {
      radius: 0.09
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}

joints {
  name: "joint_body"
  stiffness: 100.0
  parent: "body"
  child: "servo_0"
  parent_offset {
    x: 0.0
    z: 0.0
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }

  angle_limit {
      min: -90
      max: 90
  }


  angular_damping: 35
}

actuators {
  name: "actuator_0"
  joint: "joint_body"
  strength: 25.0
  torque {
  }
}

collide_include {
  first: "body"
  second: "Ground"
}
collide_include {
  first: "servo_0"
  second: "Ground"
}

dt: 0.02
substeps: 4

"""
