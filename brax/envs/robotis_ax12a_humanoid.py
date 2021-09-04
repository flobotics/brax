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


class Ax12aHumanoid(env.Env):
  """Trains a reacher arm to touch a sequence of random targets."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.target_idx = self.sys.body_idx['target']
    self.arm_idx = self.sys.body_idx['servo_bracket_fp04_f2']

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    #rng, target = self._random_target(rng)
    #pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    #qp = dataclasses.replace(qp, pos=pos)
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

    # vector from tip to target is last 3 entries of obs vector
    #reward_dist = -jnp.linalg.norm(obs[-3:])
    #reward_ctrl = -jnp.square(action).sum()
    #reward = reward_dist + reward_ctrl
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
    
    #print(f"info >{info}<")

    # some pre-processing to pull joint angles and velocities
    (joint_angle,), _ = self.sys.joint_revolute.angle_vel(qp)
    #print(f"joint_angle >{joint_angle}<")
    #print(f"vel >{_}<")

    #print(f"self.sys.joint_revolute >{self.sys.joint_revolute[1]}<")

    # qpos:
    # x,y coord of target
    qpos = [qp.pos[self.target_idx, :2]]
    

    #debug_outputs.print_bodies_and_position(self.sys.body_idx, qp)
    #debug_outputs.print_joints(self.sys.joint_revolute)

    # dist to target and speed of tip
    arm_qps = take(qp, jnp.array(self.arm_idx))
    #print(f"arm_qps >{arm_qps}<")
    
    tip_pos, tip_vel = math.to_world(arm_qps, jnp.array([0.11, 0., 0.]))
    #print(f"tip_pos >{tip_pos}<")
    #print(f"tip_vel >{tip_vel}<")
    
    tip_to_target = [tip_pos - qp.pos[self.target_idx]]
    #print(f"tip_to_target >{tip_to_target}<")
    
    cos_sin_angle = [jnp.cos(joint_angle), jnp.sin(joint_angle)]
    #print(f"cos_sin_angle >{cos_sin_angle}<")

    # qvel:
    # velocity of tip
    qvel = [tip_vel[:2]]
    #print(f"qvel >{qvel}<")

    #return jnp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)
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
  name: "ground"
  colliders {
    heightMap {
        size: 10
        data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      }
  }
  mass: 1.0
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  frozen {
    all: true
  }
}

bodies {
  name: "servo_bracket_fp04_f2"
  colliders {
    box {
      halfsize {
          x: 0.005
          y: 0.01
          z: 0.015
        }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.035604715
}

bodies {
  name: "servo_0"
  colliders {
    box {
      halfsize {
          x: 0.005
          y: 0.01
          z: 0.015
        }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.035604715
}

bodies {
  name: "target"
  colliders {
    position {
        x: 0.05
        y: 0.05
        z: 0.05
    }
    sphere {
      radius: 0.009
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
  name: "joint0"
  stiffness: 100.0
  parent: "servo_bracket_fp04_f2"
  child: "servo_0"
  parent_offset {
    x: 0.05
  }
  child_offset {
  }
  rotation {
      y: -90
  }
  angle_limit {
      min: -60
      max: 60
    }
  limit_strength: 0.0
  spring_damping: 3.0
}

actuators {
  name: "joint0"
  joint: "joint0"
  strength: 25.0
  torque {
  }
}

collide_include {
  first: "ground"
  second: "servo_bracket_fp04_f2"
}
collide_include {
  first: "ground"
  second: "servo_0"
}


friction: 0.6
gravity {
  z: -9.81
}
baumgarte_erp: 0.1
dt: 0.02
substeps: 4
frozen {
  position {
    z: 1.0
  }
  rotation {
    x: 1.0
    y: 1.0
  }
}
"""
