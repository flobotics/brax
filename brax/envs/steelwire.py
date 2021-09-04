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

"""Trains a winch to hit with its steelwire an object and move it to targets."""

from typing import Tuple

import dataclasses
import jax
import jax.numpy as jnp
import brax
from brax.envs import env
from brax.physics import math
from brax.physics.base import take

from google.protobuf import text_format


class Steelwire(env.Env):
  """Steelwire trains an agent to contact an object.

  Steelwire observes three bodies: 'Hand', 'Object', and 'Target'.
  When Object reaches Target, the agent is rewarded.
  """

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.object_idx = self.sys.body_idx['Object_2']
    self.target_idx = self.sys.body_idx['target']
    # self.steelwire_end_idx = self.sys.body_idx['Object_0']
    self.target_radius = .02
    # self.target_distance = .5

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = dataclasses.replace(qp, pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, steps, zero = jnp.zeros(4)
    metrics = {
        'hits': zero
    }
    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    # rng = state.rng
    # qp, info = self.sys.step(state.qp, action)
    # obs = self._get_obs(qp, info)
    #
    # # vector from tip to target is last 3 entries of obs vector
    # reward_dist = -jnp.linalg.norm(obs[-3:])
    # reward = reward_dist
    #
    # steps = state.steps + self.action_repeat
    # done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    #
    # target_rel = qp.pos[self.target_idx] - qp.pos[self.object_idx]
    # target_dist = jnp.linalg.norm(target_rel)
    # target_hit = jnp.where(target_dist < self.target_radius, 1.0, 0.0)
    #
    # metrics = {
    #     'hits': target_hit
    # }
    
    rng = state.rng
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # vector from tip to target is last 3 entries of obs vector
    reward_dist = -jnp.linalg.norm(obs[-3:])
    reward_ctrl = -jnp.square(action).sum()
    reward = reward_dist + reward_ctrl

    steps = state.steps + self.action_repeat
    done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    # metrics = {
    #     'rewardDist': reward_dist,
    #     'rewardCtrl': reward_ctrl,
    # }
    metrics = {
        'hits': reward_dist
    }

    return env.State(rng, qp, info, obs, reward, done, steps, metrics)


    

  @property
  def action_size(self) -> int:
    return super().action_size + 3  # 3 extra actions for translating

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Egocentric observation of target and arm body."""

    # some pre-processing to pull joint angles and velocities
    (joint_angle,), _ = self.sys.joint_revolute.angle_vel(qp)

    # qpos:
    # x,y coord of target
    qpos = [qp.pos[self.target_idx, :2]]

    # dist to target and speed of tip
    arm_qps = take(qp, jnp.array(self.object_idx))
    tip_pos, tip_vel = math.to_world(arm_qps, jnp.array([0.11, 0., 0.]))
    tip_to_target = [tip_pos - qp.pos[self.target_idx]]
    cos_sin_angle = [jnp.cos(joint_angle), jnp.sin(joint_angle)]

    # qvel:
    # velocity of tip
    qvel = [tip_vel[:2]]

    return jnp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)

  def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns new random target locations in a random circle on xz plane."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    # dist = self.target_radius + self.target_distance * jax.random.uniform(rng1)
    dist = 0.5 + 0.8 * jax.random.uniform(rng1)
    ang = jnp.pi * 2. * jax.random.uniform(rng2)
    target_x = dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = 0.5 * jax.random.uniform(rng3)
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target


_SYSTEM_CONFIG = """
bodies {
  name: "Ground"
  colliders {
    plane {
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
bodies {
  name: "winch"
  colliders {
    position { 
        x: 0.00
        z: 0
    }
    rotation { x: 0 }
    capsule {
      radius: .15
      length: .9
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
  name: "Object_0"
  colliders {
    position { 
        x: 0.00
        z: .0
    }
    rotation { x: 90 }
    capsule {
      radius: .045
      length: .06
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
  name: "Object_1"
  colliders {
    position { 
        x: .00
        z: .0
    }
    rotation { x: 90 }
    capsule {
      radius: .045
      length: .06
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
  name: "Object_2"
  colliders {
    position { 
        x: .00
        z: .0
    }
    rotation { x: 90 }
    capsule {
      radius: .045
      length: .06
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
  name: "joint_ground"
  stiffness: 100.0
  parent: "Ground"
  child: "winch"
  parent_offset {
    z: 0.45
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
      min: -360
      max: 360
    }
  angle_limit {
      min: -360
      max: 360
    }
  angle_limit {
      min: -360
      max: 360
    }
  limit_strength: 0.0
}
joints {
  name: "joint_winch"
  stiffness: 100.0
  parent: "winch"
  child: "Object_0"
  parent_offset {
    y: 0.15
    z: 0.
  }
  child_offset {
    y: 0.0
  }
  rotation {
    y: 0.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
}
joints {
  name: "joint_0"
  stiffness: 100.0
  parent: "Object_0"
  child: "Object_1"
  parent_offset {
    y: 0.06
  }
  child_offset {
    y: -0.06
  }
  rotation {
    y: 0.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
}
joints {
  name: "joint_1"
  stiffness: 100.0
  parent: "Object_1"
  child: "Object_2"
  parent_offset {
    y: 0.06
  }
  child_offset {
    y: -0.06
  }
  rotation {
    y: 0.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
}
actuators {
  name: "actuator_ground_motor"
  joint: "joint_ground"
  strength: 25.0
  angle {
  }
}


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
