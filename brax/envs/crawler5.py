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

import brax
from brax.envs import env
from brax.physics import math
from brax.physics.base import take
import jax
import jax.numpy as jnp


class SkeletonEnv(env.Env):
  """Trains a reacher arm to touch a sequence of random targets."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.servo_idx = self.sys.body_idx['servo_1']
    self.target_idx = self.sys.body_idx['target']

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    _, target = self._random_target(rng)
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = qp.replace(pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jnp.zeros(3)
    metrics = {
        'rewardDist': zero,
        'rewardCtrl': zero,
    }
    
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    # rng = state.rng
    # qp, info = self.sys.step(state.qp, action)
    # obs = self._get_obs(qp, info)
    #
    # qpos1 = -jnp.linalg.norm(obs[-1:])
    # qpos2 = -jnp.linalg.norm(obs[-2:-1])
    # reward = qpos1 - qpos2
    #
    # steps = state.steps + self.action_repeat
    # done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    # metrics = {
    #     'rewardDist': 1.0,
    #     'rewardCtrl': 1.0,
    # }
    #
    # return env.State(rng, qp, info, obs, reward, done, steps, metrics)
    
    # rng = state.rng
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # vector from tip to target is last 3 entries of obs vector
    reward_dist = -jnp.linalg.norm(obs[-3:])
    reward_ctrl = -jnp.square(action).sum()
    #reward = -reward_dist + reward_ctrl
    
    target_hit = jnp.where(reward_dist < 1.0, 1.0, 0.0)
    
    reward = reward_dist + target_hit

    #steps = state.steps + self.action_repeat
    #done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    state.metrics.update(
        rewardDist=reward_dist,
        rewardCtrl=reward_ctrl,
    )

    # teleport any hit targets
    # rng, target = self._random_target(rng)
    # target = jnp.where(target_hit, target, qp.pos[self.target_idx])
    # pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    # qp = dataclasses.replace(qp, pos=pos)
    
    #return env.State(rng, qp, info, obs, reward, done, steps, metrics)
    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    # """Egocentric observation of target and arm body."""
    #
    # #return qp.pos[self.target_idx]
    #
    # qpos1 = [qp.pos[self.servo_idx, :1]]
    # qpos2 = [qp.pos[self.target_idx, :1]]
    # return jnp.concatenate(qpos1 + qpos2)
    """Egocentric observation of target and arm body."""

    # some pre-processing to pull joint angles and velocities
    (joint_angle,), _ = self.sys.joint_revolute.angle_vel(qp)

    # qpos:
    # x,y coord of target
    qpos = [qp.pos[self.target_idx, :2]]

    # dist to target and speed of tip
    arm_qps = take(qp, jnp.array(self.servo_idx))
    tip_pos, tip_vel = math.to_world(arm_qps, jnp.array([0.01, 0., 0.]))
    tip_to_target = [tip_pos - qp.pos[self.target_idx]]
    cos_sin_angle = [jnp.cos(joint_angle), jnp.sin(joint_angle)]

    # qvel:
    # velocity of tip
    qvel = [tip_vel[:2]]

    return jnp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)

  def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a target location in a random circle slightly above xy plane."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    dist = 4.2 * jax.random.uniform(rng1)
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
  name: "block"
  colliders {
    box {
      halfsize {
          x: 0.1
          y: 0.1
          z: 0.1
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
  mass: 0.250
}

bodies {
  name: "body"
  colliders {
    box {
      halfsize {
          x: 0.002
          y: 0.04
          z: 0.01
      }
    }
    position { 
        x: -0.02
        z: 0.0
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
        x: 0.0
        y: 0.04
        z: 0.0
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
        x: 0.0
        y: -0.04
        z: 0.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.002
}

bodies {
  name: "servo_0"
  colliders {
    box {
      halfsize {
          x: 0.04
          y: 0.03
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
  mass: 0.054
}

bodies {
  name: "body_1"
  colliders {
    box {
      halfsize {
          x: 0.002
          y: 0.04
          z: 0.01
      }
    }
    position { 
        x: -0.02
        z: 0.0
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
        x: 0.0
        y: 0.04
        z: 0.0
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
        x: 0.0
        y: -0.04
        z: 0.0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.002
}

bodies {
  name: "servo_1"
  colliders {
    box {
      halfsize {
          x: 0.04
          y: 0.03
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
  mass: 0.054
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
  mass: 0.001
  frozen { all: true }
}

joints {
  name: "joint_block"
  parent: "block"
  child: "body"
  parent_offset {
    x: 0.0
    z: 0.0
  }
  child_offset {
    x: -0.12
  }
  rotation {
    z: 90.0
  }

  angle_limit {
      min: 0
      max: 0
  }

  stiffness: 15000.0
  angular_damping: 1
  spring_damping: 10.0
}

joints {
  name: "joint_body"
  parent: "body"
  child: "servo_0"
  parent_offset {
    x: 0.0
    z: 0.0
  }
  child_offset {
    x: -0.03
  }
  rotation {
    z: 90.0
  }

  angle_limit {
      min: -90
      max: 90
  }

  stiffness: 15000.0
  angular_damping: 1
  spring_damping: 10.0
  limit_strength: 1.0
}

joints {
  name: "joint_body_1"
  parent: "body_1"
  child: "servo_1"
  parent_offset {
    x: 0.0
    z: 0.0
  }
  child_offset {
    x: -0.03
  }
  rotation {
    z: 90.0
  }

  angle_limit {
      min: -90
      max: 90
  }

  stiffness: 15000.0
  angular_damping: 1
  spring_damping: 10.0
  limit_strength: 1.0
}

joints {
  name: "joint_both"
  
  parent: "servo_0"
  child: "body_1"
  parent_offset {
    x: 0.03
    z: 0.0
  }
  child_offset {
    x: -0.03
  }
  rotation {
    z: 90.0
  }

  angle_limit {
      min: 0
      max: 0
  }

  stiffness: 15000.0
  angular_damping: 1
  spring_damping: 10.0
  limit_strength: 1.0
}

actuators {
  name: "actuator_0"
  joint: "joint_body"
  strength: 50.0
  torque {
  }
}

actuators {
  name: "actuator_1"
  joint: "joint_body_1"
  strength: 50.0
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
collide_include {
  first: "body_1"
  second: "Ground"
}
collide_include {
  first: "servo_1"
  second: "Ground"
}
collide_include {
  first: "block"
  second: "Ground"
}

collide_include {
  first: "block"
  second: "body"
}
collide_include {
  first: "block"
  second: "servo_0"
}
collide_include {
  first: "block"
  second: "servo_1"
}
collide_include {
  first: "block"
  second: "body_1"
}

collide_include {
  first: "servo_0"
  second: "body"
}
collide_include {
  first: "servo_0"
  second: "servo_1"
}
collide_include {
  first: "servo_0"
  second: "body_1"
}
collide_include {
  first: "servo_1"
  second: "body_1"
}

baumgarte_erp: 0.1
friction: 0.6
gravity { z: -9.81 }
angular_damping: -0.05

dt: 0.008
substeps: 200

"""
