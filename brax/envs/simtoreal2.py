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

"""Trains a reacher to reach a target.

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

from google.protobuf import text_format


class SimToReal(env.Env):
  """Trains a reacher arm to touch a sequence of random targets."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.target_idx = self.sys.body_idx['target']
    self.arm_idx = self.sys.body_idx['arm_1']
    self.ball_idx = self.sys.body_idx['ball']

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = dataclasses.replace(qp, pos=pos)
    
    
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    #print(f"obs >{obs}<")
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
    reward_dist = -jnp.linalg.norm(obs[-3:])
    reward_ctrl = -jnp.square(action).sum()
    reward = reward_dist + reward_ctrl

    steps = state.steps + self.action_repeat
    done = jnp.where(steps >= self.episode_length, 1.0, 0.0)
    metrics = {
        'rewardDist': reward_dist,
        'rewardCtrl': reward_ctrl,
    }

    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Egocentric observation of target and arm body."""

    # some pre-processing to pull joint angles and velocities
    (joint_angle,), _ = self.sys.joint_revolute.angle_vel(qp)

    # qpos:
    # x,y coord of target
    qpos = [qp.pos[self.target_idx, :2]]

    # dist to target and speed of tip
    arm_qps = take(qp, jnp.array(self.arm_idx))
    tip_pos, tip_vel = math.to_world(arm_qps, jnp.array([0.11, 0., 0.]))
    tip_to_target = [tip_pos - qp.pos[self.target_idx]]
    cos_sin_angle = [jnp.cos(joint_angle), jnp.sin(joint_angle)]

    # qvel:
    # velocity of tip
    qvel = [tip_vel[:2]]

    return jnp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)

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
  name: "arm_0"
  colliders {
    capsule {
      radius: 0.005
      length: 0.15
    }
    position { 
        x: 0.0
        z: 0.0
        y: 0.055
    }
    rotation {
      x: -90
    }
  }
  
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.01
}

bodies {
  name: "arm_1"
  colliders {
    capsule {
      radius: 0.005
      length: 0.15
    }
    position { 
        x: 0.0
        z: 0.0
        y: 0.055
    }
    rotation {
      x: -90
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.01
}


bodies {
  name: "target"
  colliders {
    position {
    }
    sphere {
      radius: 0.03
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
  name: "ball"
  colliders {
    position {
        x: 0.15
    }
    capsule {
      radius: 0.03
      length: 0.06
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.01
}

bodies {
  name: "wall_0"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: 0.3
        z: 0.0
        y: 0.0
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
  name: "wall_1"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: 0.2
        z: 0.0
        y: 0.2
    }
    rotation {
        z: 45
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
  name: "wall_2"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: 0.0
        z: 0.0
        y: 0.3
    }
    rotation {
        z: 90
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
  name: "wall_3"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: -0.2
        z: 0.0
        y: 0.2
    }
    rotation {
        z: 135
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
  name: "wall_4"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: -0.3
        z: 0.0
        y: 0.0
    }
    rotation {
        z: 0
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
  name: "wall_5"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: -0.2
        z: 0.0
        y: -0.2
    }
    rotation {
        z: 45
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
  name: "wall_6"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: 0.0
        z: 0.0
        y: -0.3
    }
    rotation {
        z: 90
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
  name: "wall_7"
  colliders {
    box {
      halfsize { 
          x: 0.005 
          y: 0.17
          z: 0.1
        }
    }
    position { 
        x: 0.2
        z: 0.0
        y: -0.2
    }
    rotation {
        z: 135
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
  parent: "ground"
  child: "arm_0"
  parent_offset {
    z: .01
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

  limit_strength: 0.0
  spring_damping: 3.0
}

joints {
  name: "joint_arm"
  stiffness: 100.0
  parent: "arm_0"
  child: "arm_1"
  parent_offset {
    z: .01
    y: 0.11
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

  limit_strength: 0.0
  spring_damping: 3.0
}

actuators {
  name: "joint_ground"
  joint: "joint_ground"
  strength: 25.0
  torque {
  }
}
actuators {
  name: "joint_arm"
  joint: "joint_arm"
  strength: 25.0
  torque {
  }
}

collide_include {
    first: "ground"
    second: "ball"
}
collide_include {
    first: "ball"
    second: "ground"
}
collide_include {
    first: "arm_0"
    second: "ball"
}
collide_include {
    first: "arm_1"
    second: "ball"
}
collide_include {
    first: "target"
    second: "ground"
}
collide_include {
    first: "target"
    second: "wall_0"
}
collide_include {
    first: "target"
    second: "wall_1"
}
collide_include {
    first: "target"
    second: "wall_2"
}
collide_include {
    first: "target"
    second: "wall_3"
}
collide_include {
    first: "target"
    second: "wall_4"
}
collide_include {
    first: "target"
    second: "wall_5"
}
collide_include {
    first: "target"
    second: "wall_6"
}
collide_include {
    first: "target"
    second: "wall_7"
}
friction: 0.6
gravity {
  z: -9.81
}
baumgarte_erp: 0.1
dt: 0.02
substeps: 4

"""
