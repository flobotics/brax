from typing import Tuple

import dataclasses
import jax
import jax.numpy as jnp
import brax
from brax.envs import env
from brax.physics import math
from brax.physics.base import take

from google.protobuf import text_format




class SkeletonEnv(env.Env):
  """Trains a reacher arm to touch a sequence of random targets."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.target_idx = self.sys.body_idx['target']
    self.arm_idx = self.sys.body_idx['box_2']

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    qp = dataclasses.replace(qp, pos=pos)
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
    target_x = 4.0 + dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = 0.51
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target



_SYSTEM_CONFIG = """
bodies {
  name: "box_1"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  inertia {
    x: 1
    y: 1
    z: 1
  }
  mass: 1.0
}

bodies {
  name: "box_2"
  colliders {
    capsule {
      radius: 0.2
      length: 1.8
    }
  }
  inertia {
    x: 1
    y: 1
    z: 1
  }
  mass: 1.0
}

bodies {
  name: "Ground"
  colliders {
    plane {
    }
  }
  frozen {
    all: true
  }
}

bodies {
  name: "target"
  colliders {
    sphere {
      radius: 0.09
    }
  }
  frozen { all: true }
}

joints {
  name: "joint0"
  parent: "box_1"
  child: "box_2"
  parent_offset {
    x: 0.55
  }
  child_offset {
    x: -0.25
  }
  angle_limit {
      min: -60
      max: 60
  }
  rotation { x: 90 }
  limit_strength: 0.0
  spring_damping: 30.0
  stiffness: 1000.0
}

actuators {
  name: "joint0"
  joint: "joint0"
  strength: 80.0
  torque {
  }
}

collide_include {
  first: "box_1"
  second: "Ground"
}
collide_include {
  first: "box_2"
  second: "Ground"
}

gravity { z: -9.81 }
baumgarte_erp: 0.1

friction: 0.6


dt: 0.02
substeps: 4
"""


