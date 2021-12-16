"""Trains an acrobot to swing up and balance.
"""

from typing import Tuple

import brax
from brax import jumpy as jp
from brax.envs import env


class AcrobotAngle(env.Env):
  """Trains an acrobot to swing up and balance."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.arm_idx = self.sys.body.index['body1']

  def reset(self, rng: jp.ndarray) -> env.State:
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'rewardDist': zero,
        'rewardCtrl': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    reward_dist = -jp.norm(obs[0:2])
    reward_ctrl = -jp.square(action).sum()
    reward = reward_dist + reward_ctrl

    state.metrics.update(
        rewardDist=reward_dist,
        rewardCtrl=reward_ctrl,
    )

    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:

    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vels,) = self.sys.joints[0].angle_vel(qp)

    #cos_sin_angle = [jp.cos(joint_angle), jp.sin(joint_angle)]

    # qvel:
    # velocity of tip

    return jp.concatenate([joint_angle, joint_vels])


_SYSTEM_CONFIG = """
bodies {
  name: "ground"
  colliders {
    plane {
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
  name: "body0"
  colliders {
    position {
      x: 0.05
    }
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.01
      length: 0.12
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
  name: "body1"
  colliders {
    position {
      x: 0.05
    }
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.01
      length: 0.12
    }
  }
  colliders {
    position { x: .11 }
    sphere {
      radius: 0.01
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.035604715
}
joints {
  name: "joint0"
  stiffness: 100.0
  parent: "ground"
  child: "body0"
  parent_offset {
    z: 0.01
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
  name: "joint1"
  stiffness: 100.0
  parent: "body0"
  child: "body1"
  parent_offset {
    x: 0.1
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
  name: "joint1"
  joint: "joint1"
  strength: 25.0
  angle {
  }
}
collide_include {
}
gravity {
  y: -9.81
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
