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

"""An inverted pendulum environment."""

import brax
from brax import jumpy as jp
from brax.envs import env
import math

class Pendulum(env.Env):
  """Trains an inverted pendulum to remain stationary."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), math.pi - .01, math.pi + .01)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.01, .01)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'survive_reward': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""

    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    reward = -(obs[0] - 0.0)**2
    #reward = 1.0
    
    #done = jp.where(qp.pos[1, 2] > .2, jp.float32(0), jp.float32(1))
    done = jp.float32(0)
    state.metrics.update(survive_reward=reward)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    return jp.concatenate((joint_angle, joint_vel))




_SYSTEM_CONFIG = """
bodies {
  name: "cart"
  colliders {
    rotation {
    x: 90
    z: 90
    }
    capsule {
      radius: 0.1
      length: 0.4
    }
  }
  frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
  mass: 10.471975
}
bodies {
  name: "pole"
  colliders {
    capsule {
      radius: 0.049
      length: 0.69800085
    }
  }
  frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
  mass: 5.0185914
}
joints {
  name: "hinge"
  stiffness: 10000.0
  parent: "cart"
  child: "pole"
  child_offset { z: -.3 }
  rotation {
    z: 90.0
  }
  limit_strength: 0.0
  angle_limit { min: -360.0 max: 360.0 }
}

actuators{
  name: "hinge"
  joint: "hinge"
  strength: 25.0
  torque{
  }
}
collide_include {}
gravity {
  z: -9.81
}
dt: 0.04
substeps: 8
"""
