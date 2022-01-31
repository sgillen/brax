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

"""An acrobot environment."""

import brax
from brax import jumpy as jp
from brax.envs import env
import jax


class Acrobot(env.Env):
  """Trains an acrobot to remain stationary."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.01, .01)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.01, .01)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    #    qp = self.sys.default_qp()
    info = self.sys.info(qp)
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    obs = self._get_obs(qp, info, joint_angle, joint_vel)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'dist_penalty': zero,
        'vel_penalty': zero,
        'alive_bonus': zero,
        'r_tot': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    obs = self._get_obs(qp, info, joint_angle, joint_vel)

    alive_bonus = 10.0
    

    #r = jp.sum(-(obs**2))
    r = -jp.sum(joint_angle**2) + -.01*jp.sum(joint_vel**2) + alive_bonus
    
    #done = jp.where(y <= 1, jp.float32(1), jp.float32(0))
    #done = jp.where(jp.abs(joint_angle[0]) >= .5, jp.float32(1), jp.float32(0))

    done = jp.float32(0);
    state.metrics.update(
#        dist_penalty=dist_penalty,
#        vel_penalty=vel_penalty,
        r_tot=r)

    return state.replace(qp=qp, obs=obs, reward=r, done=done)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, qp: brax.QP, info: brax.Info, joint_angle: jp.ndarray,
               joint_vel: jp.ndarray) -> jp.ndarray:
    """Observe cartpole body position and velocities."""

    # position_obs = [
    #     jp.array([qp.pos[0, 0]]),  # cart x pos
    #     jp.sin(joint_angle),  # link angles
    #     jp.cos(joint_angle)
    # ]

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
      length: 1.0 
    }
  }
  frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
  mass: 2.5
}
joints {
  name: "hinge"
  stiffness: 30000.0
  parent: "cart"
  child: "pole"
  child_offset { z: -.45 }
  rotation {
    z: 90.0
  }
  limit_strength: 0.0
  spring_damping: 500.0
  angle_limit { min: 0.0 max: 0.0 }
}
bodies {
  name: "pole2"
  colliders {
    capsule {
      radius: 0.049
      length:  1.00 
    }
  }
  frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
  mass: 2.5
}
joints {
  name: "hinge2"
  stiffness: 30000.0
  parent: "pole"
  child: "pole2"
  parent_offset { z: .45 }
  child_offset { z: -.45 }
  rotation {
    z: 90.0
  }
  limit_strength: 0.0
  spring_damping: 500.0
  angle_limit { min: 0.0 max: 0.0 }
}

actuators{
  name: "hinge2"
  joint: "hinge2"
  strength: 25.0
  torque{
  }
}

defaults {
    angles { 
        name: "hinge" 
        angle{ x: 180.0 y: 0.0 z: 0.0} 
    }
}


collide_include {}
gravity {
  z: -9.81
}
dt: 0.01
substeps: 4
"""
