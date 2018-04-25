import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 600
        self.action_high = 625
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class TakeOffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        # Init Values
        init_pose = init_pose if init_pose is not None else np.array([0., 0., 10., 0., 0., 0.])
        init_velocities = init_velocities if init_velocities is not None else np.array([0., 0., 0.])
        init_angle_velocities = init_angle_velocities if init_angle_velocities is not None else np.array([0., 0., 0.])

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 600
        self.action_high = 625
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 20.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # distanceFromTarget = 1 -.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        #  zV = self.sim.pose[2]
        hV = abs(self.sim.v[0]) + abs(self.sim.v[1])
        hVPenalty = -0.1 * hV

        # print("---")
        # print(self.sim.pose)
        # print(self.sim.v)
        # print("---")


        # distancePenalty = 1 -.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        posReward = 0.2 if self.sim.pose[2] > self.sim.init_pose[2] else -0.2
        vSpeedReward = 0.1 * self.sim.v[2]
        # angularOrientationPenalty = -0.1 * (abs(self.sim.pose[3]) + abs(self.sim.pose[4]) + abs(self.sim.pose[5]))

        reward = posReward + vSpeedReward

        # clamp reward between -1 and 1
        reward = np.tanh(reward)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

