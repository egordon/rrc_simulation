#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import sys

import gym
import enum
import pybullet

import numpy as np
from scipy.spatial.transform import Rotation as R

from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube

class States(enum.Enum):
    """ Different States for StateSpacePolicy """

    #: Align fingers to 3 points above cube
    ALIGN = enum.auto()

    #: Lower coplanar with cube
    LOWER = enum.auto()

    #: Move into cube
    INTO = enum.auto()

    #: Move cube to goal
    GOAL = enum.auto()

    #: Orient correctly
    ORIENT = enum.auto()

def _quat_mult(q1, q2):
    x0, y0, z0, w0 = q2
    x1, y1, z1, w1 = q1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])

def _quat_conj(q):
    ret = np.copy(q)
    ret[:3] *= -1
    return ret

def _get_angle_axis(current, target):
    # Return:
    # (1) angle err between orientations
    # (2) Unit rotation axis
    rot = R.from_quat(_quat_mult(current, _quat_conj(target)))

    rotvec = rot.as_rotvec()
    norm = np.linalg.norm(rotvec)

    if norm > 1E-8:
        return norm, (rotvec / norm)
    else:
        return 0, np.zeros(len(rotvec))

class StateSpacePolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, env, difficulty, observation):
        self.action_space = env.action_space
        self.finger = env.platform.simfinger
        self.state = States.ALIGN
        self.difficulty = difficulty

        self.EPS = 1E-2
        self.DAMP = 1E-6
        self.CUBE_SIZE = 0.0325

        if difficulty == 4:
            # Do Pre-manpulation
            self.do_premanip = True
            self._calculate_premanip(observation)
        else:
            self.do_premanip = False

        self.t = 0

    def _calculate_premanip(self, observation):
        current = observation["achieved_goal"]["orientation"]
        target = observation["desired_goal"]["orientation"]

        # Sets pre-manipulation 90 or 180-degree rotation
        self.manip_angle = 0
        self.manip_axis = np.zeros(3)
        self.manip_arm = 0

        minAngle, _ = _get_angle_axis(current, target)

        # Check 90-degree rotations
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])]:
            new_axis = R.from_quat(current).apply(axis)
            rotation = R.from_rotvec(np.pi / 2 * new_axis)
            new_current = rotation * R.from_quat(current)

            angle, _ = _get_angle_axis(new_current.as_quat(), target)
            if angle < minAngle:
                minAngle = angle
                self.manip_angle = 90
                self.manip_axis = new_axis

        # Check 180 degree rotation
        """ NO TIME FOR 180
        new_axis = R.from_quat(current).apply(np.array([1, 0, 0]))
        rotation = R.from_rotvec(np.pi * new_axis)
        new_current = rotation * R.from_quat(current)
        angle, _ = _get_angle_axis(new_current.as_quat(), target)
        if angle < minAngle:
            minAngle = angle
            self.manip_angle = 180
            self.manip_axis = new_axis
        """

        # Determine rotation arm
        arm_angle = np.arctan2(self.manip_axis[1], self.manip_axis[0]) + np.pi/2
        if arm_angle > np.pi:
            arm_angle -= 2*np.pi
        print("Arm Angle: " + str(arm_angle))

        if arm_angle < (np.pi/2 + np.pi/3) and arm_angle > (np.pi/2 - np.pi/3):
            self.manip_arm = 0
        elif arm_angle > (-np.pi/2) and arm_angle < (np.pi/2 - np.pi/3):
            self.manip_arm = 1
        else:
            self.manip_arm = 2

        print("Manip Arm: " + str(self.manip_arm))
        print("Manip Angle: " + str(self.manip_angle))


    def _get_gravcomp(self, observation):
        # Returns: 9 torques required for grav comp
        ret = pybullet.calculateInverseDynamics(self.finger.finger_id, 
                        observation["observation"]["position"].tolist(),
                        np.zeros(len(observation["observation"]["position"])).tolist(),
                        np.zeros(len(observation["observation"]["position"])).tolist())
        ret = np.array(ret)
        return ret

    def _get_jacobians(self, observation):
        # Returns: numpy(3*num_fingers X num_joints_per_finger)
        ret = []
        for tip in self.finger.pybullet_tip_link_indices:
            J, _ = pybullet.calculateJacobian(
                    self.finger.finger_id,
                    tip,
                    np.zeros(3).tolist(),
                    observation["observation"]["position"].tolist(),
                    observation["observation"]["velocity"].tolist(),
                    np.zeros(len(observation["observation"]["position"])).tolist()
                    )
            ret.append(J)
        ret = np.vstack(ret)
        return ret

    def _get_tip_poses(self, observation):
        # Return: numpy(3 * num_fingers)
        tips = []
        for tip in self.finger.pybullet_tip_link_indices:
            state = pybullet.getLinkState(
                                self.finger.finger_id,
                                tip)
            tips.append(np.array(state[0]))
        tips = np.array(tips).flatten()
        return tips

    def prealign(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.manip_arm + 1 - i) % 3
            locs[index] = 1.5 * R.from_rotvec(np.pi/2 * i * np.array([0, 0, 1])).apply(self.manip_axis)
            locs[index][2] = 2

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
                    self.CUBE_SIZE * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 3 * self.EPS:
            print("PRE LOWER")
            self.state = States.LOWER
        return 0.08 * err

    def prelower(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        # Determine arm locations
        locs = [np.zeros(3), np.zeros(3), np.zeros(3)]

        for i in range(3):
            index = (self.manip_arm + 1 - i) % 3
            locs[index] = 1.5 * R.from_rotvec(np.pi/2 * i * np.array([0, 0, 1])).apply(self.manip_axis)
            if i == 1:
                locs[index][2] += 0.4

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
                    self.CUBE_SIZE * np.hstack(locs)

        err = desired - current
        if np.linalg.norm(err) < 3*self.EPS:
            self.previous_state = observation["observation"]["position"]
            print("PRE INTO")
            self.state = States.INTO
        return 0.08 * err

    def preinto(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)
        desired[3*self.manip_arm+2] += 0.4*self.CUBE_SIZE

        err = desired - current

        # Lower force of manip arm
        err[3*self.manip_arm:3*self.manip_arm + 3] *= 0.5

        # Read Tip Force
        tip_forces = self.finger._get_latest_observation().tip_force
        switch = True
        for f in tip_forces:
            if f < 0.051:
                switch = False

        # Override with small diff
        diff = observation["observation"]["position"] - self.previous_state
        self.previous_state = observation["observation"]["position"]

        if np.amax(diff) < 5e-5:
            switch = True

        if switch:
            self.pregoal_state = observation["achieved_goal"]["position"]
            print("PRE GOAL")
            self.state = States.GOAL

        return 0.1 * err

    def pregoal(self, observation):
        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)
        # Lower force of manip arm
        into_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        goal = self.pregoal_state
        goal[2] = 3 * self.CUBE_SIZE
        goal = np.tile(goal, 3)
        goal_err = goal - desired
        goal_err[3*self.manip_arm:3*self.manip_arm + 3] *= 0

        rot_err = np.zeros(9)
        rot_err[3*self.manip_arm:3*self.manip_arm + 3] = observation["achieved_goal"]["position"] + np.array([0, 0, 1.5 * self.CUBE_SIZE])
        rot_err[3*self.manip_arm:3*self.manip_arm + 3] -= current[3*self.manip_arm:3*self.manip_arm + 3]


        # Once manip arm is overhead, drop
        diff = np.linalg.norm(current[3*self.manip_arm:3*self.manip_arm+2] - observation["achieved_goal"]["position"][:2])
        #print("Diff: " + str(diff))

        #print("End condition: " + str(diff < 0.75 * self.CUBE_SIZE))
        if diff < 0.5 * self.CUBE_SIZE:
            print("PRE ORIENT")
            self.state = States.ORIENT
        # Once high enough, drop
        #if observation["achieved_goal"]["position"][2] > 2 * self.CUBE_SIZE:
        #    print("PRE ORIENT")
        #    self.state = States.ORIENT

        # Override with no force on manip arm
        #tip_forces = self.finger._get_latest_observation().tip_force
        #for f in tip_forces:
        #    if f < 0.051:
        #        print("PRE ORIENT")
        #        self.state = States.ORIENT

        # Override with small diff
        #diff = observation["observation"]["position"] - self.previous_state
        #self.previous_state = observation["observation"]["position"]

        #if np.amax(diff) < 1e-6:
        #    switch = True

        return 0.05 * into_err + 0.1 * goal_err + 0.25 * rot_err

    def preorient(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        err = current - desired

        # Read Tip Force
        tip_forces = self.finger._get_latest_observation().tip_force
        switch = False
        for f in tip_forces:
            if f < 0.051:
                switch = True
        if switch:
            self.manip_angle -= 90
            print("MANIP DONE")
            self.state = States.ALIGN

        return 0.1 * err

    def premanip(self, observation):
        force = np.zeros(9)

        if self.state == States.ALIGN:
            force = self.prealign(observation)

        
        elif self.state == States.LOWER:
            force = self.prelower(observation)

        
        elif self.state == States.INTO:
            force = self.preinto(observation)

        elif self.state == States.GOAL:
            force = self.pregoal(observation)

        elif self.state == States.ORIENT:
            force = self.preorient(observation)

        if self.manip_angle == 0:
            self.do_premanip = False
            self.state = States.ALIGN

        return force


    def align(self, observation):
        # Return torque for align step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
                    self.CUBE_SIZE * np.array([0, 1.6, 2, 1.6 * 0.866, 1.6 * (-0.5), 2, 1.6 * (-0.866), 1.6 * (-0.5), 2])

        err = desired - current
        if np.linalg.norm(err) < 2 * self.EPS:
            self.state = States.LOWER
        return 0.1 * err

    def lower(self, observation):
        # Return torque for lower step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3) + \
                    self.CUBE_SIZE * np.array([0, 1.6, 0, 1.6 * 0.866, 1.6 * (-0.5), 0, 1.6 * (-0.866), 1.6 * (-0.5), 0])

        err = desired - current
        if np.linalg.norm(err) < 2 * self.EPS:
            self.state = States.INTO
        return 0.1 * err

    def into(self, observation):
        # Return torque for into step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        err = desired - current

        # Read Tip Force
        tip_forces = self.finger._get_latest_observation().tip_force
        switch = True
        for f in tip_forces:
            if f < 0.0515:
                switch = False
        if switch:
            self.state = States.GOAL

        self.goal_err_sum = np.zeros(9)

        return 0.1 * err

    def goal(self, observation):
        # Return torque for goal step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        if self.difficulty == 1:
            goal[2] += 0.001 # Reduces friction with floor
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        if err_mag < 0.01 and self.difficulty == 4:
            self.state = States.ORIENT

        return 0.04 * into_err + 0.11 * goal_err + 0.0004 * self.goal_err_sum

    def orient(self, observation):
        # Return torque for lower step
        current = self._get_tip_poses(observation)

        desired = np.tile(observation["achieved_goal"]["position"], 3)

        into_err = desired - current
        into_err /= np.linalg.norm(into_err)

        goal = np.tile(observation["desired_goal"]["position"], 3)
        goal_err = goal - desired
        err_mag = np.linalg.norm(goal_err[:3])

        if err_mag < 0.1:
            self.goal_err_sum += goal_err

        angle, axis = _get_angle_axis(observation["achieved_goal"]["orientation"], observation["desired_goal"]["orientation"])
        ang_err = np.zeros(9)
        ang_err[:3] = -angle * np.cross(into_err[:3] / np.linalg.norm(into_err[:3]), axis)
        ang_err[3:6] = -angle * np.cross(into_err[3:6] / np.linalg.norm(into_err[3:6]), axis)
        ang_err[6:] = -angle * np.cross(into_err[6:] / np.linalg.norm(into_err[6:]), axis)

        return 0.04 * into_err + 0.11 * goal_err + 0.0004 * self.goal_err_sum + 0.006 * ang_err

    def predict(self, observation):
        # Get Jacobians
        J = self._get_jacobians(observation)
        self.t += 1

        force = np.zeros(9)

        if self.do_premanip:
            force = self.premanip(observation)

        elif self.state == States.ALIGN:
            force = self.align(observation)

        elif self.state == States.LOWER:
            force = self.lower(observation)

        elif self.state == States.INTO:
            force = self.into(observation)

        elif self.state == States.GOAL:
            force = self.goal(observation)

        elif self.state == States.ORIENT:
            force = self.orient(observation)

        torque = J.T.dot(np.linalg.solve(J.dot(J.T) + self.DAMP * np.eye(9), force))
        
        ret = torque + self._get_gravcomp(observation)
        return ret

def main():
    try:
        difficulty = int(sys.argv[1])
        initial_pose_json = sys.argv[2]
        goal_pose_json = sys.argv[3]
        output_file = sys.argv[4]
    except IndexError:
        print("Incorrect number of arguments.")
        print(
            "Usage:\n"
            "\tevaluate_policy.py <difficulty_level> <initial_pose>"
            " <goal_pose> <output_file>"
        )
        sys.exit(1)

    # the poses are passes as JSON strings, so they need to be converted first
    initial_pose = move_cube.Pose.from_json(initial_pose_json)
    goal_pose = move_cube.Pose.from_json(goal_pose_json)

    # create a FixedInitializer with the given values
    initializer = cube_env.FixedInitializer(
        difficulty, initial_pose, goal_pose
    )

    # TODO: Replace with your environment if you used a custom one.
    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.TORQUE,
        frameskip=1,
        visualization=True,
    )
    observation = env.reset()

    # TODO: Replace this with your model
    # Note: You may also use a different policy for each difficulty level (difficulty)
    policy = StateSpacePolicy(env, difficulty, observation)

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    is_done = False
    accumulated_reward = 0
    while not is_done:
        action = policy.predict(observation)
        observation, reward, is_done, info = env.step(action)
        accumulated_reward += reward

    print("Accumulated reward: {}".format(accumulated_reward))
    print("Reward at final step: {:.3f}".format(reward))

    # store the log for evaluation
    env.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
