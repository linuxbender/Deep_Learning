import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):

    def __init__(self):
        cube_size = 300.0
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([ cube_size / 2, cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force]),
            np.array([ max_force,  max_force,  max_force]))
        
        self.max_duration = 5.0
        self.target_z = 10.0

    def reset(self):  
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def takeoff(self, timestamp, pose, reward ):
        if pose.position.z >= self.target_z:
            reward += 10.0
            timestamp = 0
            # 2 task
            self.hover(self, timestamp, pose, reward )
        elif pose.position.z > 0 and pose.position.z < self.target_z  and timestamp < self.max_duration:
            reward += (self.target_z / 100 ) * pose.position.z
            timestamp -= 1.5
        elif timestamp > self.max_duration:
            reward -= 10.0
            done = True


    def hover(self, timestamp, pose, reward ):
        if pose.position.z == self.target_z:
            reward += 10.0
            timestamp = 0
            # 3 task
            self.landing(self, timestamp, pose, reward )
        elif pose.position.z >= (self.target_z - 0.5) and timestamp < self.max_duration:
            reward += 5
        elif pose.position.z <= (self.target_z + 0.5) and timestamp < self.max_duration:
            reward += 5
        elif timestamp > self.max_duration:
            reward -= 10.0
            done = True


    def landing(self, timestamp, pose, reward ):
        if pose.position.z == 0:
            reward += 10.0
            done = True
        elif pose.position.z < self.target_z and timestamp < self.max_duration:
            reward += (self.target_z - pose.position.z / 100)
        elif timestamp > self.max_duration:
            reward -= 10.0
            done = True


    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # debugger
        # import pdb; pdb.set_trace()

        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        
        done = False
        reward = -min(abs(self.target_z - pose.position.z), 20.0)

        # 1 task
        self.takeoff(timestamp, pose , reward)

        action = self.agent.step(state, reward, done)
        
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
