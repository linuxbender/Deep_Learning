import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):
    def __init__(self):
        # debugger
        # import pdb; pdb.set_trace()

        cube_size = 300.0
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2.0, - cube_size / 2.0, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2.0, cube_size / 2.0, cube_size,  1.0,  1.0,  1.0,  1.0]))

        max_force = 25.0

        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force]),
            np.array([ max_force, max_force, max_force]))

        self.max_duration = 5.0
        self.target_top_z = 10.0
        self.target_down_z = 1.0
        self.threshold = 1

    def reset(self):
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # debugger
        # import pdb; pdb.set_trace()

        # linear_acceleration.z
        # angular_velocity.z
        # pose.position.z

        # pos = state[0,:3]
        # vel = state[0,3:6]
        # av = state[0,6:9]        

        state = np.array([pose.position.x, pose.position.y, pose.position.z])
        done = False
        reward = -min(abs(self.target_top_z - pose.position.z), 20.0)
        # Takeoff        
        if -self.threshold+self.target_top_z < pose.position.z < self.threshold+self.target_top_z:
            reward += 10
            print("Takeoff")
            # Hover
            if (self.max_duration - timestamp) <= 2.0:
                reward += 10.0
                timestamp = 0 # reset time
                print("Hover")
                # Landing
                if -self.threshold+self.target_down_z < pose.position.z < self.threshold+self.target_down_z:
                    if (self.max_duration - timestamp) <= 2.0:
                        reward += 500.0
                        done = True
                        print("Landing")
        if not -self.threshold < pose.position.x < self.threshold:
            reward -= 10.0
            done = True
        if not -self.threshold < pose.position.y < self.threshold:
            reward -= 10.0
            done = True
        elif timestamp > self.max_duration:
            reward -= 200.0
            done = True
        
        action = self.agent.step(state, reward, done)

        if action is not None:            
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)
            #print("Action: {}, {}, {} ".format(action[0], action[1], action[2]))
            # z-position, z-linear acceleration, and a calculated per-timestep z-velocity
            return Wrench(force=Vector3(action[0], action[1], action[2])), done
        else:
            print("Empty Wrench no action...")
            return Wrench(), done
