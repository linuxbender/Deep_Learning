## Deep Learning - project RL

### Open terminal, say yes to the command prompt

- go to
 ```
 $ cd catkin_ws/
 ```

- catkin_make  # build command  
```
$ source devel/setup.bash
$ roslaunch quad_controller_rl rl_controller.launch
```

### Jupter Notebook
- start jupyter notebook
```
$ cd ~/catkin_ws/src/RL-Quadcopter/quad_controller_rl
$ jupyter notebook
```

- open jupyter notebook on the host (replace the ip)
```
http://0.0.0.0:8888/?token=61ad30e82ee94f6a1bf9fc62a7de9cbffe9e775270a99dcb
```

### ROC settings on the host
my current network settings simulator
```json
{
	"vm-ip" : "192.168.192.3",
	"vm-port" : 11311,
	"vm-override" : true,
	"host-ip": "192.168.192.1",
	"host-override" : true
}
```

### ROC run with DDPG Agent for the takeoff task
```
$ roslaunch quad_controller_rl rl_controller.launch agent:=DDPG
$ roslaunch quad_controller_rl rl_controller.launch task:=Takeoff agent:=DDPG
$ roslaunch quad_controller_rl rl_controller.launch task:=Hover agent:=DDPG
$ roslaunch quad_controller_rl rl_controller.launch task:=Landing agent:=DDPG
$ roslaunch quad_controller_rl rl_controller.launch task:=Combined agent:=DDPG
```