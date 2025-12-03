# Unitree Go2 Command Center

A comprehensive web-based dashboard for controlling the Unitree Go2 robot, managing computer vision modules, and monitoring telemetry.

## Features
- **Web Dashboard**: Control everything from a browser.
- **Computer Vision**:
  - **YOLOv8**: Object detection.
  - **Hand Detection**: Gesture control.
  - **Depth Camera**: Distance mapping.
  - **Simple Camera**: Low-latency raw feed.
- **Telemetry**: Real-time battery, temperature, and system stats.
- **Motion Control**: Virtual Gamepad (WASD + Numpad) for driving.

## Quick Start

1. **Activate Environment** (if using venv):
   ```bash
   source .venv/bin/activate
   ```

2. **Run Dashboard**:
   ```bash
   # Simulation / Loopback
   python3 -m src.dashboard.app

   # Real Robot (replace eth0 with your interface)
   python3 -m src.dashboard.app eth0
   ```

3. **Open Browser**:
   Navigate to `http://localhost:8000` (or your Jetson's IP).

## Laptop Client (Remote Control via WiFi)

Run the dashboard from your laptop and control the robot wirelessly via WebRTC.
This is useful when you want to:
- Run CV/AI on your powerful laptop GPU instead of the Jetson
- Control the robot remotely without running apps on the robot
- Monitor the robot from any device on the same WiFi network

### Setup

1. **Install WebRTC dependencies** on your laptop:
   ```bash
   pip install unitree-webrtc-connect aiortc aiohttp
   ```

2. **Connect to the Go2's WiFi** or ensure both are on the same network.

3. **Run the Laptop Client**:
   ```bash
   # Replace with your robot's IP address
   python3 -m src.laptop_client 192.168.123.18
   
   # Or edit src/laptop_client/config.py and run:
   python3 -m src.laptop_client
   ```

4. **Open Browser** at `http://localhost:8000`

### Features
- Live camera feed via WebRTC
- WASD + QE keyboard control
- Robot actions (Stand, Sit, Dance, etc.)
- Battery and temperature monitoring
- Low latency connection

## Troubleshooting

### Cleaning up stuck processes
The dashboard now automatically cleans up stuck processes on startup. However, if you need to manually kill everything:

```bash
# Kill all dashboard-related python processes
pkill -f "src\."
```

### Installation
## Dependencies
- Python >= 3.8
- cyclonedds == 0.10.2
- numpy
- opencv-python
- fastapi
- uvicorn
- psutil
- ultralytics (for YOLO)
- mediapipe (for Hand Detection)

### Installing from source
Execute the following commands in the terminal:
```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```
## FAQ
##### 1. Error when `pip3 install -e .`:
```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```
This error mentions that the cyclonedds path could not be found. First compile and install cyclonedds:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
Enter the unitree_sdk2_python directory, set `CYCLONEDDS_HOME` to the path of the cyclonedds you just compiled, and then install unitree_sdk2_python.
```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```
For details, see: https://pypi.org/project/cyclonedds/#installing-with-pre-built-binaries

# Usage
The Python sdk2 interface maintains consistency with the unitree_sdk2 interface, achieving robot status acquisition and control through request-response or topic subscription/publishing. Example programs are located in the `/example` directory. Before running the examples, configure the robot's network connection as per the instructions in the document at https://support.unitree.com/home/en/developer/Quick_start.
## DDS Communication
In the terminal, execute:
```bash
python3 ./example/helloworld/publisher.py
```
Open a new terminal and execute:
```bash
python3 ./example/helloworld/subscriber.py
```
You will see the data output in the terminal. The data structure transmitted between `publisher.py` and `subscriber.py` is defined in `user_data.py`, and users can define the required data structure as needed.
## High-Level Status and Control
The high-level interface maintains consistency with unitree_sdk2 in terms of data structure and control methods. For detailed information, refer to https://support.unitree.com/home/en/developer/sports_services.
### High-Level Status
Execute the following command in the terminal:
```bash
python3 ./example/high_level/read_highstate.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected,.
### High-Level Control
Execute the following command in the terminal:
```bash
python3 ./example/high_level/sportmode_test.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. This example program provides several test methods, and you can choose the required tests as follows:
```python
test.StandUpDown() # Stand up and lie down
# test.VelocityMove() # Velocity control
# test.BalanceAttitude() # Attitude control
# test.TrajectoryFollow() # Trajectory tracking
# test.SpecialMotions() # Special motions
```
## Low-Level Status and Control
The low-level interface maintains consistency with unitree_sdk2 in terms of data structure and control methods. For detailed information, refer to https://support.unitree.com/home/en/developer/Basic_services.
### Low-Level Status
Execute the following command in the terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The program will output the state of the right front leg hip joint, IMU, and battery voltage.
### Low-Level Motor Control
First, use the app to turn off the high-level motion service (sport_mode) to prevent conflicting instructions.
Execute the following command in the terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The left hind leg hip joint will maintain a 0-degree position (for safety, set kp=10, kd=1), and the left hind leg calf joint will continuously output 1Nm of torque.
## Wireless Controller Status
Execute the following command in the terminal:
```bash
python3 ./example/wireless_controller/wireless_controller.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The terminal will output the status of each key. For the definition and data structure of the remote control keys, refer to https://support.unitree.com/home/en/developer/Get_remote_control_status.
## Front Camera
Use OpenCV to obtain the front camera (ensure to run on a system with a graphical interface, and press ESC to exit the program):
```bash
python3 ./example/front_camera/camera_opencv.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected.

## Obstacle Avoidance Switch
```bash
python3 ./example/obstacles_avoid_switch/obstacles_avoid_switch.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The robot will cycle obstacle avoidance on and off. For details on the obstacle avoidance service, see https://support.unitree.com/home/en/developer/ObstaclesAvoidClient

## Light and volume control
```bash
python3 ./example/vui_client/vui_client_example.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected.T he robot will cycle the volume and light brightness. The interface is detailed at https://support.unitree.com/home/en/developer/VuiClient

## WebRTC Camera Streaming
To stream the Go2 camera via WebRTC using `unitree_webrtc_connect`:

1. Install dependencies:
```bash
pip install -r requirements-webrtc.txt
```

2. Run the WebRTC viewer:
```bash
# Set configuration via environment variables if needed (defaults to LocalSTA 192.168.8.181)
export WEBRTC_METHOD=LocalSTA
export WEBRTC_IP=192.168.8.181

python3 -m src.webrtc.run
```
