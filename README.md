# Go2 Dashboard (c2-go2)

This is a web-based dashboard and control application for the Unitree Go2 robot. It provides video streaming, telemetry data, and remote control capabilities.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Installation
1. Clone the repository
2. Install dependencies using `uv`:

```bash
uv sync
```

### Running the Application
To start the FastAPI server with the dashboard, run:

```bash
uv run python app/main.py
```

The server will start on `http://0.0.0.0:8000`. Open this URL in your browser to access the dashboard.

## Connecting to the Robot

To connect to the Go2 robot directly via SSH, use the following commands depending on your connection method:

### Via Ethernet (ETH)
```bash
ssh unitree@192.168.123.18
```

### Via WiFi
```bash
ssh unitree@192.168.57.xxx
```
*(Replace `xxx` with the specific IP address of your robot on the network)*
