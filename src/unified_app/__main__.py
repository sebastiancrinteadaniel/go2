"""Main entry point for Unified Dashboard Application

Run with: python -m src.unified_app [options]
"""

import sys
import argparse
import uvicorn
from src.unified_app.config import (
    CameraConfig, YoloConfig, HandDetectionConfig, DepthCameraConfig,
    SimpleCameraConfig, StreamingConfig, ServerConfig, AppConfig,
    DEFAULT_CONFIG
)
from src.unified_app.app import app


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Go2 Dashboard - Async Video Processing"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_CONFIG.server.host,
        help=f"Server host (default: {DEFAULT_CONFIG.server.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.server.port,
        help=f"Server port (default: {DEFAULT_CONFIG.server.port})",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CONFIG.camera.source,
        choices=["go2", "usb"],
        help=f"Camera source (default: {DEFAULT_CONFIG.camera.source})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    # Create config with CLI overrides
    config = AppConfig(
        camera=CameraConfig(
            source=args.camera,
            width=DEFAULT_CONFIG.camera.width,
            height=DEFAULT_CONFIG.camera.height,
            fps=DEFAULT_CONFIG.camera.fps,
            interface=DEFAULT_CONFIG.camera.interface,
        ),
        yolo=DEFAULT_CONFIG.yolo,
        hand_detection=DEFAULT_CONFIG.hand_detection,
        depth_camera=DEFAULT_CONFIG.depth_camera,
        simple_camera=DEFAULT_CONFIG.simple_camera,
        streaming=DEFAULT_CONFIG.streaming,
        server=ServerConfig(
            host=args.host,
            port=args.port,
            debug=args.debug,
            log_level="debug" if args.debug else "info",
        ),
    )

    print("\n" + "=" * 70)
    print("UNIFIED GO2 DASHBOARD - ASYNC VIDEO PROCESSING")
    print("=" * 70)
    print(f"Host: {config.server.host}")
    print(f"Port: {config.server.port}")
    print(f"Camera: {config.camera.source}")
    print(f"Debug: {config.server.debug}")
    print(f"URL: http://{config.server.host}:{config.server.port}")
    print("=" * 70 + "\n")

    # Run server
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
