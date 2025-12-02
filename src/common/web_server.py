import asyncio
import logging
import socket
import threading
from queue import Queue, Empty

import cv2
from aiohttp import web

logger = logging.getLogger(__name__)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class WebStreamer:
    def __init__(self, port=8080, queue_size=10):
        self.port = port
        self.frame_queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.loop = None
        self.runner = None
        self.thread = None
        self.stats = {
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "info": ""
        }

    def update_stats(self, stats):
        self.stats.update(stats)

    def update_fps(self, fps):
        # Backwards compatibility
        self.stats["fps"] = fps

    def put_frame(self, frame):
        if self.stop_event.is_set():
            return
        
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        self.frame_queue.put(frame)

    async def _stats(self, request):
        return web.json_response(
            self.stats,
            headers={'Access-Control-Allow-Origin': '*'}
        )

    async def _index(self, request):
        # Simple HTML to view the stream
        content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unitree Go2 Vision</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --success: #4ade80;
        }
        body {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            width: 100%;
            padding: 1rem 2rem;
            background-color: var(--card-bg);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-sizing: border-box;
        }
        .brand {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-badge {
            background-color: rgba(74, 222, 128, 0.1);
            color: var(--success);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            width: 100%;
            max-width: 1200px;
            box-sizing: border-box;
        }
        .video-container {
            background-color: var(--card-bg);
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            width: 100%;
            max-width: 800px;
            position: relative;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 0.5rem;
            display: block;
        }
        .stats-bar {
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
            padding: 0 0.5rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        .stat-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .stat-value {
            color: var(--text-primary);
            font-weight: 600;
            font-family: monospace;
            font-size: 1.1rem;
        }
        @keyframes pulse {
            0% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
            70% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
            100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
        }
        @media (max-width: 640px) {
            header { padding: 1rem; }
            main { padding: 1rem; }
            .brand span { display: none; }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            <span>Unitree Vision</span>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            LIVE
        </div>
    </header>
    <main>
        <div class="video-container">
            <img src="/video_feed" alt="Live Stream" />
            <div class="stats-bar">
                <div class="stat-item">
                    <span>FPS:</span>
                    <span id="fps" class="stat-value">--</span>
                </div>
                <div class="stat-item">
                    <span>Resolution:</span>
                    <span class="stat-value">640x480</span>
                </div>
            </div>
        </div>
    </main>
    <script>
        setInterval(async () => {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('fps').innerText = data.fps.toFixed(1);
            } catch (e) {
                console.error(e);
            }
        }, 1000);
    </script>
</body>
</html>
"""
        return web.Response(text=content, content_type='text/html')

    async def _video_feed(self, request):
        response = web.StreamResponse()
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)

        try:
            while True:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    _, jpeg = cv2.imencode('.jpg', frame)
                    data = jpeg.tobytes()
                    
                    await response.write(b'--frame\r\n')
                    await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                    await response.write(data)
                    await response.write(b'\r\n')
                else:
                    await asyncio.sleep(0.01)
        except Exception:
            pass
        return response

    def _run_server(self):
        asyncio.set_event_loop(self.loop)
        
        app = web.Application()
        app.router.add_get('/', self._index)
        app.router.add_get('/video_feed', self._video_feed)
        app.router.add_get('/stats', self._stats)
        # Disable access log to keep console clean
        self.runner = web.AppRunner(app, access_log=None)
        
        async def start():
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            ip = get_local_ip()
            logger.info(f"Stream available at http://{ip}:{self.port}")

        self.loop.run_until_complete(start())
        self.loop.run_forever()

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=1.0)

