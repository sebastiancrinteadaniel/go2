import asyncio
import logging
import os
import signal

logger = logging.getLogger(__name__)

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.log_queues = set()

    async def start_process(self, name, command):
        if name in self.processes:
            return False, "Process already running"
        
        logger.info(f"Starting process {name}: {command}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                preexec_fn=os.setsid # Create new process group so we can kill the whole tree
            )
            self.processes[name] = process
            
            # Start reading logs
            asyncio.create_task(self._read_logs(name, process))
            return True, "Started"
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False, str(e)

    async def stop_process(self, name):
        if name not in self.processes:
            return False, "Process not found"
        
        process = self.processes[name]
        
        try:
            # Kill the process group to ensure child processes (like uv's python) are killed
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass # Already dead
            
        if name in self.processes:
            del self.processes[name]
            
        return True, "Stopped"

    async def stop_all(self):
        """Stop all running processes."""
        names = list(self.processes.keys())
        for name in names:
            await self.stop_process(name)

    async def _read_logs(self, name, process):
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                if decoded:
                    msg = f"[{name}] {decoded}"
                    # Broadcast to all connected queues
                    # We iterate over a copy to avoid modification during iteration issues
                    for q in list(self.log_queues):
                        try:
                            q.put_nowait(msg)
                        except asyncio.QueueFull:
                            pass
        except Exception as e:
            logger.error(f"Error reading logs for {name}: {e}")
        finally:
            # Process ended
            if name in self.processes:
                del self.processes[name]
                msg = f"[{name}] Process exited."
                for q in list(self.log_queues):
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass

    def add_log_queue(self, queue):
        self.log_queues.add(queue)

    def remove_log_queue(self, queue):
        self.log_queues.discard(queue)

    def get_status(self):
        return {name: "running" for name in self.processes.keys()}

    def cleanup_orphans(self):
        """Kill any lingering processes matching our modules."""
        import psutil
        
        # List of module scripts to look for
        targets = [
            "src.yolo.webrtc",
            "src.hand_detection.webrtc",
            "src.depth_camera.webrtc",
            "src.simple_camera.webrtc"
        ]
        
        count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 2:
                    # Check if it matches "python3 -m src.xxx.webrtc"
                    if "python" in proc.info['name'] and any(t in cmd for t in targets for cmd in cmdline):
                        logger.warning(f"Killing orphan process {proc.pid}: {cmdline}")
                        proc.kill()
                        count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if count > 0:
            logger.info(f"Cleaned up {count} orphan processes.")
