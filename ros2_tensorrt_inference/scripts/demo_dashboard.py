#!/usr/bin/env python3
"""Local read-only browser dashboard, using ROS image/results subscriptions."""
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time
import numpy as np
from PIL import Image as PillowImage
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ros2_tensorrt_inference.msg import InferenceResult

HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROS 2 · Color inference demo</title><style>
*{box-sizing:border-box}body{font:16px system-ui;background:#101924;color:#edf4fa;margin:0;padding:5vw}
main{max-width:1100px;margin:auto}header{display:flex;justify-content:space-between;align-items:center;gap:20px}
h1{font-size:clamp(25px,4vw,44px);margin:8px 0}p{color:#aebdcc;line-height:1.6}.tag{padding:10px 14px;background:#263a4b;border-radius:30px}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:24px;margin-top:30px}.card{background:#192838;padding:24px;border-radius:18px}
img{width:100%;aspect-ratio:4/3;border-radius:10px;object-fit:cover;background:#0b121b}.metrics{display:flex;flex-wrap:wrap;gap:24px;margin:25px 0}
strong{font-size:28px;display:block}small{color:#9eb1c4}.bar{height:14px;background:#0e1925;border-radius:10px;margin:8px 0 20px;overflow:hidden}.fill{height:100%;width:0;transition:width .2s}
footer{margin-top:24px;color:#9eb1c4;font-size:14px}.label{display:flex;justify-content:space-between}@media(max-width:700px){.grid{grid-template-columns:1fr}header{display:block}}
</style><main><header><div><small>LIVE ROBOTICS PIPELINE</small><h1>ROS 2 color inference</h1></div><span class="tag" id="backend">Waiting for results</span></header>
<p>Camera topic → image preprocessing → RGB mean model → ROS results.<br>This small, deterministic model scores average red, green and blue intensity. It is not a trained object detector.</p>
<div class="grid"><section class="card"><img id="camera" alt="Latest camera frame"><p id="status">Connecting to the ROS topics…</p></section>
<section class="card"><small>DOMINANT CHANNEL</small><h1 id="winner">—</h1><div id="bars"></div>
<div class="metrics"><div><strong id="fps">—</strong><small>results / second</small></div><div><strong id="latency">—</strong><small>processing ms</small></div></div>
<p id="gpu">GPU timing unavailable</p><small id="drops">No frame counters yet</small></section></div>
<footer>The browser refreshes at 5 Hz. Inference runs independently. CPU reference mode does not demonstrate TensorRT acceleration. Throughput acceptance uses benchmark.py, not this rolling display.</footer></main>
<script>
const names=['Red','Green','Blue'], colors=['#f08082','#67d8a4','#82b6ff'];
document.getElementById('bars').innerHTML=names.map((n,i)=>`<div class="label"><span>${n}</span><span id="v${i}">—</span></div><div class="bar"><div id="b${i}" class="fill" style="background:${colors[i]}"></div></div>`).join('');
async function update(){try{let r=await fetch('/state',{cache:'no-store'});let s=await r.json();
if(s.image)document.getElementById('camera').src='data:image/jpeg;base64,'+s.image;
let live=s.result_age_seconds!==null&&s.result_age_seconds<2;
document.getElementById('status').textContent=live?'Receiving live ROS results':'Waiting for fresh inference results';
document.getElementById('backend').textContent=s.backend==='tensorrt'?'TensorRT · NVIDIA GPU':s.backend==='cpu_reference'?'CPU reference · no TensorRT':'Waiting for results';
document.getElementById('fps').textContent=live?s.fps.toFixed(1):'—';
document.getElementById('latency').textContent=live?s.processing_ms.toFixed(2):'—';
document.getElementById('gpu').textContent=s.backend==='tensorrt'?`GPU pipeline: ${s.gpu_pipeline_ms.toFixed(2)} ms`:'CPU reference mode: GPU timing is not applicable';
document.getElementById('drops').textContent=`Received ${s.received_total||0} · pending frames replaced ${s.overwritten_total||0}`;
if(s.channels){let best=s.channels.indexOf(Math.max(...s.channels));document.getElementById('winner').textContent=live?names[best]:'—';s.channels.forEach((v,i)=>{document.getElementById('v'+i).textContent=(v*100).toFixed(1)+'%';document.getElementById('b'+i).style.width=Math.max(0,Math.min(100,v*100))+'%'})}
}catch(e){document.getElementById('status').textContent='Dashboard disconnected';}setTimeout(update,200)}update();
</script></html>"""

class Dashboard(Node):
    def __init__(self):
        super().__init__('demo_dashboard')
        self.declare_parameter('port', 8765)
        self.lock = threading.Lock()
        self.state = {'backend': '', 'image': None, 'processing_ms': 0, 'gpu_pipeline_ms': 0}
        self.times = deque(maxlen=240)
        self.last_image = 0.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.images = self.create_subscription(Image, '/camera/image_raw', self.image, qos)
        self.results = self.create_subscription(InferenceResult, '/inference/results', self.result, qos)
        node = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    body, kind = HTML.encode(), 'text/html; charset=utf-8'
                elif self.path == '/state':
                    with node.lock:
                        state = dict(node.state)
                        recent = [t for t in node.times if time.monotonic()-t <= 2]
                        state['fps'] = ((len(recent)-1)/(recent[-1]-recent[0])) if len(recent)>1 else 0
                        state['result_age_seconds'] = time.monotonic()-node.times[-1] if node.times else None
                    body, kind = json.dumps(state, allow_nan=False).encode(), 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *_args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', self.get_parameter('port').value), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'Open http://127.0.0.1:{self.server.server_port} in your browser')
    def image(self, msg):
        if time.monotonic()-self.last_image < 0.2:
            return
        needed = msg.step * msg.height
        if (msg.encoding not in ('rgb8','bgr8') or msg.width <= 0 or msg.height <= 0
            or msg.step < msg.width*3 or needed > len(msg.data) or needed > 6220800):
            return
        self.last_image = time.monotonic()
        pixels = np.frombuffer(msg.data, dtype=np.uint8, count=needed).reshape(msg.height,msg.step)
        pixels = pixels[:,:msg.width*3].reshape(msg.height,msg.width,3)
        if msg.encoding == 'bgr8':
            pixels = pixels[:,:,::-1]
        preview = PillowImage.fromarray(pixels)
        preview.thumbnail((640,480))
        data = BytesIO()
        preview.save(data, format='JPEG', quality=75)
        with self.lock:
            self.state['image'] = base64.b64encode(data.getvalue()).decode()
    def result(self, msg):
        channels = None
        if msg.tensors and list(msg.tensors[0].shape) == [1,3] and len(msg.tensors[0].data) == 3:
            channels = [float(x) for x in msg.tensors[0].data]
            if not np.isfinite(channels).all():
                channels = None
        with self.lock:
            self.times.append(time.monotonic())
            self.state.update(backend=msg.backend, channels=channels, processing_ms=msg.processing_ms,
                              gpu_pipeline_ms=msg.gpu_pipeline_ms, received_total=msg.received_total,
                              overwritten_total=msg.overwritten_total)
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

def main():
    rclpy.init()
    node = Dashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
