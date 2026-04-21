import threading
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

_latest_frame = None
_frame_lock = threading.Lock()

_running = False
_quit = False
_state_lock = threading.Lock()

_CONTROL_PAGE = b"""\
<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body { margin:0; background:#111; display:flex; flex-direction:column; align-items:center; }
  img { width:100%; max-width:640px; }
  #btn { width:90%; max-width:640px; padding:30px; margin:10px; font-size:24px;
         background:#2a2; color:#fff; border:none; border-radius:12px;
         user-select:none; -webkit-user-select:none; touch-action:manipulation; }
  #btn.active { background:#c22; }
  #quit { width:90%; max-width:640px; padding:15px; font-size:18px;
          background:#555; color:#fff; border:none; border-radius:12px; }
</style></head>
<body>
  <img src="/stream">
  <button id="btn">HOLD TO RUN</button>
  <button id="quit" onclick="fetch('/quit',{method:'POST'})">QUIT</button>
<script>
const btn = document.getElementById('btn');
function start() { btn.classList.add('active'); btn.textContent='RUNNING'; fetch('/run',{method:'POST'}); }
function stop()  { btn.classList.remove('active'); btn.textContent='HOLD TO RUN'; fetch('/stop',{method:'POST'}); }
btn.addEventListener('mousedown', start);
btn.addEventListener('mouseup', stop);
btn.addEventListener('mouseleave', stop);
btn.addEventListener('touchstart', e => { e.preventDefault(); start(); });
btn.addEventListener('touchend', e => { e.preventDefault(); stop(); });
</script>
</body></html>
"""


def update_frame(frame):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


def is_running():
    with _state_lock:
        return _running


def is_quit():
    with _state_lock:
        return _quit


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        if _latest_frame is None:
                            continue
                        _, jpeg = cv2.imencode('.jpg', _latest_frame)
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except BrokenPipeError:
                pass
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(_CONTROL_PAGE)

    def do_POST(self):
        global _running, _quit
        if self.path == '/run':
            with _state_lock:
                _running = True
        elif self.path == '/stop':
            with _state_lock:
                _running = False
        elif self.path == '/quit':
            with _state_lock:
                _quit = True
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_stream_server(host='0.0.0.0', port=8080):
    server = _ThreadingHTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Control panel at http://{host}:{port}")
