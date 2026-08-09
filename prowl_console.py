# =============================================================
#  PROWL — camera-tracking laser turret console  (v2: closed loop)
#  Camera now rides ON the gimbal, so this uses CENTERING CONTROL:
#  it nudges the servos to keep the target in the middle of the frame.
#
#  SETUP (one time):
#     pip3 install flask ultralytics opencv-python pyserial
#
#  RUN:
#     python3 prowl_console.py
#     open the printed URL (http://127.0.0.1:5050) in your browser.
#  Quit: Ctrl-C in the terminal.
# =============================================================

import time
import math
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

# ------------------- CONFIG -------------------
CAMERA_INDEX = 0           # 1 = external USB module (0 = built-in webcam)
FRAME_W, FRAME_H = 640, 480

TARGET_LABELS = {"person", "cat"}
LEAD_FRAMES = 8            # how far ahead to aim (live-tunable in the UI)
CONF_MIN = 0.40

# --- centering-control gains (the visual-servoing loop) ---
KP_PAN = 0.03              # deg of pan per pixel of horizontal error
KP_TILT = 0.025            # deg of tilt per pixel of vertical error
DEADZONE = 15              # px: ignore tiny errors so it doesn't jitter

# if the turret chases the WRONG way, flip the matching one to True
INVERT_PAN = False
INVERT_TILT = False

# servo limits — the GEOFENCE lives here. keep tilt off the extremes so the
# beam can't point up toward eye level once the laser is attached.
PAN_MIN, PAN_MAX = 10, 170
TILT_MIN, TILT_MAX = 40, 150

# pounce detection (a fast lunge in pixels/frame)
POUNCE_HI = 22.0
POUNCE_LO = 8.0
POUNCE_COOLDOWN = 0.6

# ESP32 serial
USE_SERIAL = True
SERIAL_PORT = "/dev/cu.usbserial-10"   # confirm with: ls /dev/cu.*
BAUD = 115200
# ----------------------------------------------

app = Flask(__name__)

state_lock = threading.Lock()
latest_jpeg = None
paused = False

stats = {
    "status": "BOOTING", "label": "--", "conf": 0.0,
    "pan": 90, "tilt": 90, "fps": 0.0, "session": 0,
    "pounces": 0, "obsession": 0.0, "speed": 0.0,
    "top_speed": 0.0, "lead": LEAD_FRAMES, "serial": "off",
}

ser = None
if USE_SERIAL:
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        time.sleep(2.0)
        stats["serial"] = "linked"
        print(">>> serial linked")
    except Exception as e:
        print(f">>> serial off ({e})")
        ser = None
        stats["serial"] = "error"


def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf


def clampf(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def vision_loop():
    global latest_jpeg, LEAD_FRAMES

    model = YOLO("yolov8n.pt")
    kf = make_kalman()
    kf_ready = False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    # running servo position — this is what centering control updates
    pan_pos = 90.0
    tilt_pos = 90.0

    session_start = time.time()
    frames_total = 0
    frames_locked = 0
    pounces = 0
    top_speed = 0.0
    pounce_armed = True
    last_pounce = 0.0

    fps_t = time.time()
    fps_val = 0.0
    last_send = 0.0

    with state_lock:
        stats["status"] = "SEARCHING"

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        cxc, cyc = w // 2, h // 2
        frames_total += 1

        results = model(frame, verbose=False)[0]

        best = None
        best_conf = 0.0
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label in TARGET_LABELS and conf >= CONF_MIN and conf > best_conf:
                best_conf = conf
                best = (box, label)

        locked = best is not None
        cur_label = "--"
        cur_conf = 0.0

        if locked:
            box, cur_label = best
            cur_conf = best_conf
            frames_locked += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
            measurement = np.array([[np.float32(tx)], [np.float32(ty)]])
            if not kf_ready:
                kf.statePost = np.array([[tx], [ty], [0], [0]], np.float32)
                kf_ready = True
            kf.correct(measurement)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 210), 2)
            cv2.circle(frame, (tx, ty), 5, (80, 255, 210), -1)

        speed = 0.0
        if kf_ready:
            pred = kf.predict()
            px, py = int(pred[0][0]), int(pred[1][0])
            vx, vy = float(pred[2][0]), float(pred[3][0])
            speed = math.hypot(vx, vy)
            top_speed = max(top_speed, speed)

            # lead point = where the target is heading
            lead_x = int(px + vx * LEAD_FRAMES)
            lead_y = int(py + vy * LEAD_FRAMES)

            # ---- CENTERING CONTROL: drive lead point toward frame center ----
            err_x = lead_x - cxc
            err_y = lead_y - cyc

            if abs(err_x) > DEADZONE:
                d_pan = KP_PAN * err_x
                pan_pos += (-d_pan if INVERT_PAN else d_pan)
            if abs(err_y) > DEADZONE:
                d_tilt = KP_TILT * err_y
                tilt_pos += (-d_tilt if INVERT_TILT else d_tilt)

            pan_pos = clampf(pan_pos, PAN_MIN, PAN_MAX)
            tilt_pos = clampf(tilt_pos, TILT_MIN, TILT_MAX)

            # visuals: center reticle (goal) + lead dot (target) + error line
            lx = max(0, min(w - 1, lead_x))
            ly = max(0, min(h - 1, lead_y))
            cv2.drawMarker(frame, (cxc, cyc), (120, 160, 255),
                           cv2.MARKER_CROSS, 26, 1)
            cv2.line(frame, (cxc, cyc), (lx, ly), (92, 59, 255), 2)
            cv2.circle(frame, (lx, ly), 8, (92, 59, 255), -1)

            # pounce detection
            now = time.time()
            if speed > POUNCE_HI and pounce_armed and (now - last_pounce) > POUNCE_COOLDOWN:
                pounces += 1
                pounce_armed = False
                last_pounce = now
            if speed < POUNCE_LO:
                pounce_armed = True

            # send the servo command
            if ser is not None and not paused and (now - last_send) > 0.05:
                try:
                    ser.write(f"{int(pan_pos)},{int(tilt_pos)}\n".encode())
                except Exception:
                    pass
                last_send = now

        dt = time.time() - fps_t
        fps_t = time.time()
        if dt > 0:
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt)

        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with state_lock:
                latest_jpeg = buf.tobytes()
                stats["status"] = "LOCKED" if locked else "SEARCHING"
                stats["label"] = cur_label.upper()
                stats["conf"] = round(cur_conf, 2)
                stats["pan"] = int(pan_pos)
                stats["tilt"] = int(tilt_pos)
                stats["fps"] = round(fps_val, 1)
                stats["session"] = int(time.time() - session_start)
                stats["pounces"] = pounces
                stats["obsession"] = round(100.0 * frames_locked / max(1, frames_total), 1)
                stats["speed"] = round(speed, 1)
                stats["top_speed"] = round(top_speed, 1)
                stats["lead"] = LEAD_FRAMES
                if stats.get("_reset"):
                    session_start = time.time()
                    frames_total = 1
                    frames_locked = 1 if locked else 0
                    pounces = 0
                    top_speed = 0.0
                    stats["_reset"] = False


def mjpeg():
    while True:
        with state_lock:
            buf = latest_jpeg
        if buf is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return PAGE


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def get_stats():
    with state_lock:
        return jsonify(dict(stats))


@app.route("/control", methods=["POST"])
def control():
    global paused, LEAD_FRAMES
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "pause":
        paused = True
    elif action == "resume":
        paused = False
    elif action == "reset":
        with state_lock:
            stats["_reset"] = True
    elif action == "lead":
        try:
            LEAD_FRAMES = max(0, min(40, int(data.get("value", LEAD_FRAMES))))
        except Exception:
            pass
    return jsonify({"ok": True, "paused": paused, "lead": LEAD_FRAMES})


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PROWL // targeting console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --void:#0a0e1a; --panel:#121a2e; --panel2:#0f1626; --edge:#24304d;
    --laser:#ff3b5c; --phosphor:#ffc24b; --lock:#4fcdb0;
    --muted:#5f6f92; --ink:#dfe7ff; --scope:#0b111f;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{
    background:
      radial-gradient(1200px 600px at 70% -10%, #16223f 0%, transparent 60%),
      radial-gradient(900px 500px at -10% 110%, #1a1330 0%, transparent 55%),
      var(--void);
    color:var(--ink); font-family:"Chakra Petch",sans-serif;
    padding:18px; min-height:100%; letter-spacing:.02em;
  }
  .wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
  .topbar{display:flex;align-items:center;justify-content:space-between;
    border:1px solid var(--edge);border-radius:12px;padding:12px 18px;
    background:linear-gradient(180deg,var(--panel),var(--panel2))}
  .brand{display:flex;align-items:baseline;gap:12px}
  .brand h1{font-weight:700;font-size:22px;letter-spacing:.28em}
  .brand .tag{color:var(--muted);font-size:11px;letter-spacing:.32em;text-transform:uppercase}
  .status-pill{display:flex;align-items:center;gap:10px;font-family:"JetBrains Mono",monospace;
    font-size:13px;letter-spacing:.18em;padding:7px 14px;border-radius:999px;
    border:1px solid var(--edge);background:#0c1424}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--phosphor);
    box-shadow:0 0 12px var(--phosphor);animation:blink 1.4s infinite}
  .locked .dot{background:var(--lock);box-shadow:0 0 14px var(--lock);animation:none}
  @keyframes blink{50%{opacity:.35}}
  .grid{display:grid;grid-template-columns:1.6fr 1fr;gap:14px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .scope{position:relative;border-radius:14px;overflow:hidden;background:var(--scope);
    border:1px solid var(--edge);aspect-ratio:4/3;min-height:0}
  .scope img{width:100%;height:100%;object-fit:cover;display:block;filter:saturate(1.05) contrast(1.03)}
  .scope::after{content:"";position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(120% 90% at 50% 50%, transparent 60%, rgba(0,0,0,.55) 100%);mix-blend-mode:multiply}
  .bracket{position:absolute;width:34px;height:34px;border:2px solid var(--phosphor);
    opacity:.9;transition:border-color .25s, box-shadow .25s}
  .bracket.tl{top:12px;left:12px;border-right:0;border-bottom:0}
  .bracket.tr{top:12px;right:12px;border-left:0;border-bottom:0}
  .bracket.bl{bottom:12px;left:12px;border-right:0;border-top:0}
  .bracket.br{bottom:12px;right:12px;border-left:0;border-top:0}
  .locked .bracket{border-color:var(--lock);box-shadow:0 0 14px rgba(79,205,176,.5)}
  .scanline{position:absolute;left:0;right:0;height:2px;top:0;
    background:linear-gradient(90deg,transparent,var(--phosphor),transparent);
    opacity:.55;animation:scan 2.6s linear infinite}
  .locked .scanline{display:none}
  @keyframes scan{0%{top:4%}100%{top:96%}}
  .scopetag{position:absolute;left:14px;bottom:14px;font-family:"JetBrains Mono",monospace;
    font-size:12px;letter-spacing:.14em;color:var(--phosphor);
    background:rgba(8,12,22,.6);padding:4px 9px;border-radius:6px;border:1px solid var(--edge)}
  .locked .scopetag{color:var(--lock)}
  .rail{display:flex;flex-direction:column;gap:12px}
  .card{border:1px solid var(--edge);border-radius:12px;padding:14px 16px;
    background:linear-gradient(180deg,var(--panel),var(--panel2))}
  .card h2{font-size:11px;letter-spacing:.26em;color:var(--muted);text-transform:uppercase;font-weight:600;margin-bottom:12px}
  .axis{margin-bottom:12px}.axis:last-child{margin-bottom:0}
  .axis .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
  .axis .name{font-size:12px;letter-spacing:.2em;color:var(--muted)}
  .axis .val{font-family:"JetBrains Mono",monospace;font-size:18px;color:var(--phosphor)}
  .meter{height:6px;border-radius:3px;background:#0b1322;overflow:hidden;border:1px solid #1b2540}
  .meter i{display:block;height:100%;width:50%;
    background:linear-gradient(90deg,#ff9a3b,var(--phosphor));transition:width .12s linear}
  .bignums{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .stat{border:1px solid var(--edge);border-radius:10px;padding:12px;background:#0c1424}
  .stat .k{font-size:10px;letter-spacing:.24em;color:var(--muted);text-transform:uppercase}
  .stat .v{font-family:"JetBrains Mono",monospace;font-size:26px;margin-top:6px;color:var(--ink)}
  .stat .v small{font-size:12px;color:var(--muted)}
  .stat.hot .v{color:var(--laser)}
  .pop{animation:pop .4s ease}
  @keyframes pop{0%{transform:scale(1)}35%{transform:scale(1.22);color:var(--laser)}100%{transform:scale(1)}}
  .controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}
  .btn{font-family:"Chakra Petch";font-weight:600;letter-spacing:.16em;font-size:12px;
    text-transform:uppercase;color:var(--ink);background:#101a30;border:1px solid var(--edge);
    padding:10px 16px;border-radius:9px;cursor:pointer;transition:.15s}
  .btn:hover{border-color:var(--phosphor);color:var(--phosphor)}
  .btn.kill{border-color:#4a2030;color:var(--laser)}
  .btn.kill:hover{background:var(--laser);color:#0a0e1a;border-color:var(--laser)}
  .lead{display:flex;align-items:center;gap:12px;margin-left:auto;
    font-size:11px;letter-spacing:.2em;color:var(--muted);text-transform:uppercase}
  .lead input{accent-color:var(--laser);width:150px}
  .lead b{font-family:"JetBrains Mono",monospace;color:var(--phosphor);font-size:16px;min-width:26px}
  .log{border:1px solid var(--edge);border-radius:12px;background:var(--panel2);
    padding:12px 16px;font-family:"JetBrains Mono",monospace;font-size:12px;
    height:118px;overflow:hidden;display:flex;flex-direction:column-reverse}
  .log .line{padding:2px 0;color:#9fb0d4;opacity:.9}
  .log .line .t{color:var(--muted)}
  .log .line.lock{color:var(--lock)}.log .line.pounce{color:var(--laser)}.log .line.lost{color:var(--phosphor)}
  @media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="wrap" id="root">
  <div class="topbar">
    <div class="brand"><h1>PROWL</h1><span class="tag">autonomous prey engagement system</span></div>
    <div class="status-pill"><span class="dot"></span><span id="status">SEARCHING</span></div>
  </div>
  <div class="grid">
    <div class="scope" id="scope">
      <img src="/video_feed" alt="live feed">
      <div class="scanline"></div>
      <div class="bracket tl"></div><div class="bracket tr"></div>
      <div class="bracket bl"></div><div class="bracket br"></div>
      <div class="scopetag" id="scopetag">NO TARGET</div>
    </div>
    <div class="rail">
      <div class="card">
        <h2>Turret bearing</h2>
        <div class="axis">
          <div class="row"><span class="name">PAN</span><span class="val"><span id="pan">90</span>&deg;</span></div>
          <div class="meter"><i id="panbar"></i></div>
        </div>
        <div class="axis">
          <div class="row"><span class="name">TILT</span><span class="val"><span id="tilt">90</span>&deg;</span></div>
          <div class="meter"><i id="tiltbar"></i></div>
        </div>
      </div>
      <div class="card">
        <h2>Hunt telemetry</h2>
        <div class="bignums">
          <div class="stat hot"><div class="k">Pounces</div><div class="v" id="pounces">0</div></div>
          <div class="stat"><div class="k">Obsession</div><div class="v"><span id="obsession">0</span><small>%</small></div></div>
          <div class="stat"><div class="k">Top speed</div><div class="v" id="topspeed">0</div></div>
          <div class="stat"><div class="k">Session</div><div class="v" id="session">0<small>s</small></div></div>
        </div>
      </div>
    </div>
  </div>
  <div class="controls card">
    <button class="btn" id="pauseBtn">Pause tracking</button>
    <button class="btn" id="resetBtn">Reset session</button>
    <button class="btn kill" id="killBtn">Laser kill</button>
    <label class="lead">Lead <input type="range" id="leadSlider" min="0" max="30" value="8"> <b id="leadVal">8</b></label>
  </div>
  <div class="log" id="log"></div>
</div>
<script>
  const root = document.getElementById('root');
  const $ = id => document.getElementById(id);
  let prevStatus = null, prevPounces = 0, paused = false;
  function stamp(){ return new Date().toLocaleTimeString([], {hour12:false}); }
  function logLine(text, cls){
    const el = document.createElement('div');
    el.className = 'line ' + (cls||'');
    el.innerHTML = '<span class="t">'+stamp()+'</span> &rsaquo; ' + text;
    const log = $('log'); log.prepend(el);
    while(log.children.length > 40) log.removeChild(log.lastChild);
  }
  async function poll(){
    try{
      const r = await fetch('/stats'); const s = await r.json();
      root.classList.toggle('locked', s.status === 'LOCKED');
      $('status').textContent = s.status;
      $('scopetag').textContent = s.status === 'LOCKED'
        ? (s.label + '  ' + Math.round(s.conf*100) + '%') : 'NO TARGET';
      $('pan').textContent = s.pan; $('tilt').textContent = s.tilt;
      $('panbar').style.width  = (s.pan/180*100) + '%';
      $('tiltbar').style.width = (s.tilt/180*100) + '%';
      $('obsession').textContent = s.obsession;
      $('topspeed').textContent = s.top_speed;
      $('session').innerHTML = s.session + '<small>s</small>';
      const pEl = $('pounces');
      if(s.pounces !== prevPounces){
        pEl.textContent = s.pounces;
        pEl.classList.remove('pop'); void pEl.offsetWidth; pEl.classList.add('pop');
        if(s.pounces > prevPounces) logLine('POUNCE DETECTED', 'pounce');
        prevPounces = s.pounces;
      }
      if(s.status !== prevStatus){
        if(s.status === 'LOCKED') logLine('TARGET LOCKED &mdash; ' + s.label, 'lock');
        else if(prevStatus === 'LOCKED') logLine('PREY LOST &mdash; reacquiring', 'lost');
        prevStatus = s.status;
      }
      $('leadVal').textContent = s.lead;
    }catch(e){}
  }
  setInterval(poll, 150);
  async function send(action, value){
    await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action, value})});
  }
  $('pauseBtn').onclick = () => {
    paused = !paused; send(paused ? 'pause' : 'resume');
    $('pauseBtn').textContent = paused ? 'Resume tracking' : 'Pause tracking';
    logLine(paused ? 'TRACKING PAUSED' : 'TRACKING RESUMED');
  };
  $('resetBtn').onclick = () => { send('reset'); logLine('SESSION RESET'); prevPounces = 0; };
  $('killBtn').onclick = () => { send('pause'); paused = true;
    $('pauseBtn').textContent = 'Resume tracking'; logLine('LASER KILL &mdash; system safed', 'lost'); };
  const slider = $('leadSlider');
  slider.oninput = () => { $('leadVal').textContent = slider.value; send('lead', parseInt(slider.value)); };
  logLine('SYSTEM ONLINE &mdash; scanning for prey', 'lock');
</script>
</body>
</html>"""


if __name__ == "__main__":
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    print("\n>>> PROWL console running.  Open this in your browser:")
    print(">>>   http://127.0.0.1:5050\n")
    app.run(host="0.0.0.0", port=5050, threaded=True)