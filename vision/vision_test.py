# =============================================================
#  LASER LEGO MEOW (LLM) — playtime console  (v3: bright toy UI)
#  Same closed-loop tracking brain, brand new happy interface.
#
#  SETUP (one time):
#     pip3 install flask ultralytics opencv-python pyserial
#
#  RUN:
#     python3 prowl_console.py
#     open http://127.0.0.1:5050
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
CAMERA_INDEX = 1           # the index your finger-test proved is the USB module
FRAME_W, FRAME_H = 640, 480

TARGET_LABELS = {"person", "cat"}
LEAD_FRAMES = 8
CONF_MIN = 0.40

# --- centering-control gains (visual servoing) ---
KP_PAN = 0.03
KP_TILT = 0.025
DEADZONE = 15

# flip if the turret chases the wrong way
INVERT_PAN = False
INVERT_TILT = False

# servo limits — GEOFENCE lives here (keeps the beam low)
PAN_MIN, PAN_MAX = 10, 170
TILT_MIN, TILT_MAX = 40, 150

# pounce detection
POUNCE_HI = 22.0
POUNCE_LO = 8.0
POUNCE_COOLDOWN = 0.6

# ESP32-S3 serial
USE_SERIAL = True
SERIAL_PORT = "/dev/cu.usbmodem1101"
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
            # friendly blue box on the feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (232, 159, 74), 3)
            cv2.circle(frame, (tx, ty), 6, (232, 159, 74), -1)

        speed = 0.0
        if kf_ready:
            pred = kf.predict()
            px, py = int(pred[0][0]), int(pred[1][0])
            vx, vy = float(pred[2][0]), float(pred[3][0])
            speed = math.hypot(vx, vy)
            top_speed = max(top_speed, speed)

            lead_x = int(px + vx * LEAD_FRAMES)
            lead_y = int(py + vy * LEAD_FRAMES)

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

            lx = max(0, min(w - 1, lead_x))
            ly = max(0, min(h - 1, lead_y))
            cv2.drawMarker(frame, (cxc, cyc), (60, 213, 70),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.line(frame, (cxc, cyc), (lx, ly), (122, 92, 255), 3)
            cv2.circle(frame, (lx, ly), 9, (122, 92, 255), -1)

            now = time.time()
            if speed > POUNCE_HI and pounce_armed and (now - last_pounce) > POUNCE_COOLDOWN:
                pounces += 1
                pounce_armed = False
                last_pounce = now
            if speed < POUNCE_LO:
                pounce_armed = True

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
                stats["label"] = cur_label
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
<title>Laser Lego Meow — playtime</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#FFF6E2;
    --card:#FFFFFF;
    --sun:#FFC93C;
    --sky:#4A9FE8;
    --ink:#1E2A5A;
    --ink-soft:#7788B5;
    --coral:#FF5C7A;
    --mint:#46D5B3;
    --shadow:#1E2A5A;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{
    background:var(--paper);
    background-image:
      radial-gradient(circle at 12% 18%, rgba(255,201,60,.28) 0 12px, transparent 13px),
      radial-gradient(circle at 82% 62%, rgba(74,159,232,.22) 0 10px, transparent 11px),
      radial-gradient(circle at 46% 88%, rgba(255,92,122,.16) 0 8px, transparent 9px);
    background-size:340px 340px;
    color:var(--ink);
    font-family:"Nunito",system-ui,sans-serif;
    padding:20px 18px 34px;
  }
  .wrap{max-width:1140px;margin:0 auto;display:flex;flex-direction:column;gap:18px}

  /* ---------- chunky plastic card ---------- */
  .card{
    background:var(--card);
    border-radius:26px;
    padding:18px 20px;
    box-shadow:0 6px 0 rgba(30,42,90,.16);
  }

  /* ---------- header ---------- */
  .topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
  .brand{display:flex;align-items:center;gap:12px}
  .brand .mark{
    height:46px;padding:0 13px;border-radius:16px;background:var(--sun);
    display:grid;place-items:center;box-shadow:0 4px 0 rgba(30,42,90,.2);
    font-family:"Fredoka",sans-serif;font-weight:700;font-size:22px;
    letter-spacing:.06em;line-height:1;
  }
  .brand h1{
    font-family:"Fredoka",sans-serif;font-weight:700;font-size:27px;
    letter-spacing:.01em;line-height:1;
  }
  .brand .sub{font-size:13px;color:var(--ink-soft);font-weight:700;margin-top:2px}

  .statuschip{
    display:flex;align-items:center;gap:12px;
    background:var(--sun);border-radius:20px;padding:9px 18px 9px 12px;
    box-shadow:0 4px 0 rgba(30,42,90,.2);
    font-family:"Fredoka",sans-serif;font-weight:600;font-size:17px;
    transition:background .3s;
  }
  .locked .statuschip{background:var(--mint)}

  /* ---------- the cat face (signature) ---------- */
  .catface{width:40px;height:40px;flex:none}
  .catface .eye{transition:all .25s ease}
  .catface .pupil{opacity:0;transition:opacity .25s ease}
  .locked .catface .eye-lid{opacity:0}
  .locked .catface .pupil{opacity:1}
  .catface .head{transform-origin:20px 24px}
  .locked .catface .head{animation:perk .5s ease}
  @keyframes perk{0%{transform:scale(1)}45%{transform:scale(1.12)}100%{transform:scale(1)}}

  /* ---------- layout ---------- */
  .grid{display:grid;grid-template-columns:1.55fr 1fr;gap:18px;align-items:start}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}

  /* ---------- video porthole ---------- */
  .porthole{
    background:var(--sun);border-radius:32px;padding:14px;
    box-shadow:0 6px 0 rgba(30,42,90,.18);position:relative;
    transition:background .3s;
  }
  .locked .porthole{background:var(--mint)}
  .porthole .screen{
    border-radius:22px;overflow:hidden;background:#0d1730;
    aspect-ratio:4/3;position:relative;
  }
  .porthole img{width:100%;height:100%;object-fit:cover;display:block}
  .tape{
    position:absolute;top:-9px;left:50%;transform:translateX(-50%) rotate(-2deg);
    background:var(--card);color:var(--ink);font-family:"Fredoka",sans-serif;
    font-weight:600;font-size:13px;padding:5px 18px;border-radius:12px;
    box-shadow:0 3px 0 rgba(30,42,90,.16);
  }
  .seen{
    position:absolute;left:14px;bottom:14px;
    background:rgba(255,255,255,.94);border-radius:14px;padding:7px 14px;
    font-family:"Fredoka",sans-serif;font-weight:600;font-size:14px;
    box-shadow:0 3px 0 rgba(30,42,90,.14);
  }

  /* ---------- right rail ---------- */
  .rail{display:flex;flex-direction:column;gap:18px}
  .card h2{
    font-family:"Fredoka",sans-serif;font-size:15px;font-weight:600;
    color:var(--ink);margin-bottom:14px;display:flex;align-items:center;gap:8px;
  }
  .card h2 span{font-size:18px}

  .aim{margin-bottom:14px}
  .aim:last-child{margin-bottom:0}
  .aim .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px}
  .aim .name{font-size:13px;font-weight:800;color:var(--ink-soft);letter-spacing:.04em}
  .aim .val{font-family:"Fredoka",sans-serif;font-size:20px;font-weight:600}
  .bar{height:14px;border-radius:99px;background:#EDF1FA;overflow:hidden}
  .bar i{display:block;height:100%;border-radius:99px;width:50%;transition:width .14s linear}
  .bar.pan i{background:var(--sun)}
  .bar.tilt i{background:var(--sky)}

  .scores{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .score{
    border-radius:20px;padding:14px;background:#F7F9FF;text-align:center;
  }
  .score.big{grid-column:1/-1;background:#FFF3D0}
  .score .k{font-size:11px;font-weight:800;letter-spacing:.1em;
    text-transform:uppercase;color:var(--ink-soft)}
  .score .v{font-family:"Fredoka",sans-serif;font-weight:700;font-size:34px;line-height:1.15;margin-top:2px}
  .score.big .v{font-size:46px;color:var(--coral)}
  .score .v small{font-size:15px;color:var(--ink-soft);font-weight:500}
  .pop{animation:pop .45s ease}
  @keyframes pop{0%{transform:scale(1)}40%{transform:scale(1.25)}100%{transform:scale(1)}}

  .paws{display:flex;justify-content:center;gap:6px;margin-top:8px}
  .paws svg{width:17px;height:17px;opacity:.18;transition:opacity .3s, transform .3s}
  .paws svg.on{opacity:1;transform:scale(1.12)}

  /* ---------- controls ---------- */
  .controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}
  .btn{
    font-family:"Fredoka",sans-serif;font-weight:600;font-size:15px;
    color:var(--ink);background:var(--card);border:none;
    padding:13px 22px;border-radius:18px;cursor:pointer;
    box-shadow:0 5px 0 rgba(30,42,90,.2);
    transition:transform .08s, box-shadow .08s, background .2s;
  }
  .btn:hover{background:#FFFBF0}
  .btn:active{transform:translateY(4px);box-shadow:0 1px 0 rgba(30,42,90,.2)}
  .btn:focus-visible{outline:3px solid var(--sky);outline-offset:3px}
  .btn.play{background:var(--sun)}
  .btn.stop{background:var(--coral);color:#fff}

  .dial{
    display:flex;align-items:center;gap:12px;margin-left:auto;
    background:#F7F9FF;border-radius:18px;padding:10px 16px;
  }
  .dial label{font-size:12px;font-weight:800;letter-spacing:.08em;
    text-transform:uppercase;color:var(--ink-soft)}
  .dial input{accent-color:var(--coral);width:140px}
  .dial b{font-family:"Fredoka",sans-serif;font-size:19px;font-weight:600;min-width:26px;text-align:center}

  /* ---------- play-by-play ---------- */
  .log{height:132px;overflow:hidden;display:flex;flex-direction:column-reverse;gap:6px}
  .log .line{
    font-size:14px;font-weight:600;color:var(--ink);
    background:#F7F9FF;border-radius:14px;padding:7px 14px;
    display:flex;align-items:center;gap:9px;
    animation:slidein .3s ease;
  }
  .log .line .t{font-size:11px;color:var(--ink-soft);font-weight:800;margin-left:auto}
  .log .line.win{background:#E4FAF3}
  .log .line.pounce{background:#FFE7EC}
  @keyframes slidein{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}

  @media (prefers-reduced-motion: reduce){
    *{animation:none!important;transition:none!important}
  }
</style>
</head>
<body>
<div class="wrap" id="root">

  <!-- header -->
  <div class="card topbar">
    <div class="brand">
      <div class="mark">LLM</div>
      <div>
        <h1>Laser Lego Meow</h1>
        <div class="sub">the laser that thinks one step ahead</div>
      </div>
    </div>

    <div class="statuschip">
      <svg class="catface" viewBox="0 0 40 40" fill="none">
        <g class="head">
          <path d="M8 16 L7 6 L15 11 Z" fill="#1E2A5A"/>
          <path d="M32 16 L33 6 L25 11 Z" fill="#1E2A5A"/>
          <circle cx="20" cy="23" r="13" fill="#1E2A5A"/>
          <g class="eye">
            <circle class="pupil" cx="15" cy="21" r="3" fill="#FFF6E2"/>
            <circle class="pupil" cx="25" cy="21" r="3" fill="#FFF6E2"/>
            <path class="eye-lid" d="M12 21h6M22 21h6" stroke="#FFF6E2" stroke-width="2.2" stroke-linecap="round"/>
          </g>
          <path d="M18.5 27h3l-1.5 1.6z" fill="#FF9FB2"/>
          <path d="M6 24h5M6 27h5M29 24h5M29 27h5" stroke="#FFF6E2" stroke-width="1.1" stroke-linecap="round" opacity=".55"/>
        </g>
      </svg>
      <span id="status">Waking up…</span>
    </div>
  </div>

  <!-- main -->
  <div class="grid">
    <div class="porthole">
      <div class="tape">live from the turret</div>
      <div class="screen">
        <img src="/video_feed" alt="live camera view from the turret">
        <div class="seen" id="seen">Nobody around yet</div>
      </div>
    </div>

    <div class="rail">
      <div class="card">
        <h2><span>🎯</span> Where it's pointing</h2>
        <div class="aim">
          <div class="row"><span class="name">LEFT / RIGHT</span><span class="val"><span id="pan">90</span>°</span></div>
          <div class="bar pan"><i id="panbar"></i></div>
        </div>
        <div class="aim">
          <div class="row"><span class="name">UP / DOWN</span><span class="val"><span id="tilt">90</span>°</span></div>
          <div class="bar tilt"><i id="tiltbar"></i></div>
        </div>
      </div>

      <div class="card">
        <h2><span>🏆</span> Today's playtime</h2>
        <div class="scores">
          <div class="score big">
            <div class="k">Pounces</div>
            <div class="v" id="pounces">0</div>
            <div class="paws" id="paws"></div>
          </div>
          <div class="score">
            <div class="k">Interest</div>
            <div class="v"><span id="obsession">0</span><small>%</small></div>
          </div>
          <div class="score">
            <div class="k">Played</div>
            <div class="v" id="session">0<small>s</small></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- controls -->
  <div class="card controls">
    <button class="btn play" id="pauseBtn">Pause the game</button>
    <button class="btn" id="resetBtn">New session</button>
    <button class="btn stop" id="killBtn">Stop the laser</button>
    <div class="dial">
      <label for="leadSlider">How far ahead</label>
      <input type="range" id="leadSlider" min="0" max="30" value="8">
      <b id="leadVal">8</b>
    </div>
  </div>

  <!-- play by play -->
  <div class="card">
    <h2><span>📣</span> Play-by-play</h2>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
  const root = document.getElementById('root');
  const $ = id => document.getElementById(id);
  let prevStatus = null, prevPounces = 0, paused = false;

  // paw row
  const PAWS = 8;
  const pawSvg = '<svg viewBox="0 0 24 24" fill="none">' +
    '<ellipse cx="6.5" cy="9" rx="2.1" ry="2.8" fill="#FF5C7A"/>' +
    '<ellipse cx="12" cy="7" rx="2.1" ry="2.9" fill="#FF5C7A"/>' +
    '<ellipse cx="17.5" cy="9" rx="2.1" ry="2.8" fill="#FF5C7A"/>' +
    '<path d="M12 12c3.4 0 6 2.3 6 4.8 0 2-1.7 3.2-3.6 3.2-1 0-1.7-.4-2.4-.4s-1.4.4-2.4.4C7.7 20 6 18.8 6 16.8 6 14.3 8.6 12 12 12z" fill="#FF5C7A"/></svg>';
  $('paws').innerHTML = pawSvg.repeat(PAWS);

  function stamp(){ return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }
  function logLine(text, cls){
    const el = document.createElement('div');
    el.className = 'line ' + (cls||'');
    el.innerHTML = text + '<span class="t">' + stamp() + '</span>';
    const log = $('log');
    log.prepend(el);
    while(log.children.length > 24) log.removeChild(log.lastChild);
  }

  async function poll(){
    try{
      const r = await fetch('/stats');
      const s = await r.json();

      const isLocked = s.status === 'LOCKED';
      root.classList.toggle('locked', isLocked);
      $('status').textContent = isLocked ? 'Got you!' : 'Looking for kitty…';
      $('seen').textContent = isLocked
        ? (s.label === 'cat' ? 'Kitty spotted · ' : 'Human spotted · ') + Math.round(s.conf*100) + '% sure'
        : 'Nobody around yet';

      $('pan').textContent = s.pan;
      $('tilt').textContent = s.tilt;
      $('panbar').style.width  = (s.pan/180*100) + '%';
      $('tiltbar').style.width = (s.tilt/180*100) + '%';

      $('obsession').textContent = s.obsession;
      $('session').innerHTML = s.session + '<small>s</small>';

      const pEl = $('pounces');
      if(s.pounces !== prevPounces){
        pEl.textContent = s.pounces;
        pEl.classList.remove('pop'); void pEl.offsetWidth; pEl.classList.add('pop');
        const paws = $('paws').children;
        for(let i=0;i<PAWS;i++) paws[i].classList.toggle('on', i < (s.pounces % (PAWS+1)));
        if(s.pounces > prevPounces) logLine('Pounce! Nice try, kitty', 'pounce');
        prevPounces = s.pounces;
      }

      if(s.status !== prevStatus){
        if(isLocked) logLine('Found ' + (s.label === 'cat' ? 'the cat' : 'someone') + ' — game on', 'win');
        else if(prevStatus === 'LOCKED') logLine('Lost them, having a look around');
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
    paused = !paused;
    send(paused ? 'pause' : 'resume');
    $('pauseBtn').textContent = paused ? 'Back to playing' : 'Pause the game';
    logLine(paused ? 'Paused — turret is holding still' : 'Playing again');
  };
  $('resetBtn').onclick = () => { send('reset'); prevPounces = 0; logLine('Fresh session, scores back to zero'); };
  $('killBtn').onclick = () => {
    send('pause'); paused = true;
    $('pauseBtn').textContent = 'Back to playing';
    logLine('Laser stopped — everything is safe');
  };
  const slider = $('leadSlider');
  slider.oninput = () => { $('leadVal').textContent = slider.value; send('lead', parseInt(slider.value)); };

  logLine('Turret is awake and looking around', 'win');
</script>
</body>
</html>"""


if __name__ == "__main__":
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    print("\n>>> Laser Lego Meow (LLM) console running.")
    print(">>>   On this Mac:        http://127.0.0.1:5050")
    print(">>>   On your WiFi:       http://<this-mac's-IP>:5050")
    print(">>>   (find the IP with:  ipconfig getifaddr en0 )\n")
    app.run(host="0.0.0.0", port=5050, threaded=True)