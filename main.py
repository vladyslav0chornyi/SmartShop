import asyncio
import cv2
import numpy as np
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from camera_handler import CameraHandler
from api_client import APIClient
from ml.phrase_selector import PhraseSelector
from tts.tts_player import TTSPlayer
import uvicorn
import logging

logging.basicConfig(filename="main.log", level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("Main")

app = FastAPI()

# === Конфігурація та глобальні компоненти ===
cam = CameraHandler(
    camera_sources=[0, 'rtsp://cam1_ip/stream'],
    default_fps=10,
    min_fps=2,
    max_fps=25,
    config_api_url="http://localhost:5000/camera-config",
    diagnostics_api_url="http://localhost:5000/camera-diagnostics"
)
api = APIClient(
    endpoints={
        "frame": "http://localhost:5000/upload-frame",
        "stats": "http://localhost:5000/upload-stats"
    },
    auth_type="Bearer",
    token="your_api_token_here",
    batch_size=5,
    proxy=None
)
phrase_selector = PhraseSelector(
    phrase_sync_url="http://localhost:5000/phrases",
    sync_interval=300
)
tts = TTSPlayer(lang="uk", rate=160, volume=1.0)

# === Глобальні змінні для веб інтерфейсу ===
latest_frame = None
latest_phrase = ""
system_status = "running"
config = {
    "phrase_selector": {},
    "tts": {},
    "camera": {},
}

# === Обробка кадрів ===
def frame_to_jpeg_bytes(frame):
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buf.tobytes()

def analyze_frame(frame):
    # Демонстраційна аналітика: розпізнавання статі, кольору одягу, емоції (може бути замінено ML)
    # Тут можна інтегрувати реальний ML-детектор, а поки — stub
    gender = "male" if np.mean(frame) % 2 > 1 else "female"
    color = "blue" if frame[0,0,0] > 100 else "red"
    emotion = "happy" if np.median(frame) > 128 else "neutral"
    return {"gender": gender, "color": color, "emotion": emotion}

def handle_frame(frame, meta):
    global latest_frame, latest_phrase
    latest_frame = frame.copy()
    features = analyze_frame(frame)
    features.update({
        "group": "greeting",
        "accessories": [],
        "clothes": None
    })
    # Ознака часу дня для phrase_selector
    context = {"timeofday": meta.get("timeofday", "day")}
    # Вибір фрази
    phrase = phrase_selector.select(features, context)
    latest_phrase = phrase
    # Озвучування
    tts.play(phrase, blocking=False)
    # Відправка кадру на сервер
    frame_bytes = frame_to_jpeg_bytes(frame)
    if frame_bytes:
        api.add_to_batch(frame_bytes, meta)
    logger.info(f"Frame handled: phrase='{phrase}', gender={features['gender']}, color={features['color']}")

# === Цикл захоплення кадрів (паралельно) ===
def cam_main_loop():
    cam.capture_loop(handle_frame)

Thread(target=cam_main_loop, daemon=True).start()

# === Асинхронний цикл для API статистики ===
async def stats_loop():
    while True:
        try:
            await api.send_stats()
        except Exception as e:
            logger.error(f"API stats error: {e}")
        await asyncio.sleep(60)

# === FASTAPI: Веб-інтерфейс та REST API ===
@app.get("/status")
async def get_status():
    return {
        "system_status": system_status,
        "latest_phrase": latest_phrase,
        "camera_stats": cam.get_stats(),
        "api_stats": api.get_stats(),
        "diagnostics": cam.get_diagnostics()
    }

@app.get("/frame")
async def get_frame():
    global latest_frame
    if latest_frame is not None:
        img_bytes = frame_to_jpeg_bytes(latest_frame)
        return StreamingResponse(
            iter([img_bytes]),
            media_type="image/jpeg"
        )
    return JSONResponse({"error": "No frame available"}, status_code=404)

@app.get("/stats")
async def api_stats():
    return api.get_stats()

@app.get("/camera_stats")
async def camera_stats():
    return cam.get_stats()

@app.get("/diagnostics")
async def diagnostics():
    return cam.get_diagnostics()

@app.get("/phrase")
async def get_phrase():
    return {"latest_phrase": latest_phrase}

@app.post("/config")
async def update_config(req: Request):
    global config
    cfg = await req.json()
    # Гаряче оновлення для phrase_selector, tts, camera
    if "phrase_selector" in cfg:
        phrase_selector.update_config(cfg["phrase_selector"])
        config["phrase_selector"] = cfg["phrase_selector"]
    if "tts" in cfg:
        tts.set_lang(cfg["tts"].get("lang", tts.lang), cfg["tts"].get("gender"))
        tts.set_rate(cfg["tts"].get("rate", tts.rate))
        tts.set_volume(cfg["tts"].get("volume", tts.volume))
        config["tts"] = cfg["tts"]
    if "camera" in cfg:
        # Тільки часткове оновлення (можна зробити більше)
        cam.quality_threshold = cfg["camera"].get("quality_threshold", cam.quality_threshold)
        cam.fps = cfg["camera"].get("fps", cam.fps)
        cam.motion_threshold = cfg["camera"].get("motion_threshold", cam.motion_threshold)
        cam.roi = cfg["camera"].get("roi", cam.roi)
        config["camera"] = cfg["camera"]
    logger.info(f"Config updated via API: {cfg}")
    return {"status": "ok", "config": config}

@app.post("/tts")
async def tts_play(req: Request, background_tasks: BackgroundTasks):
    data = await req.json()
    text = data.get("text", "")
    lang = data.get("lang", tts.lang)
    gender = data.get("gender", tts.gender)
    background_tasks.add_task(tts.play, text, lang, gender, False)
    logger.info(f"TTS request: '{text}' lang={lang} gender={gender}")
    return {"status": "ok"}

@app.get("/selfcheck")
async def selfcheck():
    # Автоматичне тестування працездатності компонентів
    issues = []
    # 1. Камера
    try:
        diag = cam.get_diagnostics()
        if diag["errors"] > 0:
            issues.append("Camera errors detected")
        if diag["last_quality"] is not None and diag["last_quality"] < cam.quality_threshold:
            issues.append("Low video quality")
    except Exception as e:
        issues.append(f"Camera check failed: {e}")
    # 2. API
    try:
        api_stat = api.get_stats()
        if api_stat["fail"] > 0:
            issues.append("API send errors detected")
    except Exception as e:
        issues.append(f"API check failed: {e}")
    # 3. TTS
    try:
        tts.test("Selfcheck test")
    except Exception as e:
        issues.append(f"TTS check failed: {e}")
    # 4. PhraseSelector
    try:
        test_phrase = phrase_selector.test_selection({"gender": "male", "group": "greeting"}, {"timeofday": "day"})
        if not test_phrase:
            issues.append("Phrase selector returned empty")
    except Exception as e:
        issues.append(f"PhraseSelector check failed: {e}")
    logger.info(f"Selfcheck run: {issues}")
    return {"status": "ok" if not issues else "fail", "issues": issues}

@app.get("/logs")
async def get_logs():
    try:
        with open("main.log", "r") as f:
            lines = f.readlines()[-200:]
        return {"log": "".join(lines)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/alert")
async def send_alert(req: Request):
    # Інтеграція для відправки алертів (можна розширити на SMS, email, push)
    data = await req.json()
    msg = data.get("msg", "")
    logger.warning(f"ALERT: {msg}")
    return {"status": "alert logged"}

# === Запуск асинхронного stats-loop ===
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(stats_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)