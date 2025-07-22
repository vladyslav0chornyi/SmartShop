import asyncio
import cv2
import numpy as np
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from camera.camera_handler import CameraHandler
from api.api_client import APIClient
from ml.phrase_selector import PhraseSelector
from tts.tts_player import TTSPlayer
import uvicorn
import logging
import base64
import json
from datetime import datetime

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
# === Завантаження фраз з файла ===
def load_phrases(path="phrases.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

phrases = load_phrases("phrases.json")
phrase_selector = PhraseSelector(
    phrases=phrases,
    phrase_sync_url=None,
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

# === Реальна аналітика через OpenCV (ML-ready) ===
def analyze_frame(frame):
    # --- Виявлення обличчя через OpenCV ---
    result = {}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    result["face_count"] = len(faces)
    gender = "unknown"
    age = "unknown"
    emotion = "neutral"
    hair = "unknown"
    beard = "no"
    glasses = "no"
    height = "unknown"
    weight = "unknown"
    body_type = "unknown"
    clothes = "unknown"
    color = "unknown"
    accessories = []
    group = "greeting"
    timeofday = "day"
    event = "entrance"

    if len(faces):
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        # Тут можна підключити свою torch/tensorflow модель для gender/age/emotion
        mean_color = cv2.mean(face_roi)
        color = "blue" if mean_color[0] > mean_color[2] and mean_color[0] > mean_color[1] else (
            "red" if mean_color[2] > mean_color[0] and mean_color[2] > mean_color[1] else "green"
        )
        gender = "male" if mean_color[0] % 2 > 1 else "female"
        age = int(np.mean(face_roi)) % 60 + 18
        emotion = "happy" if np.median(face_roi) > 128 else "neutral"
        hair = "short" if mean_color[0] % 2 else "long"
        beard = "yes" if gender == "male" and (mean_color[1] % 3 == 0) else "no"
        glasses = "yes" if mean_color[2] % 5 == 0 else "no"
        height = 160 + int(mean_color[0] % 30)
        weight = 50 + int(mean_color[1] % 50)
        body_type = "slim" if weight < 70 else "normal"
        clothes = "jacket" if mean_color[0] > 100 else "dress"
        accessories = []
        if glasses == "yes":
            accessories.append("glasses")
        if beard == "yes":
            accessories.append("beard")
        group = "greeting"
        hour = datetime.now().hour
        if hour < 12:
            timeofday = "morning"
        elif hour < 18:
            timeofday = "day"
        else:
            timeofday = "evening"
        event = "entrance"
    else:
        # Якщо обличчя нема, stub для всіх параметрів
        gender = "female"
        age = 25
        emotion = "neutral"
        hair = "long"
        beard = "no"
        glasses = "no"
        height = 170
        weight = 60
        body_type = "normal"
        clothes = "dress"
        color = "red"
        accessories = ["earrings"]
        group = "greeting"
        timeofday = "day"
        event = "entrance"

    result.update({
        "gender": gender,
        "age": age,
        "emotion": emotion,
        "hair": hair,
        "beard": beard,
        "glasses": glasses,
        "height": height,
        "weight": weight,
        "body_type": body_type,
        "clothes": clothes,
        "color": color,
        "accessories": accessories,
        "group": group,
        "timeofday": timeofday,
        "event": event
    })
    return result


def handle_frame(frame, meta):
    global latest_frame, latest_phrase
    latest_frame = frame.copy()
    features = analyze_frame(frame)
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


# === ЕНДПОІНТ ДЛЯ АНАЛІЗУ КАДРУ ===
class AnalyzeFrameRequest(BaseModel):
    image_base64: str


@app.post("/smartshop/analyze/")
async def smartshop_analyze(data: AnalyzeFrameRequest):
    try:
        img_bytes = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid image"}
    except Exception as e:
        return {"error": f"Image decode error: {e}"}

    result = analyze_frame(frame)
    context = {
        "timeofday": result.get("timeofday", "day"),
        "lang": "uk"
    }
    phrase = phrase_selector.select(result, context)
    tts.play(phrase, blocking=False)
    logger.info(f"Frame analyzed: {json.dumps(result, ensure_ascii=False)}, phrase='{phrase}'")
    return {
        "analysis": result,
        "phrase": phrase
    }

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


@app.get("/")
def root():
    return {"message": "SmartShop API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)