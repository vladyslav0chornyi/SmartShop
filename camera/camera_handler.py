import cv2
import logging
import time
import numpy as np

class CameraHandler:
    def __init__(
            self,
            camera_sources=None,  # Список джерел: 0 або "rtsp://...", "http://...", ...
            default_fps=10,
            min_fps=2,
            max_fps=25,
            log_path="camera_handler.log",
            quality_threshold=40,
            roi=None,  # (x, y, w, h) область інтересу, None — вся картинка
            config_api_url=None,
            motion_threshold=1000,
            diagnostics_api_url=None
        ):
        self.camera_sources = camera_sources or [0]
        self.current_source_idx = 0
        self.cap = None
        self.default_fps = default_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.fps = default_fps
        self.quality_threshold = quality_threshold
        self.roi = roi
        self.motion_threshold = motion_threshold
        self.config_api_url = config_api_url
        self.diagnostics_api_url = diagnostics_api_url
        self.last_frame = None
        self.last_motion = False
        self.last_quality = None
        self.last_diag_sent = time.time()
        self.stats = {
            "frames": 0,
            "bad_quality": 0,
            "motion_events": 0,
            "last_motion_time": None,
            "roi_coords": roi,
            "errors": 0
        }

        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("CameraHandler")
        self._open_camera(self.camera_sources[self.current_source_idx])

    # --- Підтримка різних API камер ---
    def _open_camera(self, source):
        self.logger.info(f"Відкриваю камеру: {source}")
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception("Не вдалося відкрити камеру")
            self.logger.info("Камера відкрита успішно")
        except Exception as e:
            self.logger.error(f"Помилка відкриття камери {source}: {e}")
            self.stats["errors"] += 1
            self._diagnostics_report("open_error", str(e))

    def switch_camera(self, idx):
        if idx < 0 or idx >= len(self.camera_sources):
            self.logger.warning("Спроба вибрати неіснуючу камеру")
            return
        self.current_source_idx = idx
        if self.cap is not None:
            self.cap.release()
        self._open_camera(self.camera_sources[idx])

    # --- Віддалене керування налаштуваннями камери ---
    def update_config_from_api(self):
        if not self.config_api_url:
            return
        try:
            import requests
            response = requests.get(self.config_api_url, timeout=5)
            cfg = response.json()
            self.fps = cfg.get("fps", self.fps)
            self.quality_threshold = cfg.get("quality_threshold", self.quality_threshold)
            self.roi = cfg.get("roi", self.roi)
            self.motion_threshold = cfg.get("motion_threshold", self.motion_threshold)
            self.logger.info(f"Оновлено налаштування з API: {cfg}")
        except Exception as e:
            self.logger.error(f"Помилка отримання конфігу з API: {e}")

    # --- Адаптивний FPS ---
    def _adjust_fps(self, motion_detected):
        if motion_detected:
            self.fps = min(self.fps + 2, self.max_fps)
        else:
            self.fps = max(self.fps - 1, self.min_fps)
        self.logger.info(f"Адаптивний FPS: {self.fps}")

    # --- Моніторинг якості відео ---
    def _frame_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Чіткість
        brightness = np.mean(gray)
        quality_score = lap_var if brightness > 40 else lap_var / 2
        return quality_score

    # --- Виявлення події (motion detection) ---
    def _detect_motion(self, frame):
        if self.last_frame is None:
            self.last_frame = frame
            return False
        diff = cv2.absdiff(frame, self.last_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        motion = np.sum(thresh) > self.motion_threshold
        self.last_frame = frame
        return motion

    # --- Автоматичний вибір ROI ---
    def _auto_roi(self, frame):
        # Простий алгоритм: знаходження найбільшої контурної області
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            self.roi = (x, y, w, h)
            self.stats["roi_coords"] = self.roi
            self.logger.info(f"Оновлено ROI: {self.roi}")

    # --- Високорівнева діагностика ---
    def _diagnostics_report(self, error_type, details):
        now = time.time()
        if self.diagnostics_api_url and (now - self.last_diag_sent > 60):
            try:
                import requests
                payload = {
                    "error_type": error_type,
                    "details": details,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "camera": self.camera_sources[self.current_source_idx]
                }
                requests.post(self.diagnostics_api_url, json=payload, timeout=5)
                self.last_diag_sent = now
                self.logger.info(f"Відправлено діагностику: {payload}")
            except Exception as e:
                self.logger.error(f"Помилка відправки діагностики: {e}")

    # --- Основний цикл захоплення кадрів ---
    def capture_loop(self, frame_callback):
        while True:
            self.update_config_from_api()
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Не вдалося отримати кадр")
                self.stats["errors"] += 1
                self._diagnostics_report("capture_error", "Frame not received")
                time.sleep(2)
                continue

            # Авто ROI
            if self.roi is None:
                self._auto_roi(frame)
            if self.roi:
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]

            # Якість
            quality = self._frame_quality(frame)
            self.last_quality = quality
            self.stats["frames"] += 1
            if quality < self.quality_threshold:
                self.stats["bad_quality"] += 1
                self.logger.warning(f"Погана якість кадру: {quality}")
                self._diagnostics_report("bad_quality", f"Quality={quality}")

            # Motion
            motion = self._detect_motion(frame)
            if motion:
                self.stats["motion_events"] += 1
                self.stats["last_motion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.logger.info("Виявлено рух у кадрі")
                self._adjust_fps(True)
            else:
                self._adjust_fps(False)

            # Виклик зовнішньої обробки (наприклад, надсилання на сервер)
            if motion or quality >= self.quality_threshold:
                frame_callback(frame, {
                    "quality": quality,
                    "motion": motion,
                    "roi": self.roi,
                    "source": self.camera_sources[self.current_source_idx]
                })
            time.sleep(1.0 / self.fps)

    # --- Статистика роботи ---
    def get_stats(self):
        return dict(self.stats)

    # --- Діагностика для UI або API ---
    def get_diagnostics(self):
        return {
            "last_quality": self.last_quality,
            "last_motion": self.last_motion,
            "roi": self.roi,
            "fps": self.fps,
            "errors": self.stats["errors"],
            "source": self.camera_sources[self.current_source_idx]
        }