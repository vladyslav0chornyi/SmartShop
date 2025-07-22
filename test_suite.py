import unittest
import numpy as np
import os
import time
import logging
from parameterized import parameterized

# from client.ml_detector import MLDetector
# from client.storage_manager import StorageManager
# from client.config_manager import ConfigManager

# API-тестування (FastAPI)
try:
    from fastapi.testclient import TestClient
    from client.api_main import app  # Приклад: ваш FastAPI app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# UI-тестування (Selenium)
try:
    from selenium import webdriver
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Логування тестів
logging.basicConfig(filename="test_suite_run.log", level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
test_logger = logging.getLogger("TestSuite")

class TestMLDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MLDetector()

    def test_gender_detection(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        gender = self.detector.detect_gender(frame)
        test_logger.info(f"Gender detection: {gender}")
        self.assertIn(gender, ["male", "female"])

    @parameterized.expand([
        (np.ones((100, 100, 3), dtype=np.uint8) * 200, "happy"),
        (np.ones((100, 100, 3), dtype=np.uint8) * 50, "neutral"),
    ])
    def test_emotion_detection_param(self, frame, expected):
        emotion = self.detector.detect_emotion(frame)
        test_logger.info(f"Emotion detection: {emotion}")
        self.assertEqual(emotion, expected)

    def test_clothes_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        clothes = self.detector.detect_clothes(frame)
        test_logger.info(f"Clothes detection: {clothes}")
        self.assertIn(clothes["type"], ["t-shirt"])
        self.assertIn(clothes["color"], ["red", "blue"])

    def test_analyze(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
        start = time.time()
        result = self.detector.analyze(frame)
        duration = time.time() - start
        test_logger.info(f"Analyze duration: {duration:.4f}s, result: {result}")
        self.assertTrue("gender" in result and "emotion" in result and "clothes" in result)
        # Тест продуктивності: аналіз < 1 сек
        self.assertLess(duration, 1.0)

class TestStorageManager(unittest.TestCase):
    def setUp(self):
        self.storage = StorageManager(local_dir="test_storage")
        self.test_img = b"\x89PNG\r\n\x1a\n"
        self.test_vid = b"\x00\x00\x00\x18ftypmp42"
        self.test_audio = b"RIFF....WAVEfmt "
        self.test_log = "Test log text"

    @parameterized.expand([
        ("image", b"\x89PNG\r\n\x1a\n"),
        ("video", b"\x00\x00\x00\x18ftypmp42"),
        ("audio", b"RIFF....WAVEfmt "),
        ("log", "Test log text"),
    ])
    def test_save_files_param(self, ftype, content):
        if ftype == "image":
            path = self.storage.save_image(content)
        elif ftype == "video":
            path = self.storage.save_video(content)
        elif ftype == "audio":
            path = self.storage.save_audio(content)
        elif ftype == "log":
            path = self.storage.save_log(content)
        test_logger.info(f"Saved {ftype}: {path}")
        self.assertTrue(os.path.exists(path))

    def tearDown(self):
        import shutil
        shutil.rmtree("test_storage", ignore_errors=True)

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.config_file = "logs/test_config.json"
        config_data = {
            "dev": {"param1": "value1", "param2": 2},
            "prod": {"param1": "valueX", "param2": 10}
        }
        with open(self.config_file, "w") as f:
            import json
            json.dump(config_data, f)
        self.config = ConfigManager(config_path=self.config_file, profile="dev")

    @parameterized.expand([
        ("param1", "value1"),
        ("param2", 2),
    ])
    def test_load_config_param(self, key, expected):
        val = self.config.get(key)
        test_logger.info(f"Config get: {key}={val}")
        self.assertEqual(val, expected)

    def test_set_and_save(self):
        self.config.set("param2", 42, save=True)
        self.config.load_config()
        test_logger.info("Config set param2 to 42")
        self.assertEqual(self.config.get("param2"), 42)

    def test_switch_profile(self):
        self.config.switch_profile("prod")
        test_logger.info("Switched profile to prod")
        self.assertEqual(self.config.get("param2"), 10)

    def tearDown(self):
        os.remove(self.config_file)

# API-тести (FastAPI)
if FASTAPI_AVAILABLE:
    class TestAPI(unittest.TestCase):
        def setUp(self):
            self.client = TestClient(app)

        def test_api_status(self):
            resp = self.client.get("/status")
            test_logger.info(f"API /status response: {resp.status_code}")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("status", resp.json())

        def test_api_upload_image(self):
            files = {"file": ("test.jpg", b"\x89PNG\r\n\x1a\n", "image/png")}
            resp = self.client.post("/upload/image", files=files)
            test_logger.info(f"API /upload/image response: {resp.status_code}")
            self.assertEqual(resp.status_code, 200)

# UI-тести (Selenium)
if SELENIUM_AVAILABLE:
    class TestUI(unittest.TestCase):
        def setUp(self):
            self.driver = webdriver.Firefox()  # або Chrome

        def test_ui_title(self):
            self.driver.get("http://localhost:8000")
            title = self.driver.title
            test_logger.info(f"UI title: {title}")
            self.assertIn("My Project", title)

        def tearDown(self):
            self.driver.quit()

if __name__ == "__main__":
    unittest.main()