import aiohttp
import asyncio
import logging
import time

class APIClient:
    def __init__(
            self,
            endpoints=None,  # {"frame": "...", "stats": "..."}
            auth_type="Bearer",  # "Bearer", "Basic", "OAuth", "JWT"
            token=None,
            username=None,
            password=None,
            ssl_verify=True,
            proxy=None,
            batch_size=5,
            log_path="api_client.log"
        ):
        self.endpoints = endpoints or {}
        self.auth_type = auth_type
        self.token = token
        self.username = username
        self.password = password
        self.ssl_verify = ssl_verify
        self.proxy = proxy
        self.batch_size = batch_size
        self._batch_frames = []
        self.stats = {
            "success": 0,
            "fail": 0,
            "total_time": 0,
            "last_status": None,
            "last_error": None,
            "sent_batches": 0,
            "last_batch_time": None
        }
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("APIClient")

    # --- Підтримка різних типів авторизації ---
    def _build_headers(self):
        headers = {}
        if self.auth_type == "Bearer" and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.auth_type == "Basic" and self.username and self.password:
            import base64
            b64 = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {b64}"
        elif self.auth_type == "JWT" and self.token:
            headers["Authorization"] = f"JWT {self.token}"
        elif self.auth_type == "OAuth" and self.token:
            headers["Authorization"] = f"OAuth {self.token}"
        return headers

    # --- Асинхронний відправник ---
    async def send_frame(self, frame_bytes, meta=None):
        url = self.endpoints.get("frame")
        if not url:
            self.logger.error("Не задано endpoint для фреймів")
            return
        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("image", frame_bytes, filename="frame.jpg", content_type="image/jpeg")
        if meta:
            for k, v in meta.items():
                data.add_field(k, str(v))
        try:
            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.post(url, data=data, headers=headers, ssl=self.ssl_verify, proxy=self.proxy) as resp:
                    status = resp.status
                    self.stats["last_status"] = status
                    self.stats["total_time"] += time.time() - start
                    self.logger.info(f"Відправлено фрейм, статус: {status}")
                    if status == 200:
                        self.stats["success"] += 1
                    else:
                        self.stats["fail"] += 1
                        self.stats["last_error"] = await resp.text()
        except Exception as e:
            self.stats["fail"] += 1
            self.stats["last_error"] = str(e)
            self.logger.error(f"Помилка відправки фрейму: {e}")

    # --- Масова відправка ---
    async def send_batch(self, frames_meta_list):
        url = self.endpoints.get("frame")
        if not url:
            self.logger.error("Не задано endpoint для batch")
            return
        headers = self._build_headers()
        data = aiohttp.FormData()
        for i, (frame_bytes, meta) in enumerate(frames_meta_list):
            data.add_field(f"image_{i}", frame_bytes, filename=f"frame_{i}.jpg", content_type="image/jpeg")
            if meta:
                for k, v in meta.items():
                    data.add_field(f"{k}_{i}", str(v))
        try:
            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.post(url, data=data, headers=headers, ssl=self.ssl_verify, proxy=self.proxy) as resp:
                    status = resp.status
                    self.stats["last_status"] = status
                    self.stats["sent_batches"] += 1
                    self.stats["last_batch_time"] = time.time()
                    self.stats["total_time"] += time.time() - start
                    self.logger.info(f"Відправлено batch, статус: {status}")
                    if status == 200:
                        self.stats["success"] += len(frames_meta_list)
                    else:
                        self.stats["fail"] += len(frames_meta_list)
                        self.stats["last_error"] = await resp.text()
        except Exception as e:
            self.stats["fail"] += len(frames_meta_list)
            self.stats["last_error"] = str(e)
            self.logger.error(f"Помилка відправки batch: {e}")

    # --- Додавання кадру до batch ---
    def add_to_batch(self, frame_bytes, meta=None):
        self._batch_frames.append((frame_bytes, meta))
        if len(self._batch_frames) >= self.batch_size:
            asyncio.create_task(self.send_batch(self._batch_frames))
            self._batch_frames = []

    # --- Детальний трекінг статистики ---
    def get_stats(self):
        return dict(self.stats)

    # --- Гнучка робота з проксі/мережевими параметрами ---
    def set_proxy(self, proxy_url):
        self.proxy = proxy_url
        self.logger.info(f"Встановлено проксі: {proxy_url}")

    def set_ssl_verify(self, verify):
        self.ssl_verify = verify
        self.logger.info(f"SSL verify: {verify}")

    # --- Оновлення endpoint-ів на льоту ---
    def update_endpoints(self, endpoints_dict):
        self.endpoints.update(endpoints_dict)
        self.logger.info(f"Оновлено endpoints: {self.endpoints}")

    # --- Відправка статистики на сервер ---
    async def send_stats(self):
        url = self.endpoints.get("stats")
        if not url:
            self.logger.error("Не задано endpoint для статистики")
            return
        headers = self._build_headers()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=self.stats, headers=headers, ssl=self.ssl_verify, proxy=self.proxy) as resp:
                    self.logger.info(f"Відправлено статистику, статус: {resp.status}")
        except Exception as e:
            self.logger.error(f"Помилка відправки статистики: {e}")