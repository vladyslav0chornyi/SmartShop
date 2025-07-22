import os
import time
import logging
import threading
from datetime import datetime

try:
    import boto3  # AWS S3 for cloud storage
    from botocore.exceptions import NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

class StorageManager:
    """
    Менеджер збереження відео, аудіо, зображень та логів.
    Локально (Raspberry Pi) + хмарний сервіс (S3).
    """

    def __init__(
        self,
        local_dir="storage",
        enable_video=True,
        enable_audio=True,
        enable_image=True,
        enable_logs=True,
        s3_bucket=None,
        s3_access_key=None,
        s3_secret_key=None,
        s3_region=None,
        log_path="storage_manager.log"
    ):
        self.local_dir = local_dir
        self.enable_video = enable_video
        self.enable_audio = enable_audio
        self.enable_image = enable_image
        self.enable_logs = enable_logs

        self.s3_bucket = s3_bucket
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_region = s3_region

        self.s3_client = None
        if self.s3_bucket and S3_AVAILABLE:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.s3_access_key,
                    aws_secret_access_key=self.s3_secret_key,
                    region_name=self.s3_region
                )
            except Exception as e:
                print(f"S3 init error: {e}")
                self.s3_client = None

        os.makedirs(self.local_dir, exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("StorageManager")

    def _local_save(self, subdir, filename, data):
        folder = os.path.join(self.local_dir, subdir)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            f.write(data)
        self.logger.info(f"Locally saved {subdir}: {filename}")
        return path

    def _s3_upload(self, subdir, filename, local_path):
        if not self.s3_client:
            self.logger.warning("S3 client unavailable")
            return False
        s3_key = f"{subdir}/{filename}"
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            self.logger.info(f"Uploaded to S3: {s3_key}")
            return True
        except NoCredentialsError:
            self.logger.error("S3 credentials error")
            return False
        except Exception as e:
            self.logger.error(f"S3 upload error: {e}")
            return False

    def save_image(self, image_bytes, meta=None):
        if not self.enable_image:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        local_path = self._local_save("images", filename, image_bytes)
        if self.s3_client:
            threading.Thread(target=self._s3_upload, args=("images", filename, local_path), daemon=True).start()
        return local_path

    def save_video(self, video_bytes, meta=None):
        if not self.enable_video:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.mp4"
        local_path = self._local_save("videos", filename, video_bytes)
        if self.s3_client:
            threading.Thread(target=self._s3_upload, args=("videos", filename, local_path), daemon=True).start()
        return local_path

    def save_audio(self, audio_bytes, meta=None):
        if not self.enable_audio:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.wav"
        local_path = self._local_save("audio", filename, audio_bytes)
        if self.s3_client:
            threading.Thread(target=self._s3_upload, args=("audio", filename, local_path), daemon=True).start()
        return local_path

    def save_log(self, log_text, meta=None):
        if not self.enable_logs:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.txt"
        local_path = self._local_save("logs", filename, log_text.encode("utf-8"))
        if self.s3_client:
            threading.Thread(target=self._s3_upload, args=("logs", filename, local_path), daemon=True).start()
        return local_path

    def cleanup_local(self, subdir, keep_last=100):
        folder = os.path.join(self.local_dir, subdir)
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder)], key=os.path.getmtime)
        for f in files[:-keep_last]:
            try:
                os.remove(f)
                self.logger.info(f"Removed old file: {f}")
            except Exception as e:
                self.logger.error(f"Error removing {f}: {e}")

    def get_local_files(self, subdir):
        folder = os.path.join(self.local_dir, subdir)
        if not os.path.exists(folder):
            return []
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)], key=os.path.getmtime)

    def get_s3_files(self, subdir):
        if not self.s3_client:
            return []
        try:
            objects = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=f"{subdir}/")
            return [obj['Key'] for obj in objects.get('Contents', [])]
        except Exception as e:
            self.logger.error(f"S3 list error: {e}")
            return []