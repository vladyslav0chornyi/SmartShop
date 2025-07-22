import json
import os
import threading
import logging

class ConfigManager:
    """
    Централізований менеджер конфігурації для проєкту.
    Підтримує гаряче оновлення, кешування, профілі (dev/prod), secrets.
    """

    def __init__(self, config_path="config.json", profile="dev", log_path="config_manager.log"):
        self.config_path = config_path
        self.profile = profile
        self.config = {}
        self.last_mtime = 0
        self.lock = threading.Lock()

        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("ConfigManager")
        self.load_config()

    def load_config(self):
        with self.lock:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file not found: {self.config_path}")
                self.config = {}
                return self.config
            try:
                mtime = os.path.getmtime(self.config_path)
                if mtime > self.last_mtime:
                    with open(self.config_path, "r") as f:
                        data = json.load(f)
                        self.config = data.get(self.profile, {})
                    self.last_mtime = mtime
                    self.logger.info(f"Config loaded for profile '{self.profile}'")
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        return self.config

    def get(self, key, default=None):
        with self.lock:
            return self.config.get(key, default)

    def set(self, key, value, save=False):
        with self.lock:
            self.config[key] = value
            if save:
                self.save_config()

    def save_config(self):
        with self.lock:
            try:
                # Read full config file, update only the current profile
                if os.path.exists(self.config_path):
                    with open(self.config_path, "r") as f:
                        data = json.load(f)
                else:
                    data = {}
                data[self.profile] = self.config
                with open(self.config_path, "w") as f:
                    json.dump(data, f, indent=2)
                self.last_mtime = os.path.getmtime(self.config_path)
                self.logger.info(f"Config saved for profile '{self.profile}'")
            except Exception as e:
                self.logger.error(f"Failed to save config: {e}")

    def reload_if_changed(self):
        """Гаряче оновлення: перевірити чи змінився config-файл і перезавантажити."""
        with self.lock:
            if os.path.exists(self.config_path):
                mtime = os.path.getmtime(self.config_path)
                if mtime > self.last_mtime:
                    self.load_config()
                    return True
        return False

    def switch_profile(self, new_profile):
        with self.lock:
            self.profile = new_profile
            self.load_config()
            self.logger.info(f"Switched to profile: {new_profile}")

    def get_secret(self, key):
        # Секрети можна зберігати окремо (наприклад, secrets.json), тут stub
        secret_path = "secrets.json"
        if not os.path.exists(secret_path):
            return None
        try:
            with open(secret_path, "r") as f:
                secrets = json.load(f)
            return secrets.get(key)
        except Exception as e:
            self.logger.error(f"Failed to get secret: {e}")
            return None