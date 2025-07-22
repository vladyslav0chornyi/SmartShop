import importlib
import os
import sys
import logging
import subprocess
from typing import List, Optional, Dict, Any

class PluginLoader:
    """
    Модуль для завантаження та керування сторонніми плагінами.
    Плагіни можуть розширювати функціонал системи: обробка кадрів, планувальник, фрази, інтеграції тощо.
    Автоматично встановлює requirements.txt для кожного плагіна при завантаженні.
    Підтримка як .py файлів, так і пакетів-директорій.
    """

    def __init__(self, plugins_dir: str = "plugins", log_path: str = "plugin_loader.log"):
        self.plugins_dir = plugins_dir
        self.plugins: Dict[str, Any] = {}
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("PluginLoader")
        self._init_plugins_dir()

    def _init_plugins_dir(self):
        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir)
        # Додаємо до sys.path для імпорту пакетів
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)

    def discover_plugins(self) -> List[str]:
        """
        Повертає список доступних плагінів:
        - .py файли (без _ на початку)
        - директорії з __init__.py
        """
        plugins = []
        for fname in os.listdir(self.plugins_dir):
            fpath = os.path.join(self.plugins_dir, fname)
            # Файл-плагін
            if fname.endswith(".py") and not fname.startswith("_"):
                plugins.append(fname[:-3])
            # Пакет-плагін
            elif os.path.isdir(fpath) and os.path.isfile(os.path.join(fpath, "__init__.py")):
                plugins.append(fname)
        self.logger.info(f"Discovered plugins: {plugins}")
        return plugins

    def _install_requirements(self, plugin_name: str) -> bool:
        """
        Якщо існує requirements.txt для плагіна, встановити залежності через pip.
        Плагін може бути файлом або папкою.
        """
        # Шляхи для requirements.txt
        pkg_path = os.path.join(self.plugins_dir, plugin_name)
        req_path_dir = os.path.join(pkg_path, "requirements.txt")
        req_path_file = os.path.join(self.plugins_dir, f"{plugin_name}_requirements.txt")
        req_path = None
        if os.path.isfile(req_path_dir):
            req_path = req_path_dir
        elif os.path.isfile(req_path_file):
            req_path = req_path_file

        if not req_path:
            self.logger.info(f"No requirements.txt for {plugin_name}")
            return True  # Немає requirements — все ок

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
            self.logger.info(f"Installed requirements for plugin {plugin_name}: {req_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to install requirements for {plugin_name}: {str(e)}")
            return False

    def load_plugin(self, plugin_name: str) -> Optional[Any]:
        """Динамічно імпортує та повертає плагін. Встановлює залежності, якщо треба."""
        self._install_requirements(plugin_name)
        try:
            module = importlib.import_module(plugin_name)
            self.plugins[plugin_name] = module
            self.logger.info(f"Loaded plugin: {plugin_name}")
            return module
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {str(e)}")
            return None

    def load_all_plugins(self):
        """Завантажити всі доступні плагіни."""
        plugins = self.discover_plugins()
        for name in plugins:
            self.load_plugin(name)

    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Повертає вже завантажений плагін."""
        return self.plugins.get(plugin_name)

    def reload_plugin(self, plugin_name: str) -> Optional[Any]:
        """Перезавантажити плагін."""
        try:
            if plugin_name in self.plugins:
                importlib.reload(self.plugins[plugin_name])
                self.logger.info(f"Reloaded plugin: {plugin_name}")
                return self.plugins[plugin_name]
            else:
                return self.load_plugin(plugin_name)
        except Exception as e:
            self.logger.error(f"Failed to reload plugin {plugin_name}: {str(e)}")
            return None

    def unload_plugin(self, plugin_name: str):
        """Видалити плагін із пам'яті."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.logger.info(f"Unloaded plugin: {plugin_name}")

    def call_plugin(self, plugin_name: str, func_name: str, *args, **kwargs):
        """Виклик функції з плагіна."""
        plugin = self.get_plugin(plugin_name)
        if plugin and hasattr(plugin, func_name):
            return getattr(plugin, func_name)(*args, **kwargs)
        else:
            self.logger.warning(f"Function {func_name} not found in plugin {plugin_name}")
            return None