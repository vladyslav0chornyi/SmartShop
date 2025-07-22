import logging
import threading
import time
import requests
import json
import random
from datetime import datetime

class PhraseSelector:
    def __init__(
            self,
            phrases=None,
            weights=None,  # {"gender": 2, "clothes": 1, "color": 1, "accessories": 1, "popularity": 0.5, "timeofday": 1, "group": 1}
            log_path="phrase_selector.log",
            fallback_phrases=None,
            phrase_sync_url=None,
            sync_interval=300,  # seconds
            lang="uk",
            config_url=None,
            history_path="phrase_history.json",
            filter_repeat=True,
            filter_window=5,
            enable_visualization=True,
            enable_testing=True
        ):
        self.phrases = phrases or []
        self.weights = weights or {
            "gender": 2, "clothes": 1, "color": 1, "accessories": 1,
            "popularity": 0.5, "timeofday": 1, "group": 1
        }
        self.fallback_phrases = fallback_phrases or [
            {"text": "Вітаємо у нашому магазині!", "lang": "uk", "group": "greeting"},
            {"text": "Welcome to our store!", "lang": "en", "group": "greeting"},
            {"text": "Раді вас бачити!", "lang": "uk", "group": "greeting"},
            {"text": "Happy shopping!", "lang": "en", "group": "greeting"},
        ]
        self.phrase_sync_url = phrase_sync_url
        self.sync_interval = sync_interval
        self.lang = lang
        self.config_url = config_url
        self.history_path = history_path
        self.filter_repeat = filter_repeat
        self.filter_window = filter_window
        self.enable_visualization = enable_visualization
        self.enable_testing = enable_testing

        self.history = []
        self._stop_sync = False
        self.config = {}

        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("PhraseSelector")

        self._load_history()
        if self.phrase_sync_url:
            threading.Thread(target=self._sync_phrases_loop, daemon=True).start()
        if self.config_url:
            threading.Thread(target=self._sync_config_loop, daemon=True).start()

    # --- Мультимовність ---
    def set_lang(self, lang):
        self.lang = lang

    # --- Синхронізація фраз ---
    def update_phrases(self, new_phrases):
        self.phrases = new_phrases
        self.logger.info(f"Оновлено список фраз: {len(new_phrases)} шт.")

    def _sync_phrases_loop(self):
        while not self._stop_sync:
            try:
                response = requests.get(self.phrase_sync_url, timeout=10)
                response.raise_for_status()
                phrase_list = response.json()
                self.update_phrases(phrase_list)
            except Exception as e:
                self.logger.error(f"Помилка синхронізації фраз: {e}")
            time.sleep(self.sync_interval)

    # --- Віддалене конфігурування ---
    def update_config(self, new_config):
        self.config = new_config
        self.logger.info(f"Оновлено конфігурацію: {self.config}")
        # Оновити ваги, fallback, фільтри тощо
        self.weights.update(new_config.get("weights", {}))
        self.fallback_phrases = new_config.get("fallback_phrases", self.fallback_phrases)
        self.filter_repeat = new_config.get("filter_repeat", self.filter_repeat)
        self.filter_window = new_config.get("filter_window", self.filter_window)

    def _sync_config_loop(self):
        while not self._stop_sync:
            try:
                response = requests.get(self.config_url, timeout=10)
                response.raise_for_status()
                config_data = response.json()
                self.update_config(config_data)
            except Exception as e:
                self.logger.error(f"Помилка синхронізації конфігурації: {e}")
            time.sleep(self.sync_interval)

    def stop_sync(self):
        self._stop_sync = True

    # --- Локальна історія вибору фраз ---
    def _load_history(self):
        try:
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
        except:
            self.history = []

    def _save_history(self):
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history[-100:], f)
        except Exception as e:
            self.logger.error(f"Не вдалося зберегти історію: {e}")

    # --- Вибір фрази ---
    def select(self, features, context=None):
        """
        Вибирає найрелевантнішу фразу за ознаками (з урахуванням ваг, часу, групи, популярності, мови, фільтра повторів).
        features: {"gender": "...", "clothes": "...", "color": "...", "accessories": [...], "group": "..."}
        context: {"timeofday": "...", "lang": "...", "event": "...", ...}
        """
        if context is None:
            context = {}
        lang = context.get("lang", self.lang)
        timeofday = context.get("timeofday", self._get_timeofday())
        group = features.get("group") or context.get("group")
        event = context.get("event")

        candidates = [p for p in self.phrases if p.get("lang", "uk") == lang]
        if group:
            candidates = [p for p in candidates if p.get("group") == group]

        # Фільтрація повторів
        if self.filter_repeat and self.history:
            recent = [h["text"] for h in self.history[-self.filter_window:]]
            candidates = [p for p in candidates if p["text"] not in recent]

        best_score = -1
        best_phrase = None
        score_map = {}

        for phrase in candidates:
            score = self._phrase_score(phrase, features, timeofday, group, event)
            score_map[phrase["text"]] = score
            if score > best_score:
                best_score = score
                best_phrase = phrase

        if best_phrase and best_score > 0:
            self._add_history(best_phrase, features, score_map)
            self.logger.info(f"Вибрана фраза: {best_phrase['text']} (score={best_score})")
            if self.enable_visualization:
                self._visualize_choice(best_phrase, features, score_map)
            return best_phrase["text"]
        else:
            chosen_fallback = self._choose_fallback(features, lang, group, timeofday)
            self._add_history({"text": chosen_fallback}, features, score_map)
            self.logger.warning(f"Fallback фраза: {chosen_fallback}")
            if self.enable_visualization:
                self._visualize_choice({"text": chosen_fallback}, features, score_map, fallback=True)
            return chosen_fallback

    def _phrase_score(self, phrase, features, timeofday, group, event):
        score = 0
        # Ознаки з вагами
        for key in ["gender", "clothes", "color"]:
            if phrase.get(key) and features.get(key):
                if phrase[key].lower() == features[key].lower():
                    score += self.weights.get(key, 1)
        # Аксесуари
        if phrase.get("accessories") and features.get("accessories"):
            for acc in features["accessories"]:
                if acc.lower() in phrase["accessories"].lower():
                    score += self.weights.get("accessories", 1)
                    break
        # Група
        if phrase.get("group") and group and phrase["group"].lower() == group.lower():
            score += self.weights.get("group", 1)
        # Час дня
        if phrase.get("timeofday") and timeofday and phrase["timeofday"].lower() == timeofday.lower():
            score += self.weights.get("timeofday", 1)
        # Подія
        if phrase.get("event") and event and phrase["event"].lower() == event.lower():
            score += self.weights.get("event", 1)
        # Популярність
        if phrase.get("popularity"):
            score += float(phrase["popularity"]) * self.weights.get("popularity", 0.5)
        return score

    def _choose_fallback(self, features, lang, group, timeofday):
        # Вибір fallback з урахуванням мови, групи, часу дня, ознак
        f_candidates = [f for f in self.fallback_phrases if f.get("lang", "uk") == lang]
        if group:
            f_candidates = [f for f in f_candidates if f.get("group") == group]
        if timeofday:
            f_candidates = [f for f in f_candidates if f.get("timeofday") == timeofday]
        if not f_candidates:
            f_candidates = self.fallback_phrases
        return random.choice(f_candidates)["text"] if f_candidates else "Вітаємо!"

    def _add_history(self, phrase, features, score_map):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "text": phrase.get("text"),
            "features": features,
            "score_map": score_map
        }
        self.history.append(event)
        self._save_history()

    def _get_timeofday(self):
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 18:
            return "day"
        else:
            return "evening"

    # --- Візуалізація вибору ---
    def _visualize_choice(self, phrase, features, score_map, fallback=False):
        info = {
            "chosen": phrase.get("text"),
            "features": features,
            "score_map": score_map,
            "fallback": fallback
        }
        try:
            with open("phrase_choice_log.json", "a") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Не вдалося записати звіт вибору фрази: {e}")

    # --- Тестування та симуляція ---
    def test_selection(self, features, context=None):
        if not self.enable_testing:
            return "Тестування вимкнено"
        text = self.select(features, context)
        print(f"Test: features={features}, context={context}, selected='{text}'")
        return text