import logging
import pyttsx3
from threading import Thread
import re

class TTSPlayer:
    def __init__(
            self,
            lang="uk",
            gender=None,  # "male" or "female"
            rate=160,
            volume=1.0,
            log_path="tts_player.log",
            error_log_path="tts_player_error.log",
            voice_id=None,
            audio_output=None,  # device name or id
            callback=None
        ):
        self.lang = lang
        self.gender = gender
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.audio_output = audio_output
        self.callback = callback

        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("TTSPlayer")
        self.error_logger = logging.getLogger("TTSPlayerError")
        handler = logging.FileHandler(error_log_path)
        self.error_logger.addHandler(handler)

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        self._set_voice(self.lang, self.gender, self.voice_id)
        if self.audio_output:
            self._set_audio_output(self.audio_output)

        self._paused = False

    def _set_voice(self, lang, gender, voice_id):
        voices = self.engine.getProperty('voices')
        selected_voice = None
        # Вибір голосу за lang, gender або voice_id
        for v in voices:
            if voice_id and v.id == voice_id:
                selected_voice = v.id
                break
            # gender detection by name or id (platform dependent)
            gender_match = True
            if gender:
                if gender.lower() == "male" and ("male" in v.name.lower() or "male" in v.id.lower()):
                    gender_match = True
                elif gender.lower() == "female" and ("female" in v.name.lower() or "female" in v.id.lower()):
                    gender_match = True
                else:
                    gender_match = False
            if lang in v.languages or lang in v.name or lang in v.id:
                if gender_match:
                    selected_voice = v.id
                    break
        if selected_voice:
            self.engine.setProperty('voice', selected_voice)
            self.logger.info(f"Встановлено голос: {selected_voice}")
        else:
            self.logger.warning(f"Голос для мови '{lang}' і статті '{gender}' не знайдено, використовую стандартний.")

    def _set_audio_output(self, output_device):
        try:
            self.engine.setProperty('output_device', output_device)
            self.logger.info(f"Встановлено аудіо-вихід: {output_device}")
        except Exception as e:
            self.error_logger.error(f"Не вдалося встановити аудіо-вихід: {e}")

    def set_lang(self, lang, gender=None, voice_id=None):
        self.lang = lang
        self.gender = gender
        self._set_voice(lang, gender, voice_id)
        self.logger.info(f"Встановлено мову: {lang}, стать: {gender}")

    def set_rate(self, rate):
        self.rate = rate
        self.engine.setProperty('rate', rate)
        self.logger.info(f"Встановлено швидкість: {rate}")

    def set_volume(self, volume):
        self.volume = volume
        self.engine.setProperty('volume', volume)
        self.logger.info(f"Встановлено гучність: {volume}")

    def pause(self):
        self._paused = True
        self.engine.pause()
        self.logger.info("Озвучування призупинено")

    def resume(self):
        self._paused = False
        self.engine.resume()
        self.logger.info("Озвучування відновлено")

    def stop(self):
        self.engine.stop()
        self.logger.info("Озвучування зупинено")

    def play(self, text, lang=None, gender=None, blocking=True, callback=None):
        """
        Озвучити текст. Якщо blocking=False — у фоновому потоці.
        Автоматичне визначення мови, попередня обробка тексту, callback після завершення.
        """
        lang = lang or self._detect_lang(text) or self.lang
        gender = gender or self.gender
        self.set_lang(lang, gender)
        clean_text = self._preprocess_text(text)

        def speak():
            try:
                self.engine.say(clean_text)
                self.engine.runAndWait()
                self.logger.info(f"Озвучено: {clean_text}")
                if callback:
                    callback()
                elif self.callback:
                    self.callback()
            except Exception as e:
                self.error_logger.error(f"Помилка озвучки: {e}")

        if blocking:
            speak()
        else:
            Thread(target=speak, daemon=True).start()

    def available_voices(self):
        voices = self.engine.getProperty('voices')
        info = []
        for v in voices:
            info.append({
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
                "gender": "female" if "female" in v.name.lower() or "female" in v.id.lower() else "male"
            })
        self.logger.info(f"Доступні голоси: {info}")
        return info

    def test(self, text="Тест синтезу мови!", lang=None, gender=None):
        self.play(text, lang, gender, blocking=True)

    # --- Попередня обробка тексту ---
    def _preprocess_text(self, text):
        # Прибрати спецсимволи, скорегувати числа для кращої вимови тощо
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Можна додати додаткові правила
        return text

    # --- Автоматичне визначення мови тексту ---
    def _detect_lang(self, text):
        # Дуже просте визначення: якщо є англійські слова — "en", інакше "uk"
        if re.search(r'[a-zA-Z]', text):
            return "en"
        else:
            return "uk"

    # --- Callback підтримка ---
    def set_callback(self, callback):
        self.callback = callback

    # --- Логування критичних помилок ---
    def log_error(self, msg):
        self.error_logger.error(msg)

    # --- Швидке прискорення ---
    def accelerate(self, factor=1.25):
        new_rate = int(self.rate * factor)
        self.set_rate(new_rate)
        self.logger.info(f"Швидкість прискорена до {new_rate}")

    # --- Вибір аудіо-виходу ---
    def set_audio_output(self, output_device):
        self.audio_output = output_device
        self._set_audio_output(output_device)