import cv2
import numpy as np
from openvino.runtime import Core

class PersonAttributesDetector:
    def __init__(self, model_path):
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        self.compiled_model = self.ie.compile_model(self.model, "CPU")
        self.input_layer = next(iter(self.model.inputs))
        self.output_layer = next(iter(self.model.outputs))

    def preprocess(self, frame):
        resized = cv2.resize(frame, (64, 128))
        img = resized.transpose((2, 0, 1))  # HWC -> CHW
        img = img[np.newaxis, :]
        img = img.astype(np.float32) / 255.0
        return img

    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        result = self.compiled_model([input_tensor])[self.output_layer]
        attrs = result.flatten()
        gender = "male" if attrs[0] > 0.5 else "female"
        hair = "short" if attrs[1] > 0.5 else "long"
        glasses = "yes" if attrs[2] > 0.5 else "no"
        hat = "yes" if attrs[3] > 0.5 else "no"

        roi = frame[frame.shape[0]//4:frame.shape[0]*3//4, frame.shape[1]//4:frame.shape[1]*3//4]
        avg_color = cv2.mean(cv2.resize(roi, (32, 32)))[:3]
        color = self._get_color_name(avg_color)
        clothes = self._detect_clothes(frame)
        body_type = self._estimate_body_type(frame)
        accessories = []
        if glasses == "yes":
            accessories.append("glasses")
        if hat == "yes":
            accessories.append("hat")

        age = self._estimate_age(frame)
        emotion = self._estimate_emotion(frame)
        beard = "no"
        height = self._estimate_height(frame)
        weight = self._estimate_weight(frame, body_type)

        result_dict = {
            "gender": gender,
            "hair": hair,
            "glasses": glasses,
            "hat": hat,
            "clothes": clothes,
            "color": color,
            "body_type": body_type,
            "accessories": accessories,
            "age": age,
            "emotion": emotion,
            "beard": beard,
            "height": height,
            "weight": weight,
        }
        return result_dict

    def _get_color_name(self, avg_color):
        b, g, r = avg_color
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif abs(r - g) < 15 and abs(r - b) < 15:
            return "grey"
        else:
            return "unknown"

    def _detect_clothes(self, frame):
        return "t-shirt"

    def _estimate_body_type(self, frame):
        h, w, _ = frame.shape
        aspect = w / h
        if aspect > 0.6:
            return "athletic"
        else:
            return "normal"

    def _estimate_age(self, frame):
        return 28

    def _estimate_emotion(self, frame):
        return "neutral"

    def _estimate_height(self, frame):
        return 180

    def _estimate_weight(self, frame, body_type):
        if body_type == "athletic":
            return 80
        else:
            return 70