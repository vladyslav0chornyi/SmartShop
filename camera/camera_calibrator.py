import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

class CameraCalibrator:
    """
    Модуль для калібрування камери за допомогою chessboard/charuco/aruco патернів.
    Підтримка збереження/завантаження коефіцієнтів, автоматичний пошук куточків, оцінка якості калібрування.
    """

    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6), square_size: float = 1.0, log_path="camera_calibrator.log"):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger("CameraCalibrator")

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Додає кадр для калібрування, автоматично шукає chessboard.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if found:
            objp = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
            self.logger.info("Chessboard detected and points added.")
            return True
        else:
            self.logger.warning("Chessboard not detected in frame.")
            return False

    def calibrate(self, frame_shape: Tuple[int, int]) -> bool:
        """
        Проводить калібрування камери, повертає True якщо успішно.
        """
        if len(self.objpoints) < 3:
            self.logger.warning("Not enough frames for calibration.")
            return False
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, frame_shape[::-1], None, None)
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.logger.info("Camera calibration successful.")
            return True
        else:
            self.logger.error("Camera calibration failed.")
            return False

    def save(self, path: str):
        """
        Зберігає коефіцієнти калібрування у файл.
        """
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            np.savez(path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
            self.logger.info(f"Calibration saved to {path}")

    def load(self, path: str):
        """
        Завантажує коефіцієнти калібрування з файлу.
        """
        try:
            data = np.load(path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.logger.info(f"Calibration loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load calibration from {path}: {str(e)}")

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Виконує корекцію дисторсії кадру.
        """
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        else:
            self.logger.warning("Calibration not loaded; returning original frame.")
            return frame