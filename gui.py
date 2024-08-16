import sys
import cv2
import numpy as np
import debugpy
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from algorithms.classification import detect_signs

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, method: str):
        super().__init__()
        self._run_flag = True
        self.detection_time = []
        self.method = method

    def run(self):
        # Adding thread to debugger
        debugpy.debug_this_thread()
        self.setObjectName("VideoThread")

        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture("C:\\signs-recognition-files\\full\\videos\\afternoon1.wmv")
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Placeholder for image analysis
                analyzed_frame = self.analyze_image(frame)
                
                qformat = QImage.Format_Indexed8
                if len(analyzed_frame.shape) == 3:
                    if analyzed_frame.shape[2] == 4:
                        qformat = QImage.Format_RGBA8888
                    else:
                        qformat = QImage.Format_RGB888
                out_image = QImage(analyzed_frame, analyzed_frame.shape[1], analyzed_frame.shape[0], analyzed_frame.strides[0], qformat)
                out_image = out_image.rgbSwapped()
                self.change_pixmap_signal.emit(out_image)
        cap.release()

    def analyze_image(self, frame):
        # Placeholder for image analysis logic
        # Example: Convert the image to grayscale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detect_signs(frame, self.method)
        if result:
            self.detection_time.append()
        return frame

    def analyze_performance(self):
        print('Average execution time: ', np.mean(np.asarray(self.detection_time)))
        print('Standard deviation for execution time: ', np.std(np.asarray(self.detection_time)))

    def stop(self):
        self._run_flag = False
        self.wait()

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.classifier_selector = QComboBox(self)
        self.classifier_selector.addItem("SVM")
        self.classifier_selector.addItem("Azure CNN")
        self.classifier_selector.addItem("CNN")
        self.layout.addWidget(self.classifier_selector)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_classification)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_classification)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        self.thread = None

    def start_classification(self):
        if self.thread is not None:
            self.thread.stop()
        self.thread = VideoThread(self.classifier_selector.currentText())
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def stop_classification(self):
        if self.thread is not None:
            self.thread.stop()

    def update_image(self, cv_img):
        self.image_label.setPixmap(QPixmap.fromImage(cv_img))
        self.image_label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec_())
