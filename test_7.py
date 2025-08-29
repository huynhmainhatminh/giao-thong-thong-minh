import sys
import av
import time
import asyncio
import aiohttp
import numpy as np
import cv2
import json
import ast
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QListWidgetItem, QLabel, QDialog
from untitled_ui3 import Ui_MainWindow
from dialog_thongtin import Dialog_ThongTin
from dialog_help import Dialog_Help
from PyQt6 import QtCore, QtGui, QtWidgets
from io import BytesIO
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO



class AsyncWorker_CameraLive(QThread):
    finished = pyqtSignal()

    def __init__(self, camera_url, title_camera, label_title, label_camera, label_close, label_time, checkbox_detection_1
                 ):
        super().__init__()
        self.camera_url = camera_url
        self.title_camera = title_camera

        self.label_title = label_title
        self.label_camera = label_camera
        self.label_close = label_close
        self.label_time = label_time

        self.running = True

        files_coordinates = json.loads(open("video.json", encoding="UTF-8").read())['data'][camera_url]


        self.model = YOLO("yolo11m.pt")  # Load YOLO11 model (nano version)

        self.module = YOLO(f"module/{files_coordinates['module']}")


        self.label_camera.setMaximumSize(1280, 720)

        self.label_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_time.setStyleSheet("""
                QLabel {
                    background-color: rgb(255, 0, 0);
                    color: rgb(255, 255, 255);
                }
            """)

        self.checkbox_detection_1 = checkbox_detection_1  # phát hiện và nhận diện toàn bộ xe
        self.checkbox_detection_1.stateChanged.connect(self.on_checkbox_1_state_changed)
        self.bool_detection_1 = False
        self.coordinates_1: list = ast.literal_eval(f"[{files_coordinates['check_box1']}]")



    def is_box_in_polygon(self, box, polygon):
        x1, y1, x2, y2 = map(int, box)
        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Tâm của hộp giới hạn
        return cv2.pointPolygonTest(polygon, box_center, False) >= 0


    def on_checkbox_1_state_changed(self, state):
        if state == 2:  # Trạng thái đã chọn (checked)
            self.bool_detection_1 = True
            # label.setText("Đã chọn")
        else:  # Trạng thái chưa chọn (unchecked)
            self.bool_detection_1 = False


    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.update_camera_live())


    def detect_vehicle(self, img):

        pts = np.array(self.coordinates_1, np.int32)

        pts = pts.reshape((-1, 1, 2))  # Định dạng cho OpenCV

        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        results = self.model.predict(
            img,
            classes=[1, 2, 3, 5, 7],
            conf=0.3,  # Adjusted confidence threshold for better accuracy
            # iou=0.5  # IoU threshold for NMS
        )

        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp giới hạn
                cls_id = int(box.cls)
                conf = box.conf[0]  # Độ tin cậy
                class_name = result.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                # Kiểm tra xem hộp giới hạn có nằm trong vùng tứ giác không
                if self.is_box_in_polygon([x1, y1, x2, y2], pts):
                    # Vẽ hộp giới hạn màu xanh lá (0, 255, 0) cho xe mô tô trong vùng
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img


    def detection_red_light(self, img):
        class_names = ["do", "xanh"]

        # Run prediction on the image

        # Run prediction on the image
        results = self.module.predict(source=img, show=False, save=False)  # save=False to avoid saving output

        for result in results:
            # Get the annotated image from YOLO (in BGR format for OpenCV)

            # Extract class names
            classes = result.boxes.cls  # Class IDs
            if len(classes) == 0:
                print("No traffic lights detected in the image.")
            else:
                for class_id, box in zip(
                        classes, result.boxes.xyxy
                ):  # Loop through each class and corresponding bounding box
                    class_name = class_names[int(class_id)]  # Map class ID to name

                    if class_name == "do":
                        x1, y1, x2, y2 = box
                        cv2.rectangle(
                            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                        )

                        cv2.putText(
                            img, "do", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA
                            )

                        # cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                        # self.detection_red_vehicle(img)


                    elif class_name == "xanh":
                        x1, y1, x2, y2 = box
                        cv2.rectangle(
                            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                        )

                        cv2.putText(
                            img, "xanh", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            cv2.LINE_AA
                        )
                        # cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        return img





    async def update_camera_live(self):


        container = av.open(self.camera_url, options={
        'buffer_size': '2048000',  # Tăng kích thước bộ đệm để giảm giật
        'timeout': '10',           # Đặt timeout để tránh treo khi kết nối chậm
        'http_persistent': '1',    # Giữ kết nối HTTP liên tục
        # 'max_delay': '500000',  # Giới hạn độ trễ tối đa (microseconds)
    })
        last_pts = None

        for frame in container.decode(video=0):
            if not self.running:
                break

            # Chuyển frame thành mảng numpy (BGR)
            img = frame.to_ndarray(format='bgr24')
            img = cv2.convertScaleAbs(img, alpha=1.2)  # Tăng độ sáng/tương phản
            # img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            img = cv2.resize(img, (1280, 720))

            # Tính delay giữa các frame
            cur_pts = float(frame.pts * frame.time_base)

            if self.bool_detection_1:
                img = self.detect_vehicle(img)

            self.detection_red_light(img)

            if last_pts is not None:
                delay = cur_pts - last_pts
                if delay > 0:
                    time.sleep(delay)
            last_pts = cur_pts

            # Chuyển đổi sang QImage để hiển thị trong QLabel
            height, width, channel = img.shape
            qimage = QImage(img.data, width, height, width * channel, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.label_camera.setPixmap(pixmap.scaled(self.label_camera.size(), Qt.AspectRatioMode.KeepAspectRatio))


class AsyncWorker_CameraAnh(QThread):
    finished = pyqtSignal()

    def __init__(self,
                 camera_id, title_camera, label_title, label_camera, label_close, label_time,
                 checkbox_detection_1, checkbox_detection_2, checkbox_detection_3
                 ):
        super().__init__()

        files_coordinates = json.loads(open("address_id.json", encoding="UTF-8").read())['data'][camera_id]

        self.camera_id = camera_id
        self.title_camera = title_camera

        self.label_title = label_title
        self.label_camera = label_camera
        self.label_close = label_close
        self.label_time = label_time

        #### Phát hiện toàn bộ xem ####
        self.checkbox_detection_1 = checkbox_detection_1  # phát hiện và nhận diện toàn bộ xe
        self.checkbox_detection_1.stateChanged.connect(self.on_checkbox_1_state_changed)
        self.bool_detection_1 = False
        self.coordinates_1: list = ast.literal_eval(f"[{files_coordinates['check_box1']}]")


        #### Phát hiện xe vượt đèn đỏ ####

        self.checkbox_detection_2 = checkbox_detection_2  # phát hiện xe vượt đèn đỏ
        self.checkbox_detection_2.stateChanged.connect(self.on_checkbox_2_state_changed)
        self.bool_detection_2 = False
        self.coordinates_2: list = ast.literal_eval(f"[{files_coordinates['check_box2']}]")

        self.checkbox_detection_3 = checkbox_detection_3  # phát hiện xe vượt đèn đỏ
        self.checkbox_detection_3.stateChanged.connect(self.on_checkbox_3_state_changed)
        self.bool_detection_3 = False
        self.coordinates_3: list = ast.literal_eval(f"[{files_coordinates['check_box3']}]")

        self.module = YOLO(f"module/{files_coordinates['module']}")


        self.timeout = aiohttp.ClientTimeout(total=60.0)

        self.model = YOLO("yolo11m.pt")  # Load YOLO11 model (nano version)


        self.label_title.setText(f"{self.title_camera}")

        self.label_close.clicked.connect(self.stop)

        self.time_left = 6
        self.running = True

        self.label_camera.setMaximumSize(1280, 720)

        self.label_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Define ROI coordinates for original image size (1280x720)



    def stop(self):
        self.running = False  # Dừng vòng lặp
        self.quit()  # Quit the thread
        self.wait()  # Wait for the thread to finish

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.update_camera_anh())

    def is_box_in_polygon(self, box, polygon):
        x1, y1, x2, y2 = map(int, box)
        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Tâm của hộp giới hạn
        return cv2.pointPolygonTest(polygon, box_center, False) >= 0


    def on_checkbox_1_state_changed(self, state) -> None:
        if state == 2:  # Trạng thái đã chọn (checked)
            self.bool_detection_1 = True
            # label.setText("Đã chọn")
        else:  # Trạng thái chưa chọn (unchecked)
            self.bool_detection_1 = False


    def on_checkbox_2_state_changed(self, state) -> None:
        if state == 2:
            self.bool_detection_2 = True
        else:
            self.bool_detection_2 = False


    def on_checkbox_3_state_changed(self, state) -> None:
        if state == 2:
            self.bool_detection_3 = True
        else:
            self.bool_detection_3 = False


    def detect_sidewalk(self, img):
        pts = np.array(self.coordinates_3, np.int32)

        pts = pts.reshape((-1, 1, 2))  # Định dạng cho OpenCV

        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        results = self.model.predict(
            img,
            classes=[1, 2, 3, 5, 7],
            conf=0.3,  # Adjusted confidence threshold for better accuracy
        )

        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp giới hạn
                cls_id = int(box.cls)
                conf = box.conf[0]  # Độ tin cậy
                class_name = result.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                # Kiểm tra xem hộp giới hạn có nằm trong vùng tứ giác không
                if self.is_box_in_polygon([x1, y1, x2, y2], pts):
                    # Vẽ hộp giới hạn màu xanh lá (0, 255, 0) cho xe mô tô trong vùng
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

    def detect_vehicle(self, img):


        pts = np.array(self.coordinates_1, np.int32)

        pts = pts.reshape((-1, 1, 2))  # Định dạng cho OpenCV


        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)


        results = self.model.predict(
                img,
                classes=[1, 2, 3, 5, 7],
                conf=0.3,  # Adjusted confidence threshold for better accuracy
            )

            # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp giới hạn
                cls_id = int(box.cls)
                conf = box.conf[0]  # Độ tin cậy
                class_name = result.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                # Kiểm tra xem hộp giới hạn có nằm trong vùng tứ giác không
                if self.is_box_in_polygon([x1, y1, x2, y2], pts):
                    # Vẽ hộp giới hạn màu xanh lá (0, 255, 0) cho xe mô tô trong vùng
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

    def detection_red_vehicle(self, img):
        pts = np.array(self.coordinates_2, np.int32)
        pts = pts.reshape((-1, 1, 2))  # Định dạng cho OpenCV

        # Tiến hành dự đoán với mô hình YOLO
        results = self.model.predict(
            img,
            classes=[1, 2, 3, 5, 7],
            conf=0.3,  # Điều chỉnh ngưỡng độ tin cậy cho kết quả chính xác hơn
        )

        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp giới hạn
                cls_id = int(box.cls)
                conf = box.conf[0]  # Độ tin cậy
                class_name = result.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                # Kiểm tra xem hộp giới hạn có nằm trong vùng tứ giác không
                if self.is_box_in_polygon([x1, y1, x2, y2], pts):
                    # Vẽ hộp giới hạn màu xanh lá (0, 255, 0) cho xe mô tô trong vùng
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Chụp lại khu vực của xe (cắt vùng chứa xe)
                    vehicle_crop = img[y1:y2, x1:x2]  # Cắt ảnh theo tọa độ của hộp giới hạn

                    # Lưu ảnh xe vào một file mới
                    vehicle_filename = f"vehicle_{cls_id}_{conf:.2f}.jpg"
                    cv2.imwrite(vehicle_filename, vehicle_crop)  # Lưu ảnh với tên tương ứng

                    print(f"Vehicle cropped and saved as {vehicle_filename}")

        return img

    def detection_red_light(self, img):

        pts = np.array(self.coordinates_2, np.int32)

        pts = pts.reshape((-1, 1, 2))  # Định dạng cho OpenCV

        class_names = ["do", "xanh"]

        # Run prediction on the image

        # Run prediction on the image
        results = self.module.predict(source=img, show=False, save=False, conf=0.6)  # save=False to avoid saving output

        for result in results:
            # Get the annotated image from YOLO (in BGR format for OpenCV)

            # Extract class names
            classes = result.boxes.cls  # Class IDs
            if len(classes) == 0:
                print("No traffic lights detected in the image.")
            else:
                for class_id, box in zip(
                        classes, result.boxes.xyxy
                        ):  # Loop through each class and corresponding bounding box
                    class_name = class_names[int(class_id)]  # Map class ID to name

                    if class_name == "do":
                        x1, y1, x2, y2 = box
                        cv2.rectangle(
                            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                            )

                        cv2.putText(img, "do", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                        self.detection_red_vehicle(img)


                    elif class_name == "xanh":
                        x1, y1, x2, y2 = box
                        cv2.rectangle(
                            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                        )

                        cv2.putText(
                            img, "xanh", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            cv2.LINE_AA
                            )
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        return img


    async def get_images(self, session):


        async with session.get(f'https://api.binhdinh.ttgt.vn/v4/cameras/{self.camera_id}/snapshot',
            headers={
                'authority': 'api.binhdinh.ttgt.vn',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/jxl,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'accept-language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
                'cache-control': 'no-cache',
                'dnt': '1',
                'pragma': 'no-cache',
                'sec-ch-ua': '"Chromium";v="117", "Not;A=Brand";v="8"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'none',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            }, timeout=self.timeout) as response:

            image_data = await response.content.read()

            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if self.bool_detection_1:
                img = self.detect_vehicle(img)

            if self.bool_detection_2:
                img = self.detection_red_light(img)

            if self.bool_detection_3:
                img = self.detect_sidewalk(img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)

            self.label_camera.setPixmap(pixmap.scaled(self.label_camera.size(), Qt.AspectRatioMode.KeepAspectRatio))

    async def update_camera_anh(self):

        async with aiohttp.ClientSession() as session:
            await self.get_images(session)
            while self.running:
                # Giảm 1 giây mỗi lần gọi
                self.time_left -= 1
                self.label_time.setText(f'Đếm Ngược : {self.time_left} Giây')

                # Nếu hết thời gian, dừng tác vụ
                if self.time_left == 0:
                    self.time_left = 6
                    await self.get_images(session)

                else:
                    # Nếu chưa hết thời gian, tiếp tục đếm ngược
                    await asyncio.sleep(1)

            self.label_camera.setPixmap(QtGui.QPixmap("ImageHandler.png"))
            self.label_time.setText(f'Đếm Ngược : 5 Giây')
            # self.finished.emit()  # Phát tín hiệu khi dừng


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Mắt Thần")

        self.pushButton_9.click()
        self.pushButton_10.click()
        self.pushButton_21.click()

        # hình ảnh | tiêu đề | thời gian | button close | phát hiện các loại xe | phát hiện xe vượt đèn đỏ | phát
        # hiện đi trên vỉa hè |

        self.list_theo_doi_camera_anh = [
            "label_12|label_23|label_22|pushButton_2|checkBox|checkBox_2|checkBox_3|checkBox_4",
            "label_13|label_30|label_31|pushButton_4|checkBox_9|checkBox_10",
            "label_14|label_33|label_32|pushButton_8|checkBox_17|checkBox_18",
            "label_15|label_36|label_35|pushButton_13|checkBox_21|checkBox_22",
            "label_16|label_39|label_38|pushButton_14|checkBox_25|checkBox_26",
            "label_17|label_42|label_41|pushButton_15|checkBox_29|checkBox_30",
            "label_18|label_45|label_78|pushButton_16|checkBox_33|checkBox_34",
            "label_19|label_48|label_47|pushButton_17|checkBox_37|checkBox_38",
            "label_20|label_51|label_50|pushButton_18|checkBox_41|checkBox_42"
        ]

        self.list_theo_doi_camera_live = [
            "label_28|label_26|label_25|pushButton_26|checkBox_5|checkBox_6",
            "label_53|label_77|label_27|pushButton_27|"
        ]

        self.files_camera_anh = open("address_id.txt", "r", encoding="UTF-8").read().strip().split("\n")

        self.files_camera_live = open("video_live.txt", "r", encoding="UTF-8").read().strip().split("\n")

        for line in self.files_camera_anh:
            text = line.split("|")[1]
            self.listWidget.addItem(text)

        for line in self.files_camera_live:
            text = line.split("|")[1]
            self.listWidget_2.addItem(text)

        self.listWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(self.on_context_menu_camera_anh)

        self.listWidget_2.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.listWidget_2.customContextMenuRequested.connect(self.on_context_menu_camera_live)


        self.pushButton_11.clicked.connect(self.show_menu_camera_anh) # danh sách camera ảnh
        self.pushButton_12.clicked.connect(self.show_menu_theo_doi_camera_anh) # danh sách camera ảnh

        self.pushButton_19.clicked.connect(self.show_menu_camera_live) # danh sách camera live
        self.pushButton_20.clicked.connect(self.show_menu_theo_doi_camera_live) # danh sách camera live

        self.pushButton_7.clicked.connect(self.show_thong_tin) # thông tin
        self.display_thong_tin = None

        self.pushButton_6.clicked.connect(self.show_help) # help
        self.display_help = None


    def on_context_menu_camera_live(self, pos):
        item = self.listWidget_2.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        qss = """
            QMenu {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;   /* Đảm bảo các góc của menu mềm mại */
                padding: 5px;
            }
            QMenu::item {
                color: #ffffff;
                padding: 5px;
            }
            QMenu::item:selected {
                background-color: #3d7848;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #444;
                margin: 5px 0;
            }
        """
        menu.setStyleSheet(qss)
        act_theodoi_camera = menu.addAction("Theo Dõi")
        action = menu.exec(self.listWidget.mapToGlobal(pos))

        if action == act_theodoi_camera:
            self.theo_doi_camera_live(item)



    def theo_doi_camera_live(self,  item: QListWidgetItem):

        self.worker = None
        if not self.list_theo_doi_camera_live:
            return

        row = self.listWidget_2.row(item)
        #
        data_camera_live = str(self.list_theo_doi_camera_live.pop(0)).split("|")

        name_camera = data_camera_live[0]
        title_camera = data_camera_live[1]
        time_camera = data_camera_live[2]
        close_camera = data_camera_live[3]
        checkBox_1 = data_camera_live[4]
        label_checkBox_1 = getattr(self, checkBox_1, None)


        label_camera = getattr(self, name_camera, None)
        label_title = getattr(self, title_camera, None)
        label_time = getattr(self, time_camera, None)
        label_close = getattr(self, close_camera, None)

        for line in self.files_camera_live:
            data_camera = line.split("|")
            if data_camera[1] == item.text():
                camera_url = data_camera[0]
                title_camera = data_camera[1]

                self.worker = AsyncWorker_CameraLive(
                    camera_url=camera_url, title_camera=title_camera, label_camera=label_camera,
                    label_title=label_title, label_time=label_time, label_close=label_close, checkbox_detection_1=label_checkBox_1
                )
                self.worker.start()
                break



    def on_context_menu_camera_anh(self, pos):
        item = self.listWidget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        qss = """
            QMenu {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;   /* Đảm bảo các góc của menu mềm mại */
                padding: 5px;
            }
            QMenu::item {
                color: #ffffff;
                padding: 5px;
            }
            QMenu::item:selected {
                background-color: #3d7848;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #444;
                margin: 5px 0;
            }
        """
        menu.setStyleSheet(qss)
        act_theodoi_camera = menu.addAction("Theo Dõi")
        action = menu.exec(self.listWidget.mapToGlobal(pos))

        if action == act_theodoi_camera:
            self.theo_doi_camera_anh(item)


    def theo_doi_camera_anh(self,  item: QListWidgetItem):

        self.worker = None
        if not self.list_theo_doi_camera_anh:
            return

        row = self.listWidget.row(item)
        #
        data_camera_anh = str(self.list_theo_doi_camera_anh.pop(0)).split("|")

        name_camera = data_camera_anh[0]
        title_camera = data_camera_anh[1]
        time_camera = data_camera_anh[2]
        close_camera = data_camera_anh[3]
        checkBox_1 = data_camera_anh[4]
        checkBox_2 = data_camera_anh[5]
        checkBox_3 = data_camera_anh[6]

        label_camera = getattr(self, name_camera, None)
        label_title = getattr(self, title_camera, None)
        label_time = getattr(self, time_camera, None)
        label_close = getattr(self, close_camera, None)
        label_checkBox_1 = getattr(self, checkBox_1, None)
        label_checkBox_2 = getattr(self, checkBox_2, None)
        label_checkBox_3 = getattr(self, checkBox_3, None)

        for line in self.files_camera_anh:
            data_camera = line.split("|")
            if data_camera[1] == item.text():
                camera_id = data_camera[0]
                title_camera = data_camera[1]

                self.worker = AsyncWorker_CameraAnh(
                    camera_id=camera_id, title_camera=title_camera, label_camera=label_camera,
                    label_title=label_title, label_time=label_time, label_close=label_close,
                    checkbox_detection_1=label_checkBox_1, checkbox_detection_2=label_checkBox_2, checkbox_detection_3=label_checkBox_3
                )

                self.worker.start()
                break

    def show_menu_camera_anh(self):
        self.stackedWidget.setCurrentIndex(2)

    def show_menu_camera_live(self):
        self.stackedWidget.setCurrentIndex(0)

    def show_menu_theo_doi_camera_anh(self):
        self.stackedWidget.setCurrentIndex(3)

    def show_menu_theo_doi_camera_live(self):
        self.stackedWidget.setCurrentIndex(4)

    def show_thong_tin(self):
        self.display_dialog = QDialog()
        self.display_thong_tin = Dialog_ThongTin()
        self.display_thong_tin.setupUi(self.display_dialog)
        self.display_dialog.exec()


    def show_help(self):
        self.display_dialog = QDialog()
        self.display_help = Dialog_Help()
        self.display_help.setupUi(self.display_dialog)
        self.display_dialog.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
