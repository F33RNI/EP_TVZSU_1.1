"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import sys
import threading

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph.opengl as gl
import qimage2ndarray
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from cv2 import aruco
from ubidots import ApiClient


class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    """
    This class creates custom text element for GLViewWidget object
    """

    def __init__(self, gl_view_widget, x, y, z, text, color=QColor(0, 255, 0)):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.gl_view_widget = gl_view_widget
        self.x = x
        self.y = y
        self.z = z
        self.text = text
        self.color = color

    def set_x(self, x):
        self.x = x
        self.update()

    def set_y(self, y):
        self.y = y
        self.update()

    def set_z(self, z):
        self.z = z
        self.update()

    def set_text(self, text):
        self.text = text
        self.update()

    def set_color(self, color):
        self.color = color
        self.update()

    def update_object(self):
        self.update()

    def paint(self):
        self.gl_view_widget.qglColor(self.color)
        self.gl_view_widget.renderText(self.x, self.y, self.z, self.text)


class Window(QMainWindow):
    """
    This this a main class
    """

    def __init__(self):
        super(Window, self).__init__()

        # Load .ui file
        uic.loadUi('EP_1_1.ui', self)

        # Setup
        self.camera_matrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
        self.camera_distortion = np.loadtxt('cameraDistortion.txt', delimiter=',')
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = 10

        # System variables
        self.source_capture = None
        self.loop_running = False
        self.opengl_updater_running = False
        self.aruco_dictionary = None
        self.points_surface = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]))
        self.text_objects = []
        self.text_objects_last = []
        self.points_line = gl.GLLinePlotItem()
        self.ubidots_points = []
        self.ubidots_points_text = []
        self.ubidots_points_line = gl.GLLinePlotItem()
        self.ubidots_points_surface = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]))
        self.marker_size = 0

        # Connect buttons
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_retrieve_data.clicked.connect(self.retrieve_data)

        # Update and show OpenGL view
        self.openGLWidget.addItem(gl.GLAxisItem())
        self.openGLWidget.addItem(gl.GLGridItem())
        self.openGLWidget.addItem(self.points_surface)
        self.openGLWidget.addItem(self.points_line)
        for i in range(10):
            self.ubidots_points_text.append(CustomTextItem(self.openGLWidget, 0, 0, 0, '', QColor(0, 0, 0, 0)))
            self.openGLWidget.addItem(self.ubidots_points_text[i])
        self.openGLWidget.addItem(self.ubidots_points_surface)
        self.openGLWidget.addItem(self.ubidots_points_line)

        # Add camera object
        camera_lines_factor = 5
        camera_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1 * camera_lines_factor, 1 * camera_lines_factor, -0.56 * camera_lines_factor]]),
            color=[1, 0, 0, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1 * camera_lines_factor, 1 * camera_lines_factor, 0.56 * camera_lines_factor]]),
            color=[1, 0.5, 0, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1 * camera_lines_factor, -1 * camera_lines_factor, -0.56 * camera_lines_factor]]),
            color=[1, 1, 0, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1 * camera_lines_factor, -1 * camera_lines_factor, 0.56 * camera_lines_factor]]),
            color=[0, 1, 0, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(pos=np.array(
            [[1 * camera_lines_factor, 1 * camera_lines_factor, -0.56 * camera_lines_factor],
             [1 * camera_lines_factor, -1 * camera_lines_factor, -0.56 * camera_lines_factor]]),
            color=[0, 0, 1, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(pos=np.array(
            [[1 * camera_lines_factor, 1 * camera_lines_factor, 0.56 * camera_lines_factor],
             [1 * camera_lines_factor, -1 * camera_lines_factor, 0.56 * camera_lines_factor]]),
            color=[0, 0, 1, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(pos=np.array(
            [[1 * camera_lines_factor, 1 * camera_lines_factor, -0.56 * camera_lines_factor],
             [1 * camera_lines_factor, 1 * camera_lines_factor, 0.56 * camera_lines_factor]]),
            color=[0.5, 0, 1, 1])
        self.openGLWidget.addItem(camera_line)
        camera_line = gl.GLLinePlotItem(pos=np.array(
            [[1 * camera_lines_factor, -1 * camera_lines_factor, -0.56 * camera_lines_factor],
             [1 * camera_lines_factor, -1 * camera_lines_factor, 0.56 * camera_lines_factor]]),
            color=[0.5, 0, 1, 1])
        self.openGLWidget.addItem(camera_line)

        # Timer for removing / appending new text objects to the openGLWidget
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_text)
        self.timer.start(100)

        # Show GUI
        self.show()

    def update_text(self):
        if not self.text_objects_last == self.text_objects:
            # Remove and then add all items if they changed
            if self.text_objects_last is not None:
                for text_object in self.text_objects_last:
                    self.openGLWidget.removeItem(text_object)

            if self.text_objects is not None:
                for text_object in self.text_objects:
                    self.openGLWidget.addItem(text_object)

        self.text_objects_last = self.text_objects.copy()

    # noinspection PyBroadException
    def retrieve_data(self):
        try:
            ubidots_api = ApiClient(token=self.api_key.text())
            variables = ubidots_api.get_variables()
            variables_str = str(variables).replace('[', '').replace(']', '').replace(' ', '').split(',')

            self.ubidots_points = []
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_0')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_1')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_2')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_3')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_4')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_5')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_6')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_7')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_8')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
            self.ubidots_points.append(list(map(float, str(
                variables[variables_str.index('point_9')].get_values(1)[0]['context']['position']).replace('\"',
                                                                                                           '').split(
                ','))))
        except:
            print('Error reading Ubidots data!')

    def start(self):
        # Source capture
        if self.radio_source_video.isChecked():
            # Source from video file
            self.source_capture = cv2.VideoCapture(self.line_source_video.text())
        else:
            # Source from camera
            if self.check_source_camera_dshow.isChecked():
                self.source_capture = cv2.VideoCapture(self.spin_source_camera_id.value(), cv2.CAP_DSHOW)
            else:
                self.source_capture = cv2.VideoCapture(self.spin_source_camera_id.value())
            self.source_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.source_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.source_capture.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.source_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.source_capture.set(cv2.CAP_PROP_FOCUS, 0)

        # Define dictionary
        self.aruco_dictionary = cv2.aruco.Dictionary_get(self.spin_dictionary.value())

        # Define marker size
        self.marker_size = self.spin_marker_size.value()

        # Start main cycle as thread
        self.loop_running = True
        thread = threading.Thread(target=self.opencv_loop)
        thread.start()

    def stop(self):
        # Stop main cycle
        self.loop_running = False

        # Release captures
        if self.source_capture is not None:
            self.source_capture.release()

        # Destroy OpenCV windows
        cv2.destroyAllWindows()

    def opencv_loop(self):
        while self.loop_running:
            # Read both frames
            source_ret, source_frame = self.source_capture.read()

            # Check for both frames
            if source_ret:
                destination_frame = source_frame.copy()
                gray_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)

                # Detect ARUCO markers
                markers_corners, marker_ids, rejected_candidates = \
                    cv2.aruco.detectMarkers(gray_frame, self.aruco_dictionary,
                                            parameters=self.parameters,
                                            cameraMatrix=self.camera_matrix,
                                            distCoeff=self.camera_distortion)

                # Sort markers
                if marker_ids is not None:
                    ids_array = np.array([item[0] for item in marker_ids])
                    ids_permut = ids_array.argsort()
                    marker_ids = marker_ids[ids_permut]
                    markers_corners_sorted = markers_corners.copy()
                    for i in range(len(ids_permut)):
                        markers_corners_sorted[i] = markers_corners[ids_permut[i]]
                    markers_corners = markers_corners_sorted

                # Draw detected markers
                aruco.drawDetectedMarkers(destination_frame, markers_corners, marker_ids)

                estimates_points = []
                if marker_ids is not None:
                    # Estimate markers position
                    for marker_corners in markers_corners:
                        ret = aruco.estimatePoseSingleMarkers(marker_corners, self.marker_size, self.camera_matrix,
                                                              self.camera_distortion)

                        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                        marker_x = tvec[0]
                        marker_y = tvec[1]
                        marker_z = tvec[2]
                        estimates_points.append([marker_z, -marker_x, -marker_y])

                    # Create or remove text objects
                    while len(self.text_objects) < len(marker_ids):
                        self.text_objects.append(CustomTextItem(self.openGLWidget, 0, 0, 0, '', QColor(255, 0, 0)))
                    while len(self.text_objects) > len(marker_ids):
                        del self.text_objects[-1]

                    if len(estimates_points) > 1:
                        # Calculate color map
                        data_list = np.array(np.array(range(len(estimates_points))))
                        cmap = plt.get_cmap('hsv')
                        min_data = np.min(data_list)
                        max_data = np.max(data_list)
                        rgba_img = cmap(1.0 - (data_list - min_data) / (max_data - min_data))

                        # Draw lines
                        self.points_line.setData(pos=np.array(estimates_points), color=rgba_img)
                    else:
                        rgba_img = np.array([[1, 0, 0, 1]])
                        self.points_line.setData(pos=np.array([[0, 0, 0], [0, 0, 0]]), color=[0, 0, 0, 0])

                    self.points_surface.setData(pos=np.array(estimates_points), color=rgba_img)

                    for i in range(len(marker_ids)):
                        # Update text's position, ID and color
                        self.text_objects[i].set_x(estimates_points[i][0])
                        self.text_objects[i].set_y(estimates_points[i][1])
                        self.text_objects[i].set_z(estimates_points[i][2])
                        self.text_objects[i].set_text(str(marker_ids[i][0]))
                        color = rgba_img[i]
                        self.text_objects[i].set_color(QColor(color[0] * 255, color[1] * 255, color[2] * 255))
                else:
                    # Draw 0, 0, 0 if no markers found
                    self.text_objects = []
                    self.points_surface.setData(pos=np.array([[0, 0, 0]]), color=[0, 0, 0, 0])
                    self.points_line.setData(pos=np.array([[0, 0, 0], [0, 0, 0]]), color=[0, 0, 0, 0])

                if self.ubidots_points is not None and len(self.ubidots_points) > 0:
                    self.ubidots_points_surface.setData(pos=np.array(self.ubidots_points), color=[1, 1, 1, 1])
                    self.ubidots_points_line.setData(pos=np.array(self.ubidots_points), color=[1, 1, 1, 1])
                    for i in range(len(self.ubidots_points)):
                        self.ubidots_points_text[i].set_text('IoT_' + str(i))
                        self.ubidots_points_text[i].set_x(self.ubidots_points[i][0])
                        self.ubidots_points_text[i].set_y(self.ubidots_points[i][1])
                        self.ubidots_points_text[i].set_z(self.ubidots_points[i][2])
                        self.ubidots_points_text[i].set_color(QColor(255, 255, 255))

                # Show image
                self.image_original.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                    cv2.cvtColor(imutils.resize(destination_frame, height=self.image_original.height()),
                                 cv2.COLOR_BGR2RGB))))
            else:
                print('Error reading frames!')
                self.stop()
                break

            # Press q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopping a cycle')
                self.stop()
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    win = Window()
    sys.exit(app.exec_())
