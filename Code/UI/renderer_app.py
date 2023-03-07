import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import numpy as np
from Other.utility_functions import str2bool
from PyQt5.QtCore import QSize, Qt, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QStackedLayout, \
    QComboBox, QSlider, QFileDialog
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QEvent, Qt
from Code.renderer import Camera, Scene, TransferFunction, RawData
from Code.UI.utils import Arcball, torch_float_to_numpy_uint8
from Code.Models.options import load_options
from Code.Models.models import load_model
from typing import List
import imageio.v3 as imageio

# For locking renderer actions
render_mutex = QMutex()

# Directories for things of interest
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
data_folder = os.path.join(project_folder_path, "Data")
savedmodels_folder = os.path.join(project_folder_path, "SavedModels")
tf_folder = os.path.join(project_folder_path, "Colormaps")


    
class MainWindow(QMainWindow):
    
    loading_model = True
    last_img = np.zeros([1,1,3],dtype=np.uint8)
    updates_per_second = pyqtSignal(float)
    frame_time = pyqtSignal(float)
    vram_use = pyqtSignal(float)
    status_text_update = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        project_folder_path = os.path.dirname(os.path.abspath(__file__))
        project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
        
        self.setWindowTitle("Neural Volume Renderer") 
            
        # Find all available models/colormaps        
        self.available_models = os.listdir(savedmodels_folder)
        self.available_tfs = os.listdir(tf_folder)
        self.available_data = os.listdir(data_folder)
        
        # Full screen layout
        layout = QHBoxLayout()        
        
        # Render area
        self.render_view = QLabel()   
        self.render_view.mousePressEvent = self.mouseClicked
        self.render_view.mouseReleaseEvent = self.mouseReleased
        self.render_view.mouseMoveEvent = self.mouseMove    
        
        # Settings area
        self.settings_ui = QVBoxLayout()       
         
        self.load_box = QHBoxLayout()  
        self.load_box.addWidget(QLabel("Load from: "))
        self.load_from_dropdown = QComboBox()        
        self.load_from_dropdown.addItems(["Model", "Data"])   
        self.load_from_dropdown.currentTextChanged.connect(self.data_box_update) 
        self.load_box.addWidget(self.load_from_dropdown)
           
        self.datamodel_box = QHBoxLayout()  
        self.datamodel_box.addWidget(QLabel("Model/data: "))
        self.models_dropdown = self.load_models_dropdown()
        self.models_dropdown.currentTextChanged.connect(self.load_model)
        self.datamodel_box.addWidget(self.models_dropdown)
        
        self.tf_box = QHBoxLayout()  
        self.tf_box.addWidget(QLabel("Colormap:"))
        self.tfs_dropdown = self.load_colormaps_dropdown()
        self.tfs_dropdown.currentTextChanged.connect(self.load_tf)
        self.tf_box.addWidget(self.tfs_dropdown)
        
        self.batch_slider_box = QHBoxLayout()      
        self.batch_slider_label = QLabel("Batch size (2^x): 20")  
        self.batch_slider_box.addWidget(self.batch_slider_label)
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setMinimum(18)
        self.batch_slider.setMaximum(25)
        self.batch_slider.setValue(20)   
        self.batch_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_slider.setTickInterval(1)  
        self.batch_slider.valueChanged.connect(self.change_batch_visual)
        self.batch_slider.sliderReleased.connect(self.change_batch)
        self.batch_slider_box.addWidget(self.batch_slider)   
        
        self.spp_slider_box = QHBoxLayout()      
        self.spp_slider_label = QLabel("Samples per ray: 256")  
        self.spp_slider_box.addWidget(self.spp_slider_label)
        self.spp_slider = QSlider(Qt.Horizontal)
        self.spp_slider.setMinimum(7)
        self.spp_slider.setMaximum(13)
        self.spp_slider.setValue(8)   
        self.spp_slider.setTickPosition(QSlider.TicksBelow)
        self.spp_slider.setTickInterval(1)  
        self.spp_slider.valueChanged.connect(self.change_spp_visual)
        self.spp_slider.sliderReleased.connect(self.change_spp)
        self.spp_slider_box.addWidget(self.spp_slider)   
        
        self.view_xy_button = QPushButton("reset view to xy-plane")
        self.view_xy_button.setFixedHeight(25)
        self.view_xy_button.clicked.connect(lambda: self.render_worker.view_xy.emit())
        
        self.transfer_function_box = QVBoxLayout()
        self.transfer_function_bg = QWidget()
        self.transfer_function_bg.setMinimumHeight(150)
        self.transfer_function_bg.setStyleSheet("background-color: white; border: 1px solid black;")
        self.transfer_function_box.addWidget(self.transfer_function_bg)
                        
        self.status_text = QLabel("") 
        self.memory_use_label = QLabel("VRAM use: -- GB") 
        self.update_framerate_label = QLabel("Update framerate: -- fps") 
        self.frame_time_label = QLabel("Last frame time: -- sec.") 
        self.status_text_update.connect(self.update_status_text)
        self.vram_use.connect(self.update_vram)
        self.updates_per_second.connect(self.update_updates)
        self.frame_time.connect(self.update_frame_time)
        self.save_img_button = QPushButton("Save image")
        self.save_img_button.clicked.connect(self.save_img)
        
        self.settings_ui.addLayout(self.load_box)
        self.settings_ui.addLayout(self.datamodel_box)
        self.settings_ui.addLayout(self.tf_box)
        self.settings_ui.addLayout(self.batch_slider_box)
        self.settings_ui.addLayout(self.spp_slider_box)
        self.settings_ui.addWidget(self.view_xy_button)
        self.settings_ui.addLayout(self.transfer_function_box)
        self.settings_ui.addStretch()
        self.settings_ui.addWidget(self.status_text)
        self.settings_ui.addWidget(self.memory_use_label)
        self.settings_ui.addWidget(self.update_framerate_label)
        self.settings_ui.addWidget(self.frame_time_label)
        self.settings_ui.addWidget(self.save_img_button)
        
        # UI full layout        
        layout.addWidget(self.render_view, stretch=4)
        layout.addLayout(self.settings_ui, stretch=1)        
        layout.setContentsMargins(0,0,10,10)
        layout.setSpacing(20)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)
        
        # Set up render thread
        self.render_thread = QThread()
        self.render_worker = RendererThread(self)        
        self.load_renderer()
        
        # Variables to use for interaction
        self.rotating = False
        self.panning = False
        self.last_x = None
        self.last_y = None
   
   
    def save_img(self):
        folderpath,_ = QFileDialog.getSaveFileName(self, 'Select Save Location')
        if ".jpg" not in folderpath and ".png" not in folderpath:
            folderpath = folderpath + ".png"
        
        print(f"Saving to {folderpath}")
        
        imageio.imwrite(folderpath, self.last_img)
        
        
    def update_status_text(self, val):
        self.status_text.setText(f"{val}")
        
    def update_vram(self, val):
        self.memory_use_label.setText(f"VRAM use: {val:0.02f} GB")
        
    def update_updates(self, val):
        self.update_framerate_label.setText(f"Update framerate: {val:0.02f} fps")
    
    def update_frame_time(self, val):
        self.frame_time_label.setText(f"Last frame time: {val:0.02f} sec.")
     
    def data_box_update(self, s):
        self.loading_model = "Model" in s
        if(self.loading_model):
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_models)
        else:            
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_data)
       
    def load_models_dropdown(self):
        dropdown = QComboBox()        
        dropdown.addItems(self.available_models)        
        return dropdown

    def load_colormaps_dropdown(self):
        dropdown = QComboBox()        
        dropdown.addItems(self.available_tfs)        
        return dropdown
                    
    def load_renderer(self):    
        self.render_worker.moveToThread(self.render_thread)                
        self.render_worker.progress.connect(self.set_render_image)      
        self.render_thread.started.connect(self.render_worker.run)
                
        self.render_thread.start()
        
    def finish_worker(self):
        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.finished.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self.render_worker.deleteLater)
    
    def resizeEvent(self, event):
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        self.render_worker.resize.emit(w,h)
        QMainWindow.resizeEvent(self, event)
     
    def mouseClicked(self, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.startRotate()
            if event.button() == Qt.MiddleButton:
                self.startPan()
                             
    def startPan(self):
        self.panning = True
        
    def startRotate(self):
        self.rotating = True
  
    def load_model(self, s):
        if s == "":
            return
        self.status_text_update.emit(f"Loading model {s}...")
        if(self.loading_model):
            self.render_worker.load_new_model.emit(s)
        else:
            self.render_worker.load_new_data.emit(s)
        self.status_text_update.emit("")
        
    def load_tf(self, s):
        print(f"TF changed {s}")
        self.render_worker.change_transfer_function.emit(s)
    
    def change_batch_visual(self):
        val = int(self.batch_slider.value())
        self.batch_slider_label.setText(f"Batch size (2^x): {val}")
        
    def change_batch(self):
        val = int(self.batch_slider.value())
        self.batch_slider_label.setText(f"Batch size (2^x): {val}")
        self.render_worker.change_batch_size.emit(val)
     
    def change_spp_visual(self):
        val = int(self.spp_slider.value())
        self.spp_slider_label.setText(f"Samples per ray: {2**val}")
        
    def change_spp(self):
        val = int(self.spp_slider.value())
        self.spp_slider_label.setText(f"Samples per ray: {2**val}")
        self.render_worker.change_spp.emit(2**val)
     
    def mouseReleased(self, event):
        if event.type() == QEvent.MouseButtonRelease:
            if(event.button()) == Qt.LeftButton:
                self.endRotate()
            if(event.button()) == Qt.MiddleButton:
                self.endPan()
    
    def endPan(self):
        self.panning = False
        self.last_x = None
        self.last_y = None
        
    def endRotate(self):
        self.rotating = False
        self.last_x = None
        self.last_y = None
        
    def mouseMove(self, event):
        
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        x = (event.x() / w) * 2 - 1
        y = -((event.y() / h) * 2 - 1)
        
        if(self.last_x is None or self.last_y is None):
            self.last_x = x
            self.last_y = y
            return
        if(x is None or y is None):
            print("x or y was None!")
            return
        
        if self.rotating:
            self.render_worker.rotate.emit(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
        if self.panning:
            self.render_worker.pan.emit(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
    
    def wheelEvent(self,event):
        scroll = event.angleDelta().y()/120
        self.render_worker.zoom.emit(-scroll)
         
    def set_render_image(self, img:np.ndarray):  
        height, width, channel = img.shape
        self.last_img = img
        bytesPerLine = channel * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.render_view.setPixmap(QPixmap(qImg))
        
class RendererThread(QObject):
    progress = pyqtSignal(np.ndarray)
    rotate = pyqtSignal(float, float, float, float)
    pan = pyqtSignal(float, float, float, float)
    zoom = pyqtSignal(float)
    resize = pyqtSignal(int, int)
    change_spp = pyqtSignal(int)
    load_new_model = pyqtSignal(str)
    load_new_data = pyqtSignal(str)
    change_transfer_function = pyqtSignal(str)
    change_batch_size = pyqtSignal(int)
    change_opacity_controlpoints = pyqtSignal(np.ndarray)
    view_xy = pyqtSignal()
    
    def __init__(self, parent=None):
        super(RendererThread, self).__init__()
        self.parent = parent
        
        # Local variables needed to keep track of 
        self.device = "cuda:0"
        self.spp = 256
        self.batch_size = 2**20
        self.resolution = [256,256]
        self.full_shape = [1,1,1]
        self.opt = None  
        self.model = None
        self.camera = None
        self.update_rate = []
        self.frame_rate = []
        self.tf = TransferFunction(self.device)      
        
        self.initialize_model()  
        self.parent.status_text_update.emit("Initializing scene...") 
        self.initialize_camera()        
        self.scene = Scene(self.model, self.camera, 
                           self.full_shape, self.resolution, 
                           self.batch_size, self.spp, 
                           self.tf, self.device)
        self.scene.on_setting_change()
        
        # Set up events
        self.pan.connect(self.do_pan)
        self.rotate.connect(self.do_rotate)
        self.zoom.connect(self.do_zoom)
        self.resize.connect(self.do_resize)
        self.change_batch_size.connect(self.do_change_batch_size)
        self.change_spp.connect(self.do_change_spp)
        self.change_transfer_function.connect(self.do_change_transfer_function)
        self.load_new_model.connect(self.do_change_model)
        self.load_new_data.connect(self.do_change_data)
        self.view_xy.connect(self.do_view_xy)
        self.parent.status_text_update.emit(f"")
        
    def run(self):
        last_spot = 0
        current_spot = 0
        while True:
            current_spot = self.scene.current_order_spot
            render_mutex.lock()
            if(self.scene.current_order_spot == 0):
                frame_start_time = time.time()
            update_start_time = time.time()
            self.scene.one_step_update()
            if(current_spot < len(self.scene.render_order)):
                update_time = time.time() - update_start_time
                self.update_rate.append(update_time)
            if(current_spot == len(self.scene.render_order) and 
                last_spot < current_spot):
                    frame_time = time.time() - frame_start_time
                    self.frame_rate.append(frame_time)
                    last_frame_time = self.frame_rate[-1]
                    self.parent.frame_time.emit(last_frame_time)
            img = torch_float_to_numpy_uint8(self.scene.temp_image)
            render_mutex.unlock()

            self.progress.emit(img)
            self.parent.vram_use.emit(self.scene.get_mem_use())
            
            if(len(self.update_rate) > 20):
                self.update_rate.pop(0)
            if(len(self.frame_rate) > 5):
                self.frame_rate.pop(0)
            if(len(self.update_rate) > 0):
                average_update_fps = 1/np.array(self.update_rate).mean()
                self.parent.updates_per_second.emit(average_update_fps)
            last_spot = current_spot

    def do_resize(self, w, h):
        render_mutex.lock()
        self.scene.image_resolution = [h,w]
        self.scene.on_resize()
        self.update_rate = []
        self.frame_rate = []
        render_mutex.unlock()
        
    def do_rotate(self, last_x, last_y, x, y):
        render_mutex.lock()
        self.camera.mouse_start = np.array([last_x, last_y])
        self.camera.mouse_curr = np.array([x, y])
        self.camera.rotate()
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_pan(self, last_x, last_y, x, y):
        render_mutex.lock()
        mouse_start = np.array([last_x, last_y])
        mouse_curr = np.array([x, y])
        mouse_delta = mouse_curr - mouse_start
        self.camera.pan(mouse_delta)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_zoom(self, zoom):
        render_mutex.lock()
        self.camera.zoom(zoom)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_view_xy(self):
        render_mutex.lock()
        self.camera.reset_view_xy()
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_transfer_function(self, s):
        render_mutex.lock()
        self.tf.loadColormap(s)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_batch_size(self, b):
        render_mutex.lock()
        self.batch_size = 2**b
        self.scene.batch_size = 2**b
        self.scene.on_resize()
        render_mutex.unlock()

    def do_change_spp(self, b):
        render_mutex.lock()
        self.spp = b
        self.scene.spp = b
        self.scene.on_resize()
        render_mutex.unlock()

    def initialize_camera(self):
        aabb = np.array([0.0, 0.0, 0.0, 
                        self.full_shape[0]-1,
                        self.full_shape[1]-1,
                        self.full_shape[2]-1], 
                        dtype=np.float32)
        self.camera = Arcball(
            scene_aabb=aabb,
            coi=np.array(
                [aabb[3]/2, 
                aabb[4]/2, 
                aabb[5]/2], 
                dtype=np.float32), 
            dist=(aabb[3]**2 + \
                aabb[4]**2 + \
                aabb[5]**2)**0.5,
            fov=60.0
        )
    
    def initialize_model(self):
        first_model = os.listdir(savedmodels_folder)[0]
        print(f"Loading model {first_model}")
        self.parent.status_text_update.emit(f"Loading model {first_model}...")
        self.opt = load_options(os.path.abspath(os.path.join('SavedModels', first_model)))
        self.full_shape = self.opt['full_shape']
        self.model = load_model(self.opt, self.device).to(self.device)
        self.model.eval()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        self.tf.set_minmax(self.model.min(), self.model.max())  
        self.parent.status_text_update.emit(f"")   
    
    def do_change_model(self, s):
        render_mutex.lock()
        print(f"Loading model {s}")
        self.opt = load_options(os.path.abspath(os.path.join('SavedModels', s)))
        self.full_shape = self.opt['full_shape']
        self.model = load_model(self.opt, self.device).to(self.device)
        self.model.eval()
        self.tf.set_minmax(self.model.min(), self.model.max())        
        self.scene.model = self.model
        self.scene.set_aabb([ 
            self.full_shape[2]-1,
            self.full_shape[1]-1,
            self.full_shape[0]-1
            ])
        self.camera.update_coi(
            np.array([ 
            self.full_shape[2]/2,
            self.full_shape[1]/2,
            self.full_shape[0]/2
            ], dtype=np.float32)            
        )
        self.camera.update_dist((self.full_shape[0]**2 + \
                self.full_shape[1]**2 + \
                self.full_shape[2]**2)**0.5)
        #self.scene.precompute_occupancy_grid()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        self.scene.on_setting_change()
        render_mutex.unlock()
    
    def do_change_data(self, s):
        render_mutex.lock()
        print(f"Loading model {s}")
        self.model = RawData(s, self.device)
        self.full_shape = self.model.shape
        self.model.eval()
        self.tf.set_minmax(self.model.min(), self.model.max())        
        self.scene.model = self.model
        self.scene.set_aabb([ 
            self.full_shape[2]-1,
            self.full_shape[1]-1,
            self.full_shape[0]-1
            ])
        self.camera.update_coi(
            np.array([ 
            self.full_shape[2]/2,
            self.full_shape[1]/2,
            self.full_shape[0]/2
            ], dtype=np.float32)            
        )
        self.camera.update_dist((self.full_shape[0]**2 + \
                self.full_shape[1]**2 + \
                self.full_shape[2]**2)**0.5)
        #self.scene.precompute_occupancy_grid()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        self.scene.on_setting_change()
        render_mutex.unlock()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()    
    window.show()
    sys.exit(app.exec())