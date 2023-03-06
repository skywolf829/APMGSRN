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
    QComboBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from Code.renderer import Camera, Scene, TransferFunction
from Code.UI.utils import Arcball, torch_float_to_numpy_uint8
from Code.Models.options import load_options
from Code.Models.models import load_model

# For locking renderer actions
render_mutex = QMutex()

# Directories for things of interest
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
data_folder = os.path.join(project_folder_path, "Data")
savedmodels_folder = os.path.join(project_folder_path, "SavedModels")
tf_folder = os.path.join(project_folder_path, "Colormaps")
    
class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        project_folder_path = os.path.dirname(os.path.abspath(__file__))
        project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
        
        self.setWindowTitle("My App") 
            
        # Find all available models/colormaps        
        self.available_models = os.listdir(savedmodels_folder)
        self.available_tfs = os.listdir(tf_folder)
        
        # Full screen layout
        layout = QHBoxLayout()        
        
        # Render area
        self.render_view = QLabel()   
        self.render_view.mousePressEvent = self.startRotate
        self.render_view.mouseReleaseEvent = self.endRotate
        self.render_view.mouseMoveEvent = self.doRotate    
        
        # Settings area
        self.settings_ui = QVBoxLayout()        
        self.models_dropdown = self.load_models_dropdown()
        self.models_dropdown.currentTextChanged.connect(self.load_model)
        self.tfs_dropdown = self.load_colormaps_dropdown()
        self.tfs_dropdown.currentTextChanged.connect(self.load_tf)
        self.settings_ui.addWidget(self.models_dropdown)
        self.settings_ui.addWidget(self.tfs_dropdown)
        
        # UI full layout        
        layout.addWidget(self.render_view)
        layout.addLayout(self.settings_ui)        
        layout.setContentsMargins(0,0,10,10)
        layout.setSpacing(20)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)
        
        # Set up render thread
        self.render_thread = QThread()
        self.render_worker = RendererThread()        
        self.load_renderer()
        
        # Variables to use for interaction
        self.rotating = False
        self.last_x = None
        self.last_y = None
        
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
        #self.start_polling()
        
    def finish_worker(self):
        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.finished.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self.render_worker.deleteLater)
    
    def resizeEvent(self, event):
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        self.render_worker.resize.emit(w,h)
        QMainWindow.resizeEvent(self, event)
        
    def startRotate(self, QMouseEvent):
        self.rotating = True
  
    def load_model(self, s):
        print(f"Model changed {s}")
        self.render_worker.load_new_model.emit(s)
        
    def load_tf(self, s):
        print(f"TF changed {s}")
        self.render_worker.change_transfer_function.emit(s)
        
    def endRotate(self, event):
        self.rotating = False
        self.last_x = None
        self.last_y = None
        
    def doRotate(self, event):
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        x = (event.x() / w) * 2 - 1
        y = -((event.y() / h) * 2 - 1)
        
        if(self.last_x is None or self.last_y is None):
            self.last_x = x
            self.last_y = y
            return
        
        if self.rotating:
            self.render_worker.rotate.emit(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
    
    def wheelEvent(self,event):
        scroll = event.angleDelta().y()/120
        self.render_worker.zoom.emit(-scroll)
         
    def set_render_image(self, img:np.ndarray):  
        height, width, channel = img.shape
        bytesPerLine = channel * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.render_view.setPixmap(QPixmap(qImg))
        
class RendererThread(QObject):    
    progress = pyqtSignal(np.ndarray)
    rotate = pyqtSignal(float, float, float, float)
    zoom = pyqtSignal(float)
    resize = pyqtSignal(int, int)
    change_spp = pyqtSignal(int)
    load_new_model = pyqtSignal(str)
    change_transfer_function = pyqtSignal(str)
    change_batch_size = pyqtSignal(int)
    
    def __init__(self):
        super(RendererThread, self).__init__()
        
        # Local variables needed to keep track of 
        self.device = "cuda:0"
        self.batch_size = 2**20
        self.spp = 256
        self.resolution = [256,256]
        self.full_shape = [1,1,1]
        self.opt = None  
        self.model = None
        self.camera = None
        self.tf = TransferFunction(self.device)      
        
        self.initialize_model()   
        self.initialize_camera()
        
        self.scene = Scene(self.model, self.camera, 
                           self.full_shape, self.resolution, 
                           self.batch_size, self.spp, 
                           self.tf, self.device)
        self.scene.on_setting_change()
        
        # Set up events
        self.rotate.connect(self.do_rotate)
        self.zoom.connect(self.do_zoom)
        self.resize.connect(self.do_resize)
        self.change_transfer_function.connect(self.do_change_transfer_function)
        self.load_new_model.connect(self.do_change_model)
        
    def run(self):
        while True:
            render_mutex.lock()
            self.scene.one_step_update()
            img = torch_float_to_numpy_uint8(self.scene.temp_image)
            render_mutex.unlock()
            self.progress.emit(img)
            
    def do_resize(self, w, h):
        render_mutex.lock()
        self.scene.image_resolution = [h,w]
        self.scene.on_resize()
        render_mutex.unlock()
        
    def do_rotate(self, last_x, last_y, x, y):
        render_mutex.lock()
        self.camera.mouse_start = np.array([last_x, last_y])
        self.camera.mouse_curr = np.array([x, y])
        self.camera.rotate()
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
    
    def do_zoom(self, zoom):
        render_mutex.lock()
        self.camera.zoom(zoom)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_transfer_function(self, s):
        render_mutex.lock()
        self.tf.loadColormap(s)
        self.scene.on_rotate_zoom_pan()
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
        self.opt = load_options(os.path.abspath(os.path.join('SavedModels', "Supernova_AMGSRN_small")))
        self.full_shape = self.opt['full_shape']
        self.model = load_model(self.opt, self.device).to(self.device)
        self.model.eval()
        self.tf.set_minmax(self.model.min(), self.model.max())     
    
    def do_change_model(self, s):
        render_mutex.lock()
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
        self.scene.on_setting_change()
        render_mutex.unlock()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()    
    window.show()
    sys.exit(app.exec())