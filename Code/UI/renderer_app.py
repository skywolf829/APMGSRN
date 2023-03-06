import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import numpy as np
from Other.utility_functions import str2bool
from PyQt5.QtCore import QSize, Qt, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from Code.renderer import Camera, Scene, TransferFunction
from Code.UI.utils import Arcball, torch_float_to_numpy_uint8
from Code.Models.options import load_options
from Code.Models.models import load_model

render_mutex = QMutex()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("My App")    
        self.render_view = QLabel()   
        self.render_view.mousePressEvent = self.startRotate
        self.render_view.mouseReleaseEvent = self.endRotate
        self.render_view.mouseMoveEvent = self.doRotate        
        
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        layout = QVBoxLayout()
        layout.addWidget(self.render_view)
        self.centralWidget.setLayout(layout)
        
        self.render_thread = QThread()
        self.render_worker = RendererThread()
        
        self.load_renderer()
        
        self.rotating = False
        self.last_x = None
        self.last_y = None
                
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
  
    # overriding the mouse release event
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
        
        device = "cuda:0"
        batch_size = 2**20
        spp = 256
        opt = load_options(os.path.abspath(os.path.join('SavedModels', "Supernova_AMGSRN_small")))
        full_shape = opt['full_shape']
        self.model = load_model(opt, device).to(device)
        self.model.eval()
        tf = TransferFunction(
            device=device,
            min_value=opt['data_min'],
            max_value=opt['data_max'],
            colormap=None,
        )
        aabb = np.array([0.0, 0.0, 0.0, 
                        full_shape[0]-1,
                        full_shape[1]-1,
                        full_shape[2]-1])
        dist = ((aabb[3]**2 + aabb[4]**2 + aabb[5]**2)**0.5)
        self.arcball = Arcball(
            scene_aabb=aabb,
            coi=(aabb[3:]/2), # camera lookat center of aabb,
            dist=float(dist),
            fov=60.
        )
        self.scene = Scene(self.model, self.arcball, 
                           full_shape, [256,256], 
                           batch_size, spp, tf, device)
        

        self.renderer_settings_changed = True
        self.current_render_image = np.zeros([256,256,3], dtype=np.uint8)   

        self.rotate.connect(self.do_rotate)
        self.zoom.connect(self.do_zoom)
        self.resize.connect(self.do_resize)
    
    def run(self):        
        while True:
            render_mutex.lock()
            self.scene.one_step_update()
            render_mutex.unlock()
            self.progress.emit(torch_float_to_numpy_uint8(self.scene.temp_image))
            
    def do_resize(self, w, h):
        render_mutex.lock()
        self.scene.image_resolution = [h,w]
        self.scene.on_resize()
        render_mutex.unlock()
        
    def do_rotate(self, last_x, last_y, x, y):
        render_mutex.lock()
        self.arcball.mouse_start = np.array([last_x, last_y])
        self.arcball.mouse_curr = np.array([x, y])
        self.arcball.rotate()
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
    
    def do_zoom(self, zoom):
        render_mutex.lock()
        self.arcball.zoom(zoom)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()    
    window.show()
    sys.exit(app.exec())