from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import torch
import numpy as np
import time, threading
import cv2
# from Code.renderer import render_checkerboard

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, debug=True)

# communication flags
img_updated = False
scene_update_fulfilled = False

# rendering parameter
img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
azi = 0
polar = 90
dist = 500
tf = None

tmp_xy = 0

def render_checkerboard():
    '''
    continuously update img until it's done or img completely rendered
    '''
    n_passes = 8
    strides = int(n_passes**0.5) + 1
    while True:
        while not scene_update_fulfilled:
            for y in range(strides):
                for x in range(strides):
                    # render partial image and interpolate to full img
                    
                    
                    pass
            # full image rendered, send the complete img
            scene_update_fulfilled = True
        time.sleep(1)
    
counter = 0
@app.route('/')
def index():
    return render_template('index.html')
def gen():
    global counter
    while True:
        img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
        ret, img_enc = cv2.imencode('.png', img)
        jpeg = img_enc.tobytes()
        counter += 1
        print(counter)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + jpeg + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 1. let event trigger routes constantly
# 2. cache
# ....
# 3. websocket

# @socketio.on('scene_update')
# def handle_update(data):
#     # reset rendering with following infor
#     # drot = data['drot']
#     # ddist = data['ddist']
#     # tf = data['tf']
#     # update azi, polar, dist, and tf
#     scene_update_fulfilled = False

# counter = 0
# @socketio.on("request_img")
# def send_img_updates():
#     global img, counter
#     counter += 1
#     print(counter)
#     return
#     if not scene_update_fulfilled:
#         # if counter < 10:
#         #     img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
#         # else:
#         #     img = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
#         img += 1
#         counter += 1
#         print(counter)
#         # print(counter, img.shape)
#         socketio.emit("img_update", {
#             # "img": img.tobytes(),
#             "width": img.shape[1],
#             "height": img.shape[0],
#         })
        

# # https://stackoverflow.com/a/48709380
# class setInterval :
#     def __init__(self,interval,action) :
#         self.interval=interval
#         self.action=action
#         self.stopEvent=threading.Event()
#         thread=threading.Thread(target=self.__setInterval)
#         thread.start()

#     def __setInterval(self) :
#         nextTime=time.time()+self.interval
#         while not self.stopEvent.wait(nextTime-time.time()) :
#             nextTime+=self.interval
#             self.action()

#     def cancel(self) :
#         self.stopEvent.set()

if __name__ == '__main__':
    # send_img_thread = setInterval(1/30, send_img_updates)
    # send_img_thread.start()
    render_thread = threading.Thread(target=render_checkerboard)
    # render_thread.start()
    socketio.run(app ,port=8000, debug=True)
    
