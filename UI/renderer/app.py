from flask import Flask, render_template, Response, request, jsonify
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

dx, dy, scroll = 0, 0, 0
curr_dist = 0

# reset dx dy at the start of rotation action
@app.route('/mousedown', methods=['GET'])
def on_mousedown():
    global dx, dy
    dx, dy = 0, 0
    print(dx, dy)
    return jsonify({"result": "success"})

# accumulate dx, dy during rotation action
@app.route('/mousemove', methods=['POST'])
def on_mousemove():
    global dx, dy
    dx += float(request.json.get('dx'))
    dy += float(request.json.get('dy'))
    print("\tdx dy:", dx, dy)
    return jsonify({"dx": dx, "dy": dy})

# accumulate dscroll during rotation action
@app.route('/wheelscroll', methods=['POST'])
def on_wheelscroll():
    global dscroll
    scroll += float(request.json.get('dscroll'))
    print("\tdscroll:", scroll)
    return jsonify({"dscroll": scroll})

# @app.route('/image_update', methods=['POST'])


def index():
    return render_template('index.html')

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
    

@app.route('/')
def index():
    return render_template('index.html')

counter = 0
def gen():
    global counter
    while True:
        time.sleep(1/60)
        img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
        ret, img_enc = cv2.imencode('.jpg', img)
        jpeg = img_enc.tobytes()
        counter += 1
        # print(ret, counter)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 1. let event trigger routes constantly (trying)
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
        

if __name__ == '__main__':
    # send_img_thread = setInterval(1/30, send_img_updates)
    # send_img_thread.start()
    render_thread = threading.Thread(target=render_checkerboard)
    # render_thread.start()
    socketio.run(app ,port=5000, debug=True)
    
