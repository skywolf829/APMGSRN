import os, sys
from flask import Flask, render_template, Response, request, jsonify
# from flask_socketio import SocketIO
import torch
import numpy as np
import base64
import time, threading
import cv2
# print(os.getcwd())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Code.Models.options import load_options
from Code.Models.models import load_model
from Code.renderer import Camera, Scene, TransferFunction
from Code.UI.utils import Arcball, torch_float_to_numpy_uint8

# from Code.renderer import render_checkerboard

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True

        
def render_background():
    '''
    continuously update img until it's done or img completely rendered
    '''
    global scene, arcball, scene_update_fulfilled, img

    while True:
        while not scene_update_fulfilled:
            # print(" @@@@@@@@@@@@@@@@@@@@@@@@rendering@@@@@@@@@@@@@@@@@@@@@")
            tmpimg, _ = scene.render_checkerboard(arcball)
            img = torch_float_to_numpy_uint8(tmpimg)
            # full image rendered, send the complete img
            scene_update_fulfilled = True
        time.sleep(1/30)
        
        
# communication flags
img_updated = False
scene_update_fulfilled = True

# model, scene, camera
device = "cuda:0"
batch_size = 2**23
hw = [128,128]
spp = 256


saved_path = os.path.abspath(os.path.join('SavedModels', "Supernova_AMGSRN_small"))
print(saved_path)
opt = load_options(saved_path)
print(opt)
# opt['device'] = "cpu"
opt['data_min'] = 0
opt['data_max'] = 1
full_shape = opt['full_shape']

print(opt['device'])

model = load_model(opt, device).to(device)
model.eval()
tf = TransferFunction(
    device=device,
    min_value=0.,
    max_value=1.,
    colormap=None,
)
scene = Scene(model, full_shape, hw, batch_size, spp, tf, device)
dist = (scene.scene_aabb.max(0)[0] - scene.scene_aabb.min(0)[0])*1.8
arcball = Arcball(
    scene_aabb=scene.scene_aabb.cpu().numpy(),
    coi=scene.scene_aabb.reshape(2,3).mean(0).cpu().numpy(), # camera lookat center of aabb,
    dist=float(dist),
    fov=60.
)
print("Camera distance to center of AABB:", dist)


# rendering parameter
print("rendering")
tmpimg, _ = scene.render_checkerboard(arcball)
print("rendering done")
print(tmpimg.max(), tmpimg.min())

img = torch_float_to_numpy_uint8(tmpimg)
print(img.max(), img.min())

render_thread = threading.Thread(target=render_background)
render_thread.setDaemon(True) # to properly termniate the app with interrupt
render_thread.start()




# reset dx dy at the start of rotation action
@app.route('/mousedown', methods=['POST'])
def on_mousedown():
    global scene_update_fulfilled
    x_start = float(request.json.get('x_start'))
    y_start = float(request.json.get('y_start'))
    arcball.mouse_start = np.array([x_start, y_start])
    scene_update_fulfilled = False
    return jsonify({"x_start": x_start, "y_start": y_start})

# accumulate dx, dy during rotation action
@app.route('/mousemove', methods=['POST'])
def on_mousemove():
    global scene_update_fulfilled
    x_curr = float(request.json.get('x_curr'))
    y_curr = float(request.json.get('y_curr'))
    arcball.mouse_curr = np.array([x_curr, y_curr])
    print("\tcurr_mouse ndc:", arcball.mouse_curr)
    arcball.rotate()
    scene_update_fulfilled = False
    return jsonify({"x_curr": x_curr, "y_curr": y_curr})

# accumulate dscroll during rotation action
@app.route('/wheelscroll', methods=['POST'])
def on_wheelscroll():
    global scene_update_fulfilled
    dscroll = float(request.json.get('dscroll'))
    arcball.zoom(dscroll)
    print("\tdscroll:", dscroll)
    scene_update_fulfilled = False
    return jsonify({"dscroll": dscroll})

@app.route('/')
def index():
    return render_template('index.html')

counter = 0
def gen():
    global counter, img
    while True:
        # img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
        # print(img.max(), img.min())
        ret, img_enc = cv2.imencode('.png', img)
        jpeg = img_enc.tobytes()
        counter += 1
        # print(ret, counter)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(1/30)
        

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # send_img_thread = setInterval(1/30, send_img_updates)
    # send_img_thread.start()
    
    # render_thread.join()
    app.run(port=5000, debug=True)
    # socketio.run(app ,port=5000, debug=True)
    
