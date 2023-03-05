
// import { io } from "socket.io-client";
const socket = io();
// 1 implementation I want to have to deal with the fps is, in the frontend, I have a flag variable to record if backend has returned an updated image. Then the logic is. We start with a rendering. As user start to interacts, one request is sent, and then no user interaction will be send to the backend, unless the previous requests are fulfilled (i.e. image rendered). Once the previous image is fulfilled,

/*
Canvas Interaction options:

mouseDown:
    start updating rendering with mouse movement.
mouseMove:
    update rendering: record current and previous x,y positions, and update rendering
mouseUp:
    stop updating rendering.

mouseWheel:
    scroll down: zoom in -> cam dist decrease
    scroll up   : zoom out -> cam dist increase
*/

function dxy_to_drotation(curr_dx, curr_dy){

}

function dscroll_to_ddist(curr_dscroll){
    
}
const canvas = document.createElement("canvas");
canvas.setAttribute("id", "canvas");
canvas.setAttribute("width", 512);
canvas.setAttribute("height", 512);

document.body.appendChild(canvas)
const ctx = canvas.getContext('2d');
// let imgdata = new ImageData(256,256);
let img = new Image();
//   <img src="{{url_for('video_feed')}}"></img>
img.src = "/video_feed";
ctx.drawImage(img, 0, 0);

setInterval(function() {
    console.log("Draw");
    ctx.drawImage(img, 100, 100);
}, 1000/60);


// let is_updating_rotation = false;
// let has_responded = false;
// let prev_x, prev_y; 
// let dx, dy, dscroll = 0;
// let tf = {
//     opacity: [],
//     color: [],
// };

// let fps = 60;
// let dt = 1000/fps;

// // function request_img() {
// //     console.log(1);
// //     socket.emit("request_img");
// // }
// // setInterval(request_img, dt);
// // image has been rendered,
// // update the image and start to listen to new camera movement
// let rgb = new Uint8ClampedArray();
// socket.on('img_update', function(data) {
//     // update image on canvas
//     console.log(data.width);
//     // take server out of the equation and test js refresh rate with js noise img
//     // imgdata = new ImageData(data.width, data.height);
//     // let rgb = new Uint8ClampedArray(data.img);
//     // imgdata.data.set(rgb);
//     // ctx.putImageData(imgdata, 0, 0);
// });

// // // send camera and TF updates
// // function send_scene_updates() {
// //     drot = dxy_to_drotation(dx, dy);
// //     ddist = dscroll_to_ddist(dscroll);
// //     dx,dy,dscroll = 0;
// //     socket.emit('scene_update', {"drot": drot, "ddist": ddist, "tf": tf});
// // }
// // setInterval(send_scene_updates, dt);

// // recording scene changes ***********************************************

// // rotation
// function onMouseDown(event) {
//     prev_x = event.clientX;
//     prev_y = event.clientY;
//     is_updating_rotation = true;
// }

// function onMouseMove(event) {
//     if (!is_updating_rotation) return;
//     curr_x = event.clientX;
//     curr_y = event.clientY;
//     // modify dx,dy, to be reported
//     dx += curr_x - prev_x;
//     dy += curr_y - prev_y;

//     prev_x = curr_x;
//     prev_y = curr_y;
// }

// function onMouseUp(event) {
//     is_updating_rotation = false;
//     curr_x = event.clientX;
//     curr_y = event.clientY;
//     // modify dx,dy, to be reported
//     dx += curr_x - prev_x;
//     dy += curr_y - prev_y;

//     prev_x = curr_x;
//     prev_y = curr_y;
// }

// // zoom percentage change
// function onWheelScroll(event) {
//     dscroll += event.deltaY;
// }

// // transfer function


