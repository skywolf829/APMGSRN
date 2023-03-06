
// import { io } from "socket.io-client";
// const socket = io();
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

let canvas_width = 512;
let canvas_height = 512;

// const canvas = document.createElement("canvas");
let canvas = document.getElementById("canvas")
// canvas.setAttribute("id", "canvas");
canvas.setAttribute("width", canvas_width);
canvas.setAttribute("height", canvas_height);
canvas.style.border = '1px solid black';

// document.body.appendChild(canvas)
const ctx = canvas.getContext('2d');
// let imgdata = new ImageData(256,256);
let img = new Image();
img.crossOrigin = 'anonymous';
img.src = "/video_feed";
img.onload = () => {
    ctx.drawImage(img, 0, 0);
}

// img.src = "{{ url_for('video_feed') }}"
setInterval(function() {
    console.log("Draw");
    // const timestamp = Date.now();
    // img.src = `/video_feed?timesteamp=${timestamp}`;
    // img.onload = () => {
    //     ctx.drawImage(img, 0, 0);
    // }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
}, 1000 / 60);

// setInterval(function() {
//     // let img = new Image();
//     // img.onload = function() {
//     //     ctx.clearRect(0, 0, canvas.width, canvas.height);
//     //     ctx.drawImage(img, 0, 0);
//     // }
//     // img.src = "/video_feed";
//     // console.log("Draw");
//     // // ctx.clearRect(0, 0, canvas.width, canvas.height);
//     // // ctx.drawImage(img, 0, 0);

// }, 1000/60);

// setInterval(function() {
//     fetch('/request_img', {
//         method: 'GET',
//         headers: {'Content-Type': 'application/json'},
//     })
//     .then(response => response.json())
//     .then(data => {

//     })
//     .catch(error => console.error(error));
// }, 1000/60);

let is_updating_rotation = false;
let has_responded = false;
let prev_x, prev_y; 
let dx, dy, dscroll = 0;
let tf = {
    opacity: [],
    color: [],
};

function rescale_ndc(x, length) {
    return x/length*2-1;
}

// // recording scene changes ***********************************************

// rotation
function onMouseDown(event) {
    is_updating_rotation = true;
    fetch('/mousedown', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            "x_start": rescale_ndc(event.clientX, canvas_width),
            "y_start": rescale_ndc(canvas_height-event.clientY, canvas_height)
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));

}

function onMouseMove(event) {
    if (!is_updating_rotation) return;
    // Make a POST request to the Flask route with the x and y coordinates
    // console.log(canvas_height-event.clientY, canvas_height, rescale_ndc(canvas_height-event.clientY, canvas_height));
    fetch('/mousemove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            "x_curr": rescale_ndc(event.clientX, canvas_width),
            "y_curr": rescale_ndc(canvas_height-event.clientY, canvas_height),
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

function onMouseUp(event) {
    if (!is_updating_rotation) return;
    is_updating_rotation = false;
    // Make a POST request to the Flask route with the x and y coordinates
    fetch('/mousemove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            "x_curr": rescale_ndc(event.clientX, canvas_width),
            "y_curr": rescale_ndc(canvas_height-event.clientY, canvas_height)
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

// zoom percentage change
function onWheelScroll(event) {
    console.log(event.deltaMode);
    // Make a POST request to the Flask route with the x and y coordinates
    fetch('/wheelscroll', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({"dscroll": event.deltaY})
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

canvas.addEventListener("mouseup", onMouseUp);
canvas.addEventListener("mousedown", onMouseDown);
canvas.addEventListener("mousemove", onMouseMove);
canvas.addEventListener("mouseout", onMouseUp);
canvas.addEventListener("wheel", onWheelScroll);

// // transfer function


