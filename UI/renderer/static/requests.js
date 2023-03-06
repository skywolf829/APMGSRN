
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
canvas.style.border = '1px solid black';

document.body.appendChild(canvas)
const ctx = canvas.getContext('2d');
// let imgdata = new ImageData(256,256);
let img = new Image();
img.src = "/video_feed";
ctx.drawImage(img, 100, 100);

setInterval(function() {
    // console.log("Draw");
    ctx.drawImage(img, 100, 100);
}, 1000/60);


let is_updating_rotation = false;
let has_responded = false;
let prev_x, prev_y; 
let dx, dy, dscroll = 0;
let tf = {
    opacity: [],
    color: [],
};

// // recording scene changes ***********************************************

// rotation
function onMouseDown(event) {
    prev_x = event.clientX;
    prev_y = event.clientY;
    is_updating_rotation = true;
    fetch('/mousedown', {
        method: 'GET',
        headers: {'Content-Type': 'application/json'}
    })
    .catch(error => console.error(error));

}

function onMouseMove(event) {
    if (!is_updating_rotation) return;

    curr_x = event.clientX;
    curr_y = event.clientY;
    // modify dx,dy, to be reported
    dx = curr_x - prev_x;
    dy = curr_y - prev_y;

    prev_x = curr_x;
    prev_y = curr_y;

    // Make a POST request to the Flask route with the x and y coordinates
    fetch('/mousemove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({"dx": dx, "dy": -dy})
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

function onMouseUp(event) {
    if (!is_updating_rotation) return;
    is_updating_rotation = false;
    curr_x = event.clientX;
    curr_y = event.clientY;
    // modify dx,dy, to be reported
    dx = curr_x - prev_x;
    dy = curr_y - prev_y;
    prev_x = curr_x;
    prev_y = curr_y;

    // Make a POST request to the Flask route with the x and y coordinates
    fetch('/mousemove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({"dx": dx, "dy": -dy})
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
}

// zoom percentage change
function onWheelScroll(event) {
    // Make a POST request to the Flask route with the x and y coordinates
    fetch('/wheelscroll', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({"dscroll": -event.deltaY})
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


