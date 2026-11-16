const MODEL_SIZE = 640;
const CLASS_NAMES = [
  "eye close",
  "eye close glass",
  "eye open",
  "eye open glass",
  "phone use",
  "yawn"
];

const loadModelBtn = document.getElementById("loadModelBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const detectionsList = document.getElementById("detectionsList");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const inputCanvas = document.createElement("canvas");
inputCanvas.width = MODEL_SIZE;
inputCanvas.height = MODEL_SIZE;
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

let session = null;
let stream = null;
let running = false;
let animationId = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

function setStatus(message) {
  statusEl.textContent = message;
}

async function loadModel() {
  try {
    setStatus("Status: loading ONNX model...");
    session = await ort.InferenceSession.create("./model/best.onnx", {
      executionProviders: ["wasm"]
    });
    setStatus("Status: model loaded successfully.");
    startBtn.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("Status: failed to load model.");
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Status: camera started. Running detection...");
    detectLoop();
  } catch (error) {
    console.error(error);
    setStatus("Status: failed to access camera.");
  }
}

function stopCamera() {
  running = false;
  if (animationId) cancelAnimationFrame(animationId);

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Status: camera stopped.");
}

function preprocessFrame() {
  inputCtx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const imageData = inputCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const input = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);

  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    input[i] = imageData[i * 4] / 255.0;
    input[MODEL_SIZE * MODEL_SIZE + i] = imageData[i * 4 + 1] / 255.0;
    input[2 * MODEL_SIZE * MODEL_SIZE + i] = imageData[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function iou(boxA, boxB) {
  const x1 = Math.max(boxA.x1, boxB.x1);
  const y1 = Math.max(boxA.y1, boxB.y1);
  const x2 = Math.min(boxA.x2, boxB.x2);
  const y2 = Math.min(boxA.y2, boxB.y2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const boxAArea = Math.max(0, boxA.x2 - boxA.x1) * Math.max(0, boxA.y2 - boxA.y1);
  const boxBArea = Math.max(0, boxB.x2 - boxB.x1) * Math.max(0, boxB.y2 - boxB.y1);

  return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

function nonMaxSuppression(boxes, iouThreshold = 0.45) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];

  while (boxes.length > 0) {
    const current = boxes.shift();
    selected.push(current);

    boxes = boxes.filter(box => {
      if (box.classId !== current.classId) return true;
      return iou(box, current) < iouThreshold;
    });
  }

  return selected;
}

function parseOutput(outputTensor, confThreshold = 0.45) {
  const dims = outputTensor.dims;
  const data = outputTensor.data;

  let attrs, numPreds, channelFirst;

  if (dims.length === 3) {
    if (dims[1] <= 20 && dims[2] > dims[1]) {
      attrs = dims[1];
      numPreds = dims[2];
      channelFirst = true;
    } else {
      attrs = dims[2];
      numPreds = dims[1];
      channelFirst = false;
    }
  } else {
    throw new Error("Unexpected output shape: " + JSON.stringify(dims));
  }

  const boxes = [];
  const scaleX = canvas.width / MODEL_SIZE;
  const scaleY = canvas.height / MODEL_SIZE;

  for (let i = 0; i < numPreds; i++) {
    const values = [];

    if (channelFirst) {
      for (let a = 0; a < attrs; a++) {
        values.push(data[a * numPreds + i]);
      }
    } else {
      const offset = i * attrs;
      for (let a = 0; a < attrs; a++) {
        values.push(data[offset + a]);
      }
    }

    const x = values[0];
    const y = values[1];
    const w = values[2];
    const h = values[3];
    const classScores = values.slice(4);

    let bestScore = -1;
    let classId = -1;

    for (let c = 0; c < classScores.length; c++) {
      if (classScores[c] > bestScore) {
        bestScore = classScores[c];
        classId = c;
      }
    }

    if (bestScore >= confThreshold) {
      const x1 = (x - w / 2) * scaleX;
      const y1 = (y - h / 2) * scaleY;
      const x2 = (x + w / 2) * scaleX;
      const y2 = (y + h / 2) * scaleY;

      boxes.push({
        x1, y1, x2, y2,
        score: bestScore,
        classId,
        label: CLASS_NAMES[classId] || `class ${classId}`
      });
    }
  }

  return nonMaxSuppression(boxes);
}

function drawDetections(detections) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.font = "18px Arial";

  detections.forEach(det => {
    ctx.strokeStyle = "#00ff66";
    ctx.fillStyle = "#00ff66";
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    const text = `${det.label}: ${(det.score * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = 22;

    ctx.fillRect(det.x1, Math.max(0, det.y1 - textHeight), textWidth + 10, textHeight);
    ctx.fillStyle = "#000000";
    ctx.fillText(text, det.x1 + 5, Math.max(16, det.y1 - 6));
  });
}

function updateDetectionsList(detections) {
  detectionsList.innerHTML = "";

  if (detections.length === 0) {
    detectionsList.innerHTML = "<li>No detections</li>";
    return;
  }

  detections.forEach(det => {
    const li = document.createElement("li");
    li.textContent = `${det.label} - ${(det.score * 100).toFixed(1)}%`;
    detectionsList.appendChild(li);
  });
}

async function detectLoop() {
  if (!running || !session) return;

  try {
    const inputTensor = preprocessFrame();
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const detections = parseOutput(output);

    drawDetections(detections);
    updateDetectionsList(detections);
  } catch (error) {
    console.error(error);
    setStatus("Status: inference error. Check browser console.");
  }

  if (running) {
    animationId = requestAnimationFrame(detectLoop);
  }
}

loadModelBtn.addEventListener("click", loadModel);
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);