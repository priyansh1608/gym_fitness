/* =========================================================
   BODY SCANNER
   Camera -> pose detection (MediaPipe, in browser) ->
   scan animation -> auto workout + diet plan -> dashboard
   ========================================================= */

const SCAN_MS = 7000;

const MP_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/" +
    "pose_landmarker/pose_landmarker_lite/float16/1/" +
    "pose_landmarker_lite.task";


const $ = (id) => document.getElementById(id);

const video = $("camera");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const stage = $("stage");
const chip = $("chip");
const countdownEl = $("countdown");
const bar = $("bar");
const statusText = $("cameraStatus");
const startBtn = $("startBtn");
const skipBtn = $("skipBtn");
const steps = $("steps");
const regionEls = {};

document.querySelectorAll("#regions li").forEach((li) => {
    regionEls[li.dataset.key] = li;
});


// ---------- body regions (scan order: top -> bottom) ----------

const REGION_ORDER = [
    "shoulders", "chest", "waist", "hips", "legs", "ankles"
];

const REGION_LABEL = {
    shoulders: "SHOULDERS", chest: "CHEST", waist: "WAIST",
    hips: "HIPS", legs: "LEGS", ankles: "ANKLES"
};

// fallback positions (fraction of frame height)
const DEFAULT_Y = {
    shoulders: 0.22, chest: 0.33, waist: 0.45,
    hips: 0.52, legs: 0.72, ankles: 0.92
};

const SKELETON = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
    [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32]
];

const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

const REQUIRED = [11, 12, 23, 24, 25, 26, 27, 28];


// ---------- state ----------

let landmarker = null;
let modelPromise = null;
let modelDone = false;
let lm = null;                 // latest landmarks (or null)
let fullBody = false;
let state = "idle";            // idle | countdown | scanning | busy
let scanStart = 0;
let scanTop = 0.03;
let scanBottom = 0.97;
let scanResolve = null;
let samples = [];
let doneRegions = new Set();
let lastVideoTime = -1;


const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const round2 = (v) => Math.round(v * 100) / 100;


// ---------- camera ----------

async function startCamera() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;

        statusText.textContent =
            "Camera on hai. Poori body frame me rakho.";

        startBtn.disabled = false;

        requestAnimationFrame(draw);

    } catch (error) {

        console.error("Camera error:", error);

        chip.textContent = "Camera not available";
        chip.classList.add("warn");

        statusText.textContent =
            "Camera permission denied ya camera nahi mila. " +
            "Aap 'Skip scan' se bina scan ke plan bana sakte ho.";
    }
}


function stopCamera() {

    if (video.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
    }
}

window.addEventListener("beforeunload", stopCamera);


// ---------- pose model (loaded in background) ----------

async function loadModel() {

    try {

        const vision = await import(MP_URL + "/vision_bundle.mjs");

        const fileset = await vision.FilesetResolver.forVisionTasks(
            MP_URL + "/wasm"
        );

        const make = (delegate) =>
            vision.PoseLandmarker.createFromOptions(fileset, {
                baseOptions: { modelAssetPath: MODEL_URL, delegate },
                runningMode: "VIDEO",
                numPoses: 1
            });

        try {
            landmarker = await make("GPU");
        } catch (gpuError) {
            landmarker = await make("CPU");
        }

    } catch (error) {

        console.warn("Pose model unavailable, demo mode:", error);

        landmarker = null;
    }
}


// ---------- math helpers ----------

function median(values) {

    const a = values.slice().sort((x, y) => x - y);
    const m = Math.floor(a.length / 2);

    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}


function isFullBody(p) {

    return REQUIRED.every((i) =>
        p[i] &&
        p[i].visibility > 0.5 &&
        p[i].x > 0 && p[i].x < 1 &&
        p[i].y > 0 && p[i].y < 1
    );
}


function frameMetrics(p, W, H) {

    const dist = (a, b) =>
        Math.hypot((p[a].x - p[b].x) * W, (p[a].y - p[b].y) * H);

    const mid = (a, b) => ({
        x: (p[a].x + p[b].x) / 2 * W,
        y: (p[a].y + p[b].y) / 2 * H
    });

    const hipWidth = dist(23, 24);

    if (hipWidth < 1) {
        return null;
    }

    const s = mid(11, 12);
    const h = mid(23, 24);
    const a = mid(27, 28);

    return {
        sh: dist(11, 12) / hipWidth,
        tl: Math.hypot(s.x - h.x, s.y - h.y) /
            Math.hypot(h.x - a.x, h.y - a.y)
    };
}


function regionY(key, p) {

    if (!p) {
        return DEFAULT_Y[key];
    }

    const sh = (p[11].y + p[12].y) / 2;
    const hp = (p[23].y + p[24].y) / 2;
    const kn = (p[25].y + p[26].y) / 2;
    const an = (p[27].y + p[28].y) / 2;

    return {
        shoulders: sh,
        chest: sh + (hp - sh) * 0.30,
        waist: sh + (hp - sh) * 0.70,
        hips: hp,
        legs: kn,
        ankles: an
    }[key];
}


// ---------- render loop ----------

function draw(now) {

    requestAnimationFrame(draw);

    if (!video.videoWidth) {
        return;
    }

    if (overlay.width !== video.videoWidth) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
    }

    const W = overlay.width;
    const H = overlay.height;

    ctx.clearRect(0, 0, W, H);

    detect(W, H);

    const px = (p) => (1 - p.x) * W;      // mirrored like the video
    const py = (p) => p.y * H;

    let lineY = null;
    let p = 0;

    if (state === "scanning") {

        p = Math.min((now - scanStart) / SCAN_MS, 1);

        lineY = scanTop + (scanBottom - scanTop) * p;

        bar.style.width = (p * 100) + "%";

        // already-scanned area tint
        ctx.fillStyle = "rgba(74, 222, 128, 0.10)";
        ctx.fillRect(0, 0, W, lineY * H);

        updateRegions(lineY);
    }

    // guide silhouette (only before scan, when body not detected yet)
    if (state === "idle" && !fullBody) {
        drawGuide(W, H);
    }

    // skeleton + joints (revealed top -> bottom as the line moves)
    if (lm) {

        const limit = lineY === null ? 1 : lineY;
        const faint = state === "idle";

        ctx.lineWidth = Math.max(2, W / 260);
        ctx.strokeStyle = faint
            ? "rgba(74, 222, 128, 0.55)"
            : "#4ade80";

        ctx.shadowColor = "#4ade80";
        ctx.shadowBlur = faint ? 0 : 12;

        SKELETON.forEach(([a, b]) => {

            if (lm[a].y > limit || lm[b].y > limit) {
                return;
            }

            ctx.beginPath();
            ctx.moveTo(px(lm[a]), py(lm[a]));
            ctx.lineTo(px(lm[b]), py(lm[b]));
            ctx.stroke();
        });

        ctx.fillStyle = "#ffffff";

        JOINTS.forEach((i) => {

            if (lm[i].y > limit) {
                return;
            }

            ctx.beginPath();
            ctx.arc(px(lm[i]), py(lm[i]), Math.max(3, W / 170),
                0, Math.PI * 2);
            ctx.fill();
        });

        ctx.shadowBlur = 0;
    }

    // labels for regions that are already scanned
    if (state === "scanning" || state === "busy") {
        drawRegionLabels(W, H);
    }

    // scan line
    if (lineY !== null) {

        const y = lineY * H;

        const g = ctx.createLinearGradient(0, y - 40, 0, y);
        g.addColorStop(0, "rgba(74, 222, 128, 0)");
        g.addColorStop(1, "rgba(74, 222, 128, 0.55)");

        ctx.fillStyle = g;
        ctx.fillRect(0, y - 40, W, 40);

        ctx.fillStyle = "#bbf7d0";
        ctx.shadowColor = "#4ade80";
        ctx.shadowBlur = 18;
        ctx.fillRect(0, y - 1, W, 3);
        ctx.shadowBlur = 0;

        if (p >= 1 && scanResolve) {
            const done = scanResolve;
            scanResolve = null;
            done();
        }
    }

    updateChip();
}


function detect(W, H) {

    if (!landmarker || video.readyState < 2 ||
        video.currentTime === lastVideoTime) {
        return;
    }

    lastVideoTime = video.currentTime;

    try {

        const res = landmarker.detectForVideo(video, performance.now());

        lm = res.landmarks && res.landmarks[0]
            ? res.landmarks[0]
            : null;

    } catch (error) {

        lm = null;
    }

    fullBody = !!lm && isFullBody(lm);

    if (state === "scanning" && fullBody) {

        const m = frameMetrics(lm, W, H);

        if (m) {
            samples.push(m);
        }
    }
}


function drawGuide(W, H) {

    ctx.save();

    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(74, 222, 128, 0.65)";

    ctx.beginPath();
    ctx.arc(W * 0.5, H * 0.14, H * 0.06, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(W * 0.36, H * 0.24);
    ctx.lineTo(W * 0.64, H * 0.24);
    ctx.lineTo(W * 0.60, H * 0.55);
    ctx.lineTo(W * 0.64, H * 0.95);
    ctx.lineTo(W * 0.54, H * 0.95);
    ctx.lineTo(W * 0.50, H * 0.62);
    ctx.lineTo(W * 0.46, H * 0.95);
    ctx.lineTo(W * 0.36, H * 0.95);
    ctx.lineTo(W * 0.40, H * 0.55);
    ctx.closePath();
    ctx.stroke();

    ctx.restore();
}


function drawRegionLabels(W, H) {

    const fs = Math.max(12, Math.round(W / 46));

    ctx.font = "bold " + fs + "px Arial";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    REGION_ORDER.forEach((key) => {

        if (!doneRegions.has(key)) {
            return;
        }

        const y = regionY(key, lm) * H;
        const text = REGION_LABEL[key] + " \u2713";
        const w = ctx.measureText(text).width + 16;

        ctx.fillStyle = "rgba(74, 222, 128, 0.25)";
        ctx.fillRect(0, y - 1, W, 2);

        ctx.fillStyle = "rgba(17, 24, 39, 0.8)";
        ctx.fillRect(W - w - 12, y - fs, w, fs * 2);

        ctx.fillStyle = "#4ade80";
        ctx.fillText(text, W - 20, y);
    });
}


function updateRegions(lineY) {

    REGION_ORDER.forEach((key) => {

        if (!doneRegions.has(key) && lineY >= regionY(key, lm)) {
            doneRegions.add(key);
        }
    });

    paintRegions(true);
}


function paintRegions(scanning) {

    let activeSet = false;

    REGION_ORDER.forEach((key) => {

        const el = regionEls[key];

        if (doneRegions.has(key)) {
            el.className = "done";
        } else if (scanning && !activeSet) {
            el.className = "active";
            activeSet = true;
        } else {
            el.className = "";
        }
    });
}


function updateChip() {

    let text;
    let warn = false;

    if (state === "countdown") {
        text = "Get ready...";
    } else if (state === "scanning") {
        text = "Scanning... stay still";
    } else if (state === "busy") {
        text = "Scan complete";
    } else if (!modelDone) {
        text = "Loading AI model...";
    } else if (!landmarker) {
        text = "Demo mode - AI model offline";
        warn = true;
    } else if (fullBody) {
        text = "Full body detected - ready";
    } else {
        text = "Step back - show head to toe";
        warn = true;
    }

    if (chip.textContent !== text) {
        chip.textContent = text;
    }

    chip.classList.toggle("warn", warn);
}


// ---------- scan flow ----------

async function startScan() {

    if (state !== "idle") {
        return;
    }

    startBtn.disabled = true;
    skipBtn.disabled = true;

    await modelPromise;

    doneRegions = new Set();
    samples = [];
    bar.style.width = "0%";
    paintRegions(false);

    state = "countdown";

    for (let n = 3; n > 0; n--) {
        countdownEl.textContent = n;
        countdownEl.classList.add("show");
        await sleep(1000);
    }

    countdownEl.classList.remove("show");

    // sweep from head to feet
    if (lm && fullBody) {
        scanTop = Math.max(0, lm[0].y - 0.06);
        scanBottom = Math.min(1, Math.max(lm[27].y, lm[28].y) + 0.05);
    } else {
        scanTop = 0.03;
        scanBottom = 0.97;
    }

    state = "scanning";
    stage.classList.add("scanning");
    statusText.textContent = "Body scan chal raha hai...";

    scanStart = performance.now();

    await new Promise((resolve) => { scanResolve = resolve; });

    finishScan();
}


async function finishScan() {

    stage.classList.remove("scanning");

    // pose detection ran, but the full body was not seen often enough
    if (landmarker && samples.length < 8) {

        state = "idle";
        doneRegions = new Set();
        paintRegions(false);
        bar.style.width = "0%";

        statusText.textContent =
            "Poori body (sir se pair tak) detect nahi hui. " +
            "Thoda peeche hat kar dobara scan karo.";

        startBtn.disabled = false;
        skipBtn.disabled = false;

        return;
    }

    state = "busy";

    REGION_ORDER.forEach((k) => doneRegions.add(k));
    paintRegions(false);
    bar.style.width = "100%";

    let bodyScan = null;

    if (landmarker) {

        bodyScan = {
            shoulder_hip_ratio: round2(median(samples.map((s) => s.sh))),
            torso_leg_ratio: round2(median(samples.map((s) => s.tl))),
            samples: samples.length
        };

        statusText.textContent = "Scan complete. Plan ban raha hai...";

    } else {

        statusText.textContent =
            "Demo scan complete (AI model load nahi hua). " +
            "Plan sirf aapki profile se ban raha hai...";
    }

    await buildPlan(bodyScan);
}


function setStep(id, cls) {
    $(id).className = cls;
}


async function buildPlan(bodyScan) {

    const profile = JSON.parse(
        sessionStorage.getItem("fitnessProfile") || "null"
    );

    if (!profile) {
        window.location.href = "/";
        return;
    }

    state = "busy";
    startBtn.disabled = true;
    skipBtn.disabled = true;

    ["stepBody", "stepWorkout", "stepDiet"].forEach(
        (id) => setStep(id, "")
    );

    steps.classList.add("show");

    const payload = Object.assign({}, profile);

    if (bodyScan) {
        payload.body_scan = bodyScan;
    }

    const request = fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(async (response) => {

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        return data;
    });

    request.catch(() => { });

    setStep("stepBody", "run");
    await sleep(900);

    setStep("stepBody", "ok");
    setStep("stepWorkout", "run");
    await sleep(900);

    setStep("stepWorkout", "ok");
    setStep("stepDiet", "run");

    let result;

    try {

        result = await request;

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Plan nahi ban paya: " + error.message;

        steps.classList.remove("show");
        state = "idle";
        startBtn.disabled = false;
        skipBtn.disabled = false;

        return;
    }

    setStep("stepDiet", "ok");

    sessionStorage.setItem("fitnessResult", JSON.stringify(result));

    statusText.textContent = "Plan ready! Dashboard khul raha hai...";

    await sleep(700);

    window.location.href = "/dashboard";
}


// ---------- init ----------

startBtn.addEventListener("click", startScan);

skipBtn.addEventListener("click", () => {

    if (state === "idle") {
        buildPlan(null);
    }
});

modelPromise = loadModel().finally(() => { modelDone = true; });

startCamera();
