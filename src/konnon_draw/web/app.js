// Konnon Drawing Model — Live Drawing Studio Client Engine (Option 1: Bézier Curves & Continuous Coords)

let currentCategory = 'bicycle';
let currentActions = [];
let currentIndex = 0;
let isPlaying = false;
let animationTimer = null;
let lastSvgContent = '';
let currentPaths = []; // Array of point arrays for smooth rendering

// Canvas and contexts
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const stylus = document.getElementById('stylus');
const stylusRipple = document.getElementById('stylus-ripple');

// Controls & Labels
const btnPlayPause = document.getElementById('btnPlayPause');
const playIcon = document.getElementById('playIcon');
const playText = document.getElementById('playText');
const btnStep = document.getElementById('btnStep');
const btnReplay = document.getElementById('btnReplay');
const btnSketch = document.getElementById('btnSketch');
const btnComplete = document.getElementById('btnComplete');
const btnClear = document.getElementById('btnClear');
const speedSlider = document.getElementById('speedSlider');
const speedLabel = document.getElementById('speedLabel');
const penStateLabel = document.getElementById('penStateLabel');
const stepCounterLabel = document.getElementById('stepCounterLabel');
const progressBarFill = document.getElementById('progressBarFill');
const tempSlider = document.getElementById('tempSlider');
const tempValueLabel = document.getElementById('tempValueLabel');
const topkSlider = document.getElementById('topkSlider');
const topkValueLabel = document.getElementById('topkValueLabel');
const seedInput = document.getElementById('seedInput');
const btnDice = document.getElementById('btnDice');
const tokenFeed = document.getElementById('tokenFeed');
const tokenCountBadge = document.getElementById('tokenCountBadge');
const svgCodeBox = document.getElementById('svgCodeBox');
const btnCopySvg = document.getElementById('btnCopySvg');
const btnDownloadSvg = document.getElementById('btnDownloadSvg');
const btnDownloadJson = document.getElementById('btnDownloadJson');
const categoryButtons = document.getElementById('categoryButtons');
const showcaseGrid = document.getElementById('showcaseGrid');

// Vector Engine Toggles (Option 1)
const chkSmoothBezier = document.getElementById('chkSmoothBezier');
const chkShowHandles = document.getElementById('chkShowHandles');
const chkContinuous = document.getElementById('chkContinuous');

// Canvas size helper ensuring fallback dimensions before first paint
function getCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    const w = (rect && rect.width > 0) ? rect.width : (canvas.clientWidth || 512);
    const h = (rect && rect.height > 0) ? rect.height : (canvas.clientHeight || 512);
    return { w, h, rect };
}

// Setup High-DPI canvas
function setupCanvas() {
    const { w, h } = getCanvasSize();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (currentActions.length > 0) {
        redrawCanvasUpTo(currentIndex);
    }
}
window.addEventListener('resize', setupCanvas);
window.addEventListener('DOMContentLoaded', setupCanvas);

// Category Selection
categoryButtons.addEventListener('click', (e) => {
    const btn = e.target.closest('.category-btn');
    if (!btn) return;
    document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentCategory = btn.dataset.cat;
    startNewDrawing();
});

// Speed slider
speedSlider.addEventListener('input', () => {
    speedLabel.textContent = `${speedSlider.value}ms`;
});

// Temperature & Top-K Sliders
tempSlider.addEventListener('input', () => {
    tempValueLabel.textContent = parseFloat(tempSlider.value).toFixed(2);
});
topkSlider.addEventListener('input', () => {
    topkValueLabel.textContent = topkSlider.value === '0' ? 'None' : topkSlider.value;
});

// Dice Roll
btnDice.addEventListener('click', () => {
    seedInput.value = Math.floor(Math.random() * 10000);
});

// Toggles Redraw Handlers
if (chkSmoothBezier) {
    chkSmoothBezier.addEventListener('change', () => {
        redrawCanvasUpTo(currentIndex);
    });
}
if (chkShowHandles) {
    chkShowHandles.addEventListener('change', () => {
        redrawCanvasUpTo(currentIndex);
    });
}
if (chkContinuous) {
    chkContinuous.addEventListener('change', () => {
        startNewDrawing();
    });
}

// Clear Canvas
btnClear.addEventListener('click', () => {
    stopAnimation();
    resetCanvas();
    userPrefixActions = [];
    userStrokeCount = 0;
    lastDrawPt = null;
    currentStylusStroke = null;
    if (btnComplete) btnComplete.style.display = 'none';
    currentActions = [];
    currentPaths = [];
    currentIndex = 0;
    updateProgressUI();
    penStateLabel.textContent = 'STATUS: IDLE (READY TO DRAW)';
    tokenFeed.innerHTML = '<div style="color:var(--text-dim); text-align:center; padding: 24px 0;">Canvas cleared — draw directly with mouse/pen or click a category</div>';
});

function resetCanvas() {
    const { w, h } = getCanvasSize();
    ctx.clearRect(0, 0, w, h);
    stylus.style.display = 'none';
    stylusRipple.style.display = 'none';
    btnReplay.disabled = true;
    btnDownloadSvg.disabled = true;
    btnDownloadJson.disabled = true;
    svgCodeBox.textContent = '<!-- SVG will appear here once drawing finishes -->';
    currentPaths = [];
}

// Hardware Stylus Pointer HUD & Recording Elements
const stylusHud = document.getElementById('stylusHud');
const stylusPointerType = document.getElementById('stylusPointerType');
const stylusPressureBar = document.getElementById('stylusPressureBar');
const stylusPressureText = document.getElementById('stylusPressureText');
const stylusTiltText = document.getElementById('stylusTiltText');

function updateStylusHud(e) {
    if (!stylusHud) return;
    const type = e.pointerType || 'mouse';
    if (stylusPointerType) stylusPointerType.textContent = type.toUpperCase();
    const p = (typeof e.pressure === 'number' && e.pressure > 0) ? e.pressure : (isDrawingStroke ? 0.7 : 0);
    if (stylusPressureBar) stylusPressureBar.style.width = `${Math.round(p * 100)}%`;
    if (stylusPressureText) stylusPressureText.textContent = p.toFixed(2);
    if (stylusTiltText) stylusTiltText.textContent = `X: ${e.tiltX || 0}° Y: ${e.tiltY || 0}°`;
}

// Collaborative Sketch & Interactive Stylus State
let isDrawingStroke = false;
let userPrefixActions = [];
let userStrokeCount = 0;
let lastDrawPt = null;
let currentStylusStroke = null;
let recordedStylusStrokes = [];

function getCanvasPoint(e) {
    const { w, h, rect } = getCanvasSize();
    const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    const clientY = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    const x = Math.max(0, Math.min(w, clientX - rect.left));
    const y = Math.max(0, Math.min(h, clientY - rect.top));
    return {
        px: x,
        py: y,
        normX: Number((x / w).toFixed(4)),
        normY: Number((y / h).toFixed(4))
    };
}

function handlePointerDown(e) {
    if (e.button !== undefined && e.button !== 0) return;

    if (isPlaying) {
        stopAnimation();
    }

    updateStylusHud(e);
    isDrawingStroke = true;
    const pt = getCanvasPoint(e);
    const p = (typeof e.pressure === 'number' && e.pressure > 0) ? e.pressure : 0.7;
    const tilt = Math.round(Math.sqrt((e.tiltX || 0) ** 2 + (e.tiltY || 0) ** 2));

    lastDrawPt = { px: pt.px, py: pt.py, p };
    userPrefixActions.push({ type: 'MOVE', x: pt.normX, y: pt.normY });
    currentStylusStroke = [{ x: pt.normX, y: pt.normY, pressure: p, tilt }];

    const dotRadius = Math.max(1.5, Math.min(6.0, 1.5 + p * 3.0));
    ctx.fillStyle = '#6366f1';
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, dotRadius, 0, Math.PI * 2);
    ctx.fill();

    appendTokenItem({ type: 'MOVE', x: pt.normX, y: pt.normY }, userPrefixActions.length - 1);
    if (btnComplete) btnComplete.style.display = 'inline-flex';
    penStateLabel.textContent = `✏️ DRAWING: Stroke in progress (pressure: ${p.toFixed(2)})`;

    try {
        canvas.setPointerCapture(e.pointerId);
    } catch (_) {}
}

function handlePointerMove(e) {
    updateStylusHud(e);
    if (!isDrawingStroke || !lastDrawPt) return;

    const pt = getCanvasPoint(e);
    const dist = Math.hypot(pt.px - lastDrawPt.px, pt.py - lastDrawPt.py);
    if (dist < 2.5) return;

    const p = (typeof e.pressure === 'number' && e.pressure > 0) ? e.pressure : 0.7;
    const tilt = Math.round(Math.sqrt((e.tiltX || 0) ** 2 + (e.tiltY || 0) ** 2));
    const strokeWidth = Math.max(2.0, Math.min(8.0, 1.5 + p * 4.5));

    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(lastDrawPt.px, lastDrawPt.py);
    ctx.lineTo(pt.px, pt.py);
    ctx.stroke();

    userPrefixActions.push({ type: 'LINE', x: pt.normX, y: pt.normY });
    if (currentStylusStroke) {
        currentStylusStroke.push({ x: pt.normX, y: pt.normY, pressure: p, tilt });
    }
    lastDrawPt = { px: pt.px, py: pt.py, p };

    appendTokenItem({ type: 'LINE', x: pt.normX, y: pt.normY }, userPrefixActions.length - 1);
}

async function handlePointerUp(e) {
    updateStylusHud(e);
    if (!isDrawingStroke) return;

    isDrawingStroke = false;
    lastDrawPt = null;

    userPrefixActions.push({ type: 'PEN_UP' });
    userStrokeCount++;
    appendTokenItem({ type: 'PEN_UP' }, userPrefixActions.length - 1);

    penStateLabel.textContent = `✏️ ${userStrokeCount} stroke(s) drawn. Click "✨ Complete" to let AI finish, or keep drawing!`;
    if (btnComplete) btnComplete.style.display = 'inline-flex';

    if (currentStylusStroke && currentStylusStroke.length >= 2) {
        recordedStylusStrokes.push(currentStylusStroke);
        try {
            await fetch('/api/record_stylus_stroke', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    points: currentStylusStroke,
                    brush: 'pencil',
                    base_width: 2.5,
                    stage: 'sketch',
                })
            });
        } catch (err) {
            console.warn('Could not sync stylus stroke', err);
        }
    }
    currentStylusStroke = null;

    try {
        canvas.releasePointerCapture(e.pointerId);
    } catch (_) {}
}

canvas.addEventListener('pointerdown', handlePointerDown);
canvas.addEventListener('pointermove', handlePointerMove);
canvas.addEventListener('pointerup', handlePointerUp);
canvas.addEventListener('pointercancel', handlePointerUp);

if (btnSketch) {
    btnSketch.addEventListener('click', () => {
        stopAnimation();
        resetCanvas();
        userPrefixActions = [];
        userStrokeCount = 0;
        lastDrawPt = null;
        currentStylusStroke = null;
        penStateLabel.textContent = 'SKETCH MODE: Draw on the canvas with your mouse/stylus, then click "✨ Complete"';
        tokenFeed.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding: 20px 0;">Draw on the canvas. When ready, click "✨ Complete"</div>';
        if (btnComplete) btnComplete.style.display = 'inline-flex';
    });
}

if (btnComplete) {
    btnComplete.addEventListener('click', () => {
        if (!userPrefixActions.length) {
            alert('Please draw at least one stroke before completing!');
            return;
        }
        if (userPrefixActions[userPrefixActions.length - 1].type !== 'PEN_UP') {
            userPrefixActions.push({ type: 'PEN_UP' });
        }
        const prefixToPass = [...userPrefixActions];
        userPrefixActions = [];
        userStrokeCount = 0;
        if (btnComplete) btnComplete.style.display = 'none';
        startNewDrawing(prefixToPass);
    });
}

// Fetch new drawing from model API
async function startNewDrawing(prefixActions = null) {
    stopAnimation();
    resetCanvas();

    penStateLabel.textContent = prefixActions ? 'AI COMPLETING USER SKETCH...' : 'NEURAL INFERENCE IN PROGRESS...';
    btnPlayPause.disabled = true;

    try {
        const payload = {
            category: currentCategory,
            seed: parseInt(seedInput.value, 10) || 42,
            temperature: parseFloat(tempSlider.value),
            top_k: topkSlider.value === '0' ? null : parseInt(topkSlider.value, 10),
            smooth: chkSmoothBezier ? chkSmoothBezier.checked : true,
            continuous: chkContinuous ? chkContinuous.checked : true,
        };

        if (prefixActions && prefixActions.length > 0) {
            payload.prefix = prefixActions;
        }

        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            throw new Error(`API error: ${res.statusText}`);
        }

        const data = await res.json();
        currentActions = data.actions;
        currentIndex = 0;
        lastSvgContent = (chkSmoothBezier && chkSmoothBezier.checked) ? data.svg_smooth : data.svg_raw;

        tokenFeed.innerHTML = '';
        tokenCountBadge.textContent = `${currentActions.length} tokens`;

        btnPlayPause.disabled = false;
        btnReplay.disabled = false;
        btnDownloadSvg.disabled = false;
        btnDownloadJson.disabled = false;

        playDrawing();
    } catch (err) {
        console.error(err);
        penStateLabel.textContent = `ERROR: ${err.message}`;
        btnPlayPause.disabled = false;
    }
}

// Fetch drawing from Transformer General text-to-vector API
async function startNewDrawingFromPrompt(prompt, prefixActions = null) {
    stopAnimation();
    resetCanvas();

    penStateLabel.textContent = prefixActions ? 'TRANSFORMER COMPLETING PREFIX WITH PROMPT...' : 'TRANSFORMER INFERENCE FROM PROMPT...';
    btnPlayPause.disabled = true;

    try {
        const payload = {
            prompt: prompt,
            seed: parseInt(seedInput.value, 10) || 42,
            temperature: parseFloat(tempSlider.value),
            top_k: topkSlider.value === '0' ? 25 : parseInt(topkSlider.value, 10),
        };
        if (prefixActions && prefixActions.length > 0) {
            payload.prefix = prefixActions;
        }

        const res = await fetch('/api/generate_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            throw new Error(`API error: ${res.statusText}`);
        }

        const data = await res.json();
        currentActions = data.actions;
        currentIndex = 0;
        lastSvgContent = data.svg;

        tokenFeed.innerHTML = '';
        tokenCountBadge.textContent = `${currentActions.length} tokens`;

        btnPlayPause.disabled = false;
        btnReplay.disabled = false;
        btnDownloadSvg.disabled = false;
        btnDownloadJson.disabled = false;

        playDrawing();
    } catch (err) {
        console.error(err);
        penStateLabel.textContent = `ERROR: ${err.message}`;
        btnPlayPause.disabled = false;
    }
}

const btnGenerateText = document.getElementById('btnGenerateText');
const textPromptInput = document.getElementById('textPromptInput');
if (btnGenerateText) {
    btnGenerateText.addEventListener('click', () => {
        const prompt = textPromptInput ? textPromptInput.value.trim() : 'a red circle above a blue square';
        startNewDrawingFromPrompt(prompt);
    });
}

// Play / Pause toggle
btnPlayPause.addEventListener('click', () => {
    if (isPlaying) {
        pauseDrawing();
    } else {
        if (currentIndex >= currentActions.length || currentActions.length === 0) {
            startNewDrawing();
        } else {
            playDrawing();
        }
    }
});

function playDrawing() {
    isPlaying = true;
    playIcon.textContent = '⏸';
    playText.textContent = 'Pause';
    stylus.style.display = 'block';
    stylusRipple.style.display = 'block';
    tick();
}

function pauseDrawing() {
    isPlaying = false;
    playIcon.textContent = '▶';
    playText.textContent = 'Resume';
    clearTimeout(animationTimer);
    penStateLabel.textContent = 'STATUS: PAUSED';
}

function stopAnimation() {
    isPlaying = false;
    playIcon.textContent = '▶';
    playText.textContent = 'Draw Live';
    clearTimeout(animationTimer);
}

// Step forward 1 action
btnStep.addEventListener('click', () => {
    if (isPlaying) pauseDrawing();
    if (currentActions.length === 0) {
        startNewDrawing();
        return;
    }
    if (currentIndex < currentActions.length) {
        executeStep(currentIndex);
        currentIndex++;
        updateProgressUI();
    }
});

// Replay current drawing
btnReplay.addEventListener('click', () => {
    stopAnimation();
    const { w, h } = getCanvasSize();
    ctx.clearRect(0, 0, w, h);
    currentIndex = 0;
    currentPaths = [];
    tokenFeed.innerHTML = '';
    playDrawing();
});

// Single animation tick
function tick() {
    if (!isPlaying) return;

    if (currentIndex < currentActions.length) {
        executeStep(currentIndex);
        currentIndex++;
        updateProgressUI();

        const delay = parseInt(speedSlider.value, 10);
        animationTimer = setTimeout(tick, delay);
    } else {
        // Complete
        stopAnimation();
        penStateLabel.textContent = 'STATUS: COMPLETED (END)';
        stylus.classList.add('pen-up');
        setTimeout(() => {
            stylus.style.display = 'none';
            stylusRipple.style.display = 'none';
        }, 800);
        svgCodeBox.textContent = lastSvgContent;
    }
}

// Re-render canvas up to specified step
function redrawCanvasUpTo(limitIdx) {
    const { w, h } = getCanvasSize();
    ctx.clearRect(0, 0, w, h);

    // Reconstruct paths up to limitIdx
    const paths = [];
    let cur = [];
    for (let i = 0; i < limitIdx; i++) {
        const a = currentActions[i];
        if (a.type === 'MOVE') {
            cur = [{ x: a.x * w, y: a.y * h }];
        } else if (a.type === 'LINE') {
            if (cur.length > 0) cur.push({ x: a.x * w, y: a.y * h });
        } else if (a.type === 'PEN_UP') {
            if (cur.length > 0) { paths.push(cur); cur = []; }
        }
    }
    if (cur.length > 0) paths.push(cur);

    // Render paths
    const useBezier = chkSmoothBezier ? chkSmoothBezier.checked : true;
    const showHandles = chkShowHandles ? chkShowHandles.checked : false;

    paths.forEach(pts => {
        if (pts.length < 2) return;
        renderPathOnCanvas(pts, useBezier, showHandles);
    });
}

function renderPathOnCanvas(points, useBezier, showHandles) {
    if (points.length < 2) return;

    if (!useBezier || points.length < 3) {
        // Simple polygonal
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 3;
        ctx.stroke();
        return;
    }

    // Catmull-Rom to Cubic Bézier Spline
    const n = points.length;
    const isClosed = Math.hypot(points[0].x - points[n - 1].x, points[0].y - points[n - 1].y) < 12;

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    const handlePairs = [];

    for (let i = 0; i < n - 1; i++) {
        const p1 = points[i];
        const p2 = points[i + 1];

        const p0 = (i > 0) ? points[i - 1] : (isClosed ? points[n - 2] : { x: 2 * p1.x - p2.x, y: 2 * p1.y - p2.y });
        const p3 = (i + 2 < n) ? points[i + 2] : (isClosed ? points[1] : { x: 2 * p2.x - p1.x, y: 2 * p2.y - p1.y });

        // Angle corner check
        const v1x = p1.x - p0.x, v1y = p1.y - p0.y;
        const v2x = p2.x - p1.x, v2y = p2.y - p1.y;
        const len1 = Math.hypot(v1x, v1y), len2 = Math.hypot(v2x, v2y);
        let isCorner = false;
        if (len1 > 1e-4 && len2 > 1e-4) {
            const dot = (v1x * v2x + v1y * v2y) / (len1 * len2);
            const angle = Math.acos(Math.max(-1, Math.min(1, dot))) * (180 / Math.PI);
            if (angle > 50.0) isCorner = true;
        }

        if (isCorner) {
            ctx.lineTo(p2.x, p2.y);
        } else {
            const cx1 = p1.x + (p2.x - p0.x) / 6.0;
            const cy1 = p1.y + (p2.y - p0.y) / 6.0;
            const cx2 = p2.x - (p3.x - p1.x) / 6.0;
            const cy2 = p2.y - (p3.y - p1.y) / 6.0;
            ctx.bezierCurveTo(cx1, cy1, cx2, cy2, p2.x, p2.y);

            if (showHandles) {
                handlePairs.push({ p1, cx1, cy1, cx2, cy2, p2 });
            }
        }
    }

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Render Tangent Control Handles if enabled
    if (showHandles && handlePairs.length > 0) {
        handlePairs.forEach(h => {
            // Tangent lines
            ctx.beginPath();
            ctx.moveTo(h.p1.x, h.p1.y);
            ctx.lineTo(h.cx1, h.cy1);
            ctx.moveTo(h.p2.x, h.p2.y);
            ctx.lineTo(h.cx2, h.cy2);
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Handle points
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.arc(h.cx1, h.cy1, 3, 0, Math.PI * 2);
            ctx.arc(h.cx2, h.cy2, 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }
}

function executeStep(idx) {
    const a = currentActions[idx];
    const { w, h } = getCanvasSize();

    // Stylus positioning
    if (a.x !== undefined && a.y !== undefined) {
        const sx = a.x * w;
        const sy = a.y * h;
        stylus.style.left = `${sx}px`;
        stylus.style.top = `${sy}px`;
        stylusRipple.style.left = `${sx}px`;
        stylusRipple.style.top = `${sy}px`;
    }

    if (a.type === 'MOVE') {
        stylus.classList.remove('pen-up');
        currentPaths.push([{ x: a.x * w, y: a.y * h }]);
        penStateLabel.textContent = `PEN DOWN: MOVE (${a.x.toFixed(3)}, ${a.y.toFixed(3)})`;
    } else if (a.type === 'LINE') {
        stylus.classList.remove('pen-up');
        if (currentPaths.length === 0) {
            currentPaths.push([{ x: a.x * w, y: a.y * h }]);
        } else {
            currentPaths[currentPaths.length - 1].push({ x: a.x * w, y: a.y * h });
        }
        penStateLabel.textContent = `DRAWING: LINE (${a.x.toFixed(3)}, ${a.y.toFixed(3)})`;
    } else if (a.type === 'PEN_UP') {
        stylus.classList.add('pen-up');
        penStateLabel.textContent = 'PEN UP: REPOSITIONING';
    } else if (a.type === 'END') {
        stylus.classList.add('pen-up');
        penStateLabel.textContent = 'FINISHED: END TOKEN';
    }

    // Incremental or full redraw
    redrawCanvasUpTo(idx + 1);

    // Append to live token feed
    appendTokenItem(a, idx);
}

function appendTokenItem(a, idx) {
    const div = document.createElement('div');
    div.className = `token-item ${a.type}`;

    let coordsStr = '';
    if (a.x !== undefined && a.y !== undefined) {
        coordsStr = `x:${a.x.toFixed(3)} y:${a.y.toFixed(3)}`;
    }

    div.innerHTML = `
        <span class="token-kind">#${idx + 1} ${a.type}</span>
        <span class="token-coords">${coordsStr}</span>
    `;

    tokenFeed.appendChild(div);
    tokenFeed.scrollTop = tokenFeed.scrollHeight;
}

function updateProgressUI() {
    const total = currentActions.length || 1;
    const progress = (currentIndex / total) * 100;
    progressBarFill.style.width = `${progress}%`;
    stepCounterLabel.textContent = `STEP: ${currentIndex} / ${currentActions.length}`;
}

// Copy & Download Handlers
btnCopySvg.addEventListener('click', () => {
    if (!lastSvgContent) return;
    navigator.clipboard.writeText(lastSvgContent);
    btnCopySvg.textContent = 'Copied!';
    setTimeout(() => { btnCopySvg.textContent = 'Copy'; }, 1500);
});

btnDownloadSvg.addEventListener('click', () => {
    if (!lastSvgContent) return;
    const blob = new Blob([lastSvgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentCategory}_${seedInput.value}.svg`;
    a.click();
    URL.revokeObjectURL(url);
});

btnDownloadJson.addEventListener('click', () => {
    if (!currentActions.length) return;
    const blob = new Blob([JSON.stringify({
        category: currentCategory,
        seed: parseInt(seedInput.value, 10),
        actions: currentActions
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentCategory}_${seedInput.value}.json`;
    a.click();
    URL.revokeObjectURL(url);
});

// Load Showcase Presets
async function loadShowcase() {
    try {
        const res = await fetch('/api/showcase');
        if (!res.ok) return;
        const items = await res.json();

        showcaseGrid.innerHTML = '';
        items.forEach(it => {
            const card = document.createElement('div');
            card.className = 'showcase-card';
            card.innerHTML = `
                <div class="showcase-thumb">
                    <object data="${it.svg_url}" type="image/svg+xml"></object>
                </div>
                <div class="showcase-label">${it.category} #${it.id.split('_')[1]}</div>
            `;
            card.addEventListener('click', async () => {
                const jRes = await fetch(it.json_url);
                if (!jRes.ok) return;
                const d = await jRes.json();

                stopAnimation();
                resetCanvas();

                document.querySelectorAll('.category-btn').forEach(b => {
                    b.classList.toggle('active', b.dataset.cat === it.category);
                });
                currentCategory = it.category;
                seedInput.value = d.seed || 100;

                currentActions = d.actions;
                currentIndex = 0;
                lastSvgContent = '';

                const sRes = await fetch(it.svg_url);
                if (sRes.ok) lastSvgContent = await sRes.text();

                tokenFeed.innerHTML = '';
                tokenCountBadge.textContent = `${currentActions.length} tokens`;
                btnReplay.disabled = false;
                btnDownloadSvg.disabled = false;
                btnDownloadJson.disabled = false;

                playDrawing();
            });
            showcaseGrid.appendChild(card);
        });
    } catch (err) {
        console.warn('Could not load showcase presets', err);
    }
}

// Load Hand v2 Demos
async function loadHandDemos() {
    try {
        const res = await fetch('/api/hand_demos');
        if (!res.ok) return;
        const items = await res.json();
        const grid = document.getElementById('handDemosGrid');
        if (!grid) return;

        grid.innerHTML = '';
        items.forEach(it => {
            const card = document.createElement('div');
            card.className = 'hand-demo-card';
            card.innerHTML = `
                <div class="hand-demo-thumb">
                    <img src="${it.png_url}" alt="${it.title}">
                </div>
                <div class="hand-demo-info">
                    <div class="hand-demo-badge">${it.medium}</div>
                    <div class="hand-demo-card-title">${it.title}</div>
                    <div class="hand-demo-desc">${it.description}</div>
                </div>
            `;
            card.addEventListener('click', () => {
                openHandDemoModal(it);
            });
            grid.appendChild(card);
        });
    } catch (err) {
        console.warn('Could not load hand demos', err);
    }
}

function openHandDemoModal(demo) {
    const modal = document.getElementById('handDemoModal');
    if (!modal) return;
    document.getElementById('modalTitle').textContent = demo.title;
    document.getElementById('modalImg').src = demo.png_url;
    document.getElementById('modalMedium').textContent = demo.medium;
    document.getElementById('modalDesc').textContent = demo.description;
    document.getElementById('modalDownloadPng').href = demo.png_url;
    document.getElementById('modalDownloadSvg').href = demo.svg_url;
    modal.classList.add('active');
}

const modalCloseBtn = document.getElementById('modalCloseBtn');
if (modalCloseBtn) {
    modalCloseBtn.addEventListener('click', () => {
        const m = document.getElementById('handDemoModal');
        if (m) m.classList.remove('active');
    });
}

const handDemoModal = document.getElementById('handDemoModal');
if (handDemoModal) {
    handDemoModal.addEventListener('click', (e) => {
        if (e.target === handDemoModal) handDemoModal.classList.remove('active');
    });
}

// Dynamic Model Status in Navbar
async function fetchModelStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        const archBadge = document.getElementById('modelArchBadge');
        const statusText = document.getElementById('modelStatusText');
        if (data.staged_model_ready && data.staged_parameters) {
            const paramsM = (data.staged_parameters / 1e6).toFixed(2);
            if (archBadge) archBadge.textContent = `Staged Transformer (v0) • ${paramsM}M Params`;
            if (statusText) {
                const ckpt = data.staged_checkpoint || data.checkpoint || '';
                const parts = ckpt.replace(/\\/g, '/').split('/');
                const shortName = parts.slice(-3).join('/');
                statusText.textContent = `Model: ${shortName}`;
            }
        }
    } catch (e) {
        console.warn('Failed to load model status:', e);
    }
}

// Initialize on page load
setupCanvas();
fetchModelStatus();
loadShowcase();
loadHandDemos();
startNewDrawing();

// =====================================================================
// Phase 5: Staged Scrubber, Stylus HUD & Recorder, and A/B Comparison
// =====================================================================

// Tab Switching
const tabBtnStudio = document.getElementById('tabBtnStudio');
const tabBtnAB = document.getElementById('tabBtnAB');
const studioViewContainer = document.getElementById('studioViewContainer');
const abViewContainer = document.getElementById('abViewContainer');

if (tabBtnStudio && tabBtnAB) {
    tabBtnStudio.addEventListener('click', () => {
        tabBtnStudio.classList.add('active');
        tabBtnStudio.style.borderBottomColor = 'var(--accent)';
        tabBtnStudio.style.color = 'var(--text-main)';
        tabBtnAB.classList.remove('active');
        tabBtnAB.style.borderBottomColor = 'transparent';
        tabBtnAB.style.color = 'var(--text-muted)';
        if (studioViewContainer) studioViewContainer.style.display = 'grid';
        if (abViewContainer) abViewContainer.style.display = 'none';
    });

    tabBtnAB.addEventListener('click', () => {
        tabBtnAB.classList.add('active');
        tabBtnAB.style.borderBottomColor = 'var(--accent)';
        tabBtnAB.style.color = 'var(--text-main)';
        tabBtnStudio.classList.remove('active');
        tabBtnStudio.style.borderBottomColor = 'transparent';
        tabBtnStudio.style.color = 'var(--text-muted)';
        if (studioViewContainer) studioViewContainer.style.display = 'none';
        if (abViewContainer) abViewContainer.style.display = 'block';
    });
}



// Staged Generation & Scrubber
let currentStagedData = null;
const btnGenerateStaged = document.getElementById('btnGenerateStaged');
const activeStagePill = document.getElementById('activeStagePill');
const stageSlider = document.getElementById('stageSlider');
const stageSliderLabel = document.getElementById('stageSliderLabel');
const stagePills = document.getElementById('stagePills');

if (btnGenerateStaged) {
    btnGenerateStaged.addEventListener('click', async () => {
        const promptInput = document.getElementById('textPromptInput');
        const prompt = (promptInput && promptInput.value.trim()) || 'a figure standing in relaxed pose';
        const seed = parseInt(seedInput.value, 10) || 42;
        const temperature = parseFloat(tempSlider.value) || 0.5;
        const top_k = parseInt(topkSlider.value, 10) || 25;

        btnGenerateStaged.disabled = true;
        btnGenerateStaged.textContent = '⏳ Drawing in stages...';
        stopAnimation();

        try {
            const res = await fetch('/api/generate_staged', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, seed, temperature, top_k })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            currentStagedData = data;

            // Display final stage SVG in canvas or svgCodeBox
            lastSvgContent = data.final_svg;
            svgCodeBox.textContent = data.final_svg;
            btnCopySvg.disabled = false;
            btnDownloadSvg.disabled = false;
            btnDownloadJson.disabled = false;

            // Set up scrubber
            stageSlider.max = data.stages.length;
            stageSlider.value = data.stages.length;
            stageSliderLabel.textContent = `Stage ${data.stages.length}/${data.stages.length}`;
            activeStagePill.textContent = `Stage: Complete (${data.total_tokens} tokens)`;

            // Render stage image to canvas
            renderStageToCanvas(data.final_canvas_data_url);

            tokenCountBadge.textContent = `${data.total_tokens} tokens`;
            tokenFeed.innerHTML = data.stages.map(stg => `
                <div class="token-row">
                    <span class="token-idx" style="color:var(--accent);">STAGE</span>
                    <span class="token-type">${stg.stage_name.toUpperCase()}</span>
                    <span class="token-coords" style="color:var(--text-muted);">${stg.token_count} tokens</span>
                </div>
            `).join('');
        } catch (err) {
            console.error('Staged generation error:', err);
            alert(`Staged generation error: ${err.message}`);
        } finally {
            btnGenerateStaged.disabled = false;
            btnGenerateStaged.textContent = '🎨 Generate Staged (v0)';
        }
    });
}

function renderStageToCanvas(dataUrl) {
    if (!dataUrl) return;
    const img = new Image();
    img.onload = () => {
        const { w, h } = getCanvasSize();
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
    };
    img.src = dataUrl;
}

if (stageSlider) {
    stageSlider.addEventListener('input', () => {
        if (!currentStagedData || !currentStagedData.stages) return;
        const val = parseInt(stageSlider.value, 10);
        if (val === 0) {
            // Blank
            const { w, h } = getCanvasSize();
            ctx.clearRect(0, 0, w, h);
            stageSliderLabel.textContent = 'Blank (0)';
            activeStagePill.textContent = 'Stage: Blank';
        } else if (val <= currentStagedData.stages.length) {
            const stg = currentStagedData.stages[val - 1];
            renderStageToCanvas(stg.canvas_data_url);
            stageSliderLabel.textContent = `Stage ${val}/${currentStagedData.stages.length}`;
            activeStagePill.textContent = `Stage: ${stg.stage_name} (${stg.token_count} toks)`;
            svgCodeBox.textContent = stg.svg;
            lastSvgContent = stg.svg;
        }
    });
}

if (stagePills) {
    stagePills.addEventListener('click', (e) => {
        const btn = e.target.closest('.stage-pill-btn');
        if (!btn || !currentStagedData) return;
        document.querySelectorAll('.stage-pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const stageName = btn.dataset.stage;
        if (stageName === 'all') {
            renderStageToCanvas(currentStagedData.final_canvas_data_url);
            activeStagePill.textContent = 'Stage: All';
            stageSlider.value = currentStagedData.stages.length;
        } else {
            const idx = currentStagedData.stages.findIndex(s => s.stage_name === stageName);
            if (idx >= 0) {
                renderStageToCanvas(currentStagedData.stages[idx].canvas_data_url);
                activeStagePill.textContent = `Stage: ${stageName}`;
                stageSlider.value = idx + 1;
            }
        }
    });
}

// A/B Benchmark Execution
const btnRunAB = document.getElementById('btnRunAB');
const abPromptInput = document.getElementById('abPromptInput');
const abStagedPreview = document.getElementById('abStagedPreview');
const abDirectPreview = document.getElementById('abDirectPreview');
const abStagedBadge = document.getElementById('abStagedBadge');
const abDirectBadge = document.getElementById('abDirectBadge');

if (btnRunAB) {
    btnRunAB.addEventListener('click', async () => {
        const prompt = (abPromptInput && abPromptInput.value.trim()) || 'a figure standing in relaxed pose';
        const seed = parseInt(seedInput.value, 10) || 42;
        const temperature = parseFloat(tempSlider.value) || 0.5;
        const top_k = parseInt(topkSlider.value, 10) || 25;

        btnRunAB.disabled = true;
        btnRunAB.textContent = '⏳ Running A/B benchmark...';
        abStagedPreview.innerHTML = '<span style="color:var(--accent);">Generating 4-stage progression...</span>';
        abDirectPreview.innerHTML = '<span style="color:var(--text-dim);">Generating direct pass baseline...</span>';

        try {
            const res = await fetch('/api/compare_ab', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, seed, temperature, top_k })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            // Populate Column A (Staged)
            abStagedPreview.innerHTML = data.staged.svg;
            abStagedBadge.textContent = `${data.staged.num_stages} Stages • ${data.staged.total_tokens} Toks`;

            // Populate Column B (Direct Pass)
            abDirectPreview.innerHTML = data.unconditioned.svg;
            abDirectBadge.textContent = `${data.unconditioned.num_stages} Stage • ${data.unconditioned.total_tokens} Toks`;
        } catch (err) {
            console.error('A/B error:', err);
            alert(`A/B benchmark error: ${err.message}`);
        } finally {
            btnRunAB.disabled = false;
            btnRunAB.textContent = '⚖️ Run A/B Benchmark';
        }
    });
}

// A/B Presets
document.querySelectorAll('.ab-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (abPromptInput) {
            abPromptInput.value = btn.dataset.prompt;
            if (btnRunAB) btnRunAB.click();
        }
    });
});
