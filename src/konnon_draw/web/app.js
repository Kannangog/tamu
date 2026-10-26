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

// Setup High-DPI canvas
function setupCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (currentActions.length > 0) {
        redrawCanvasUpTo(currentIndex);
    }
}
window.addEventListener('resize', setupCanvas);

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
    currentActions = [];
    currentPaths = [];
    currentIndex = 0;
    updateProgressUI();
    tokenFeed.innerHTML = '<div style="color:var(--text-dim); text-align:center; padding: 24px 0;">Canvas cleared</div>';
});

function resetCanvas() {
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    ctx.clearRect(0, 0, w, h);
    stylus.style.display = 'none';
    stylusRipple.style.display = 'none';
    btnReplay.disabled = true;
    btnDownloadSvg.disabled = true;
    btnDownloadJson.disabled = true;
    svgCodeBox.textContent = '<!-- SVG will appear here once drawing finishes -->';
    currentPaths = [];
}

// Fetch new drawing from model API
async function startNewDrawing() {
    stopAnimation();
    resetCanvas();

    penStateLabel.textContent = 'NEURAL INFERENCE IN PROGRESS...';
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
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
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
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
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
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;

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

// Initialize on page load
setupCanvas();
loadShowcase();
startNewDrawing();
