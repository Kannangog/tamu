// Konnon Drawing Model — Live Drawing Studio Client Engine

let currentCategory = 'bicycle';
let currentActions = [];
let currentIndex = 0;
let isPlaying = false;
let animationTimer = null;
let lastSvgContent = '';

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

// Setup High-DPI canvas
function setupCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}
window.addEventListener('resize', setupCanvas);

// Category Selection
categoryButtons.addEventListener('click', (e) => {
    const btn = e.target.closest('.category-btn');
    if (!btn) return;
    document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentCategory = btn.dataset.cat;
    // Auto-draw on category switch
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

// Clear Canvas
btnClear.addEventListener('click', () => {
    stopAnimation();
    resetCanvas();
    currentActions = [];
    currentIndex = 0;
    updateProgressUI();
    tokenFeed.innerHTML = '<div style="color:var(--text-dim); text-align:center; padding: 24px 0;">Canvas cleared</div>';
});

function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stylus.style.display = 'none';
    stylusRipple.style.display = 'none';
    btnReplay.disabled = true;
    btnDownloadSvg.disabled = true;
    btnDownloadJson.disabled = true;
    svgCodeBox.textContent = '<!-- SVG will appear here once drawing finishes -->';
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    currentIndex = 0;
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

let activePathPoints = [];

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
        activePathPoints = [{ x: a.x * w, y: a.y * h }];
        penStateLabel.textContent = `PEN DOWN: MOVE (${a.x.toFixed(2)}, ${a.y.toFixed(2)})`;
    } else if (a.type === 'LINE') {
        stylus.classList.remove('pen-up');
        const nextPt = { x: a.x * w, y: a.y * h };

        if (activePathPoints.length > 0) {
            const prevPt = activePathPoints[activePathPoints.length - 1];
            ctx.beginPath();
            ctx.moveTo(prevPt.x, prevPt.y);
            ctx.lineTo(nextPt.x, nextPt.y);
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 3;
            ctx.stroke();
        }
        activePathPoints.push(nextPt);
        penStateLabel.textContent = `DRAWING: LINE (${a.x.toFixed(2)}, ${a.y.toFixed(2)})`;
    } else if (a.type === 'PEN_UP') {
        stylus.classList.add('pen-up');
        activePathPoints = [];
        penStateLabel.textContent = 'PEN UP: REPOSITIONING';
    } else if (a.type === 'END') {
        stylus.classList.add('pen-up');
        penStateLabel.textContent = 'FINISHED: END TOKEN';
    }

    // Append to live token feed
    appendTokenItem(a, idx);
}

function appendTokenItem(a, idx) {
    const div = document.createElement('div');
    div.className = `token-item ${a.type}`;

    let coordsStr = '';
    if (a.x !== undefined && a.y !== undefined) {
        coordsStr = `x:${a.x.toFixed(2)} y:${a.y.toFixed(2)}`;
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
                // Fetch actions JSON and play on canvas
                const jRes = await fetch(it.json_url);
                if (!jRes.ok) return;
                const d = await jRes.json();

                stopAnimation();
                resetCanvas();

                // Select category button
                document.querySelectorAll('.category-btn').forEach(b => {
                    b.classList.toggle('active', b.dataset.cat === it.category);
                });
                currentCategory = it.category;
                seedInput.value = d.seed || 100;

                currentActions = d.actions;
                currentIndex = 0;
                lastSvgContent = '';

                // Fetch rendered SVG text
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
// Start initial live drawing
startNewDrawing();
