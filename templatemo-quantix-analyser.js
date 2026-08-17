const CONFIDENCE_THRESHOLD = 85;

const uploadForm = document.getElementById('uploadForm');
const uploadStatus = document.getElementById('uploadStatus');
const addSampleBtn = document.getElementById('addSampleBtn');
const batchSamplesContainer = document.getElementById('batchSamplesContainer');
const analyseBtn = document.getElementById('analyseBtn');

const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const summarySection = document.getElementById('summarySection');
const summaryTableBody = document.getElementById('summaryTableBody');
const resultBlockTemplate = document.getElementById('resultBlockTemplate');

let sampleCount = 0;

// Batch Builder Logic
function createSampleRow() {
    const idx = sampleCount++;
    const div = document.createElement('div');
    div.className = 'batch-sample-row glass-panel';
    div.style.padding = '16px';
    div.style.borderRadius = 'var(--radius-sm)';
    div.style.display = 'flex';
    div.style.gap = '16px';
    div.style.alignItems = 'center';
    div.innerHTML = `
        <span class="sample-idx" style="font-weight:600; min-width:80px;">Sample ${idx + 1}</span>
        <label class="upload-field" style="flex:1;">
            <span class="upload-field-label">Thermal (JPG/PNG)</span>
            <input type="file" name="sample_${idx}_thermal_image" accept="image/jpeg,image/png,image/jpg" required>
        </label>
        <label class="upload-field" style="flex:1;">
            <span class="upload-field-label">Acoustic (TDMS/CSV)</span>
            <input type="file" name="sample_${idx}_acoustic_file" accept=".tdms,.csv" required>
        </label>
        <button type="button" class="btn btn-ghost remove-sample-btn" style="padding:8px 12px;" aria-label="Remove Sample">&times;</button>
    `;
    
    div.querySelector('.remove-sample-btn').addEventListener('click', () => {
        div.remove();
        updateSampleLabels();
    });
    
    batchSamplesContainer.appendChild(div);
    updateSampleLabels();
}

function updateSampleLabels() {
    const rows = batchSamplesContainer.querySelectorAll('.batch-sample-row');
    rows.forEach((row, index) => {
        row.querySelector('.sample-idx').textContent = `Sample ${index + 1}`;
    });
    if (rows.length === 0) {
        createSampleRow(); // Always keep at least 1
    }
}

addSampleBtn.addEventListener('click', createSampleRow);

// Initialize with one row
createSampleRow();

function setStatus(message, isError) {
    uploadStatus.hidden = !message;
    uploadStatus.textContent = message || '';
    uploadStatus.classList.toggle('error', Boolean(isError));
}

// Prediction Rendering Logic
function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

function renderWaveform(canvas, waveformData, playheadPosition) {
    if (!waveformData || waveformData.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    const pad = { top: 16, bottom: 16, left: 8, right: 8 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH / 2);
    ctx.lineTo(w - pad.right, pad.top + plotH / 2);
    ctx.stroke();

    const len = waveformData.length;
    let maxAbs = 0;
    for (let i = 0; i < len; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(waveformData[i]));
    }
    const scale = maxAbs > 0 ? 0.85 / maxAbs : 1;

    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH / 2);
    for (let x = 0; x < plotW; x++) {
        const idx = Math.min(Math.floor((x / plotW) * len), len - 1);
        const y = pad.top + plotH / 2 - waveformData[idx] * scale * (plotH / 2);
        ctx.lineTo(pad.left + x, y);
    }
    ctx.lineTo(pad.left + plotW, pad.top + plotH / 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 107, 107, 0.12)';
    ctx.fill();

    ctx.beginPath();
    for (let x = 0; x < plotW; x++) {
        const idx = Math.min(Math.floor((x / plotW) * len), len - 1);
        const y = pad.top + plotH / 2 - waveformData[idx] * scale * (plotH / 2);
        if (x === 0) ctx.moveTo(pad.left + x, y);
        else ctx.lineTo(pad.left + x, y);
    }
    ctx.strokeStyle = '#FF6B6B';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (playheadPosition > 0) {
        const px = pad.left + playheadPosition * plotW;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, pad.top);
        ctx.lineTo(px, pad.top + plotH);
        ctx.stroke();
    }
}

function mapApiResponse(data) {
    const confidencePct = Math.round(data.prediction.confidence * 1000) / 10;
    return {
        sample_id: data.sample_id,
        run_id: data.run_id,
        predicted_label: data.prediction.label,
        confidence: confidencePct,
        other_possibilities: data.probabilities.map(item => ({
            label: item.label,
            confidence: Math.round(item.confidence * 1000) / 10,
        })),
        features: {
            amplitude: { value: data.features.amplitude, unit: 'V (peak)' },
            rms: { value: data.features.rms, unit: 'V' },
            power: { value: data.features.power, unit: 'mean-square' },
        },
        waveform: data.waveform,
        audio_url: data.audio_url,
    };
}

function populateResultBlock(mapped, blockNode, fileObj) {
    const idStr = `result-sample-${mapped.sample_id}`;
    blockNode.id = idStr;
    blockNode.querySelector('.sample-heading').textContent = `Sample ${parseInt(mapped.sample_id) + 1}`;
    
    // Waveform rendering
    const canvas = blockNode.querySelector('.waveform-canvas');
    const playBtn = blockNode.querySelector('.playBtn');
    const timeEl = blockNode.querySelector('.timeEl');
    const progressEl = blockNode.querySelector('.progressEl');
    const progressFill = blockNode.querySelector('.progressFill');
    const waveformData = Float32Array.from(mapped.waveform.samples);
    
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return; // Prevent negative plot dimensions
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        renderWaveform(canvas, waveformData, 0);
    }
    
    // Defer resize to ensure layout is complete
    setTimeout(() => requestAnimationFrame(resize), 50);
    window.addEventListener('resize', resize);
    
    const audioEl = new Audio(mapped.audio_url);
    audioEl.preload = 'auto';
    let duration = 0;
    let isPlaying = false;
    
    function updatePlayback(elapsed) {
        timeEl.textContent = formatTime(elapsed) + ' / ' + formatTime(duration || 0);
        const pct = duration > 0 ? Math.min(elapsed / duration, 1) * 100 : 0;
        progressFill.style.width = pct + '%';
        const playhead = duration > 0 ? elapsed / duration : 0;
        renderWaveform(canvas, waveformData, playhead);
    }
    
    audioEl.addEventListener('loadedmetadata', () => {
        duration = audioEl.duration || 0;
        updatePlayback(audioEl.currentTime || 0);
    });
    audioEl.addEventListener('timeupdate', () => updatePlayback(audioEl.currentTime || 0));
    audioEl.addEventListener('ended', () => {
        audioEl.pause();
        isPlaying = false;
        playBtn.classList.remove('playing');
        updatePlayback(0);
    });
    
    playBtn.addEventListener('click', () => {
        if (isPlaying) {
            audioEl.pause();
            isPlaying = false;
            playBtn.classList.remove('playing');
        } else {
            audioEl.play().catch(e => console.error(e));
            isPlaying = true;
            playBtn.classList.add('playing');
        }
    });
    
    progressEl.addEventListener('click', (e) => {
        if (!duration) return;
        const rect = progressEl.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        audioEl.currentTime = ratio * duration;
        updatePlayback(audioEl.currentTime);
    });

    // Thermal image preview
    if (fileObj) {
        const img = blockNode.querySelector('.thermalImage');
        const placeholder = blockNode.querySelector('.thermalPlaceholder');
        img.src = URL.createObjectURL(fileObj);
        img.hidden = false;
        placeholder.hidden = true;
    }

    // Features
    const grid = blockNode.querySelector('.featuresGrid');
    const entries = [
        { key: 'amplitude', label: 'Amplitude' },
        { key: 'rms', label: 'RMS' },
        { key: 'power', label: 'Power' },
    ];
    grid.innerHTML = entries.map(({ key, label }) => {
        const f = mapped.features[key];
        const val = typeof f.value === 'number' ? f.value.toFixed(4) : '—';
        return `
            <div class="feature-metric">
                <div class="fm-label">${label}</div>
                <div class="fm-value">${val}</div>
                <div class="fm-unit">${f.unit}</div>
            </div>`;
    }).join('');

    // Prediction
    const labelEl = blockNode.querySelector('.predictedLabel');
    const confEl = blockNode.querySelector('.confidenceValue');
    labelEl.textContent = mapped.predicted_label;
    labelEl.className = 'prediction-label predictedLabel';
    const lower = mapped.predicted_label.toLowerCase();
    
    // We can add colors dynamically based on label
    if (lower.includes('ripe') && !lower.includes('over') && !lower.includes('under') && !lower.includes('un')) {
        labelEl.style.color = 'var(--emerald)';
    } else if (lower.includes('un') || lower.includes('under')) {
        labelEl.style.color = 'var(--amber)';
    } else if (lower.includes('over')) {
        labelEl.style.color = 'var(--rose)';
    }

    confEl.textContent = mapped.confidence + '%';
    
    const badge = blockNode.querySelector('.dataSourceBadge');
    badge.hidden = false;
    blockNode.querySelector('.dataSourceText').textContent = 'Live result · run ' + mapped.run_id.slice(0, 8);

    // Other possibilities
    const otherPanel = blockNode.querySelector('.otherPossibilities');
    const listEl = blockNode.querySelector('.possibilitiesList');
    if (mapped.confidence < CONFIDENCE_THRESHOLD) {
        otherPanel.classList.remove('hidden');
        otherPanel.style.display = 'block';
        
        const colorMap = {
            'Unripe': 'var(--amber)',
            'Overripe': 'var(--rose)',
            'Ripe': 'var(--emerald)'
        };
        
        listEl.innerHTML = mapped.other_possibilities.map(item => {
            const barWidth = Math.max(2, item.confidence) + '%';
            const color = colorMap[item.label] || 'var(--accent)';
            return `
                <div class="possibility-row" style="display: flex; flex-direction: column; gap: 8px;">
                    <div class="possibility-label" style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: var(--text-primary);">
                        <span>${item.label}</span>
                        <span>${item.confidence}%</span>
                    </div>
                    <div class="possibility-track" style="height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; width: 100%;">
                        <div class="possibility-fill" style="height: 100%; width: ${barWidth}; background: ${color}; border-radius: 4px; transition: width 0.8s cubic-bezier(0.2, 0.8, 0.2, 1); box-shadow: 0 0 10px ${color}40;"></div>
                    </div>
                </div>`;
        }).join('');
    } else {
        otherPanel.classList.add('hidden');
        otherPanel.style.display = 'none';
        listEl.innerHTML = '';
    }
}

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Validation
    const rows = batchSamplesContainer.querySelectorAll('.batch-sample-row');
    if (rows.length === 0) {
        setStatus('Please add at least one sample.', true);
        return;
    }
    
    const formData = new FormData();
    let completeRows = 0;
    
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const tFile = row.querySelector('input[name*="_thermal_image"]').files[0];
        const aFile = row.querySelector('input[name*="_acoustic_file"]').files[0];
        
        if (!tFile || !aFile) {
            setStatus(`Sample ${i + 1} is incomplete. Please provide both files.`, true);
            return;
        }
        
        formData.append(row.querySelector('input[name*="_thermal_image"]').name, tFile);
        formData.append(row.querySelector('input[name*="_acoustic_file"]').name, aFile);
        completeRows++;
    }
    
    setStatus(`Analysing ${completeRows} sample(s)…`, false);
    analyseBtn.disabled = true;

    try {
        const res = await fetch('/api/predict/batch', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');
        
        resultsContainer.innerHTML = '';
        summaryTableBody.innerHTML = '';
        
        // Unhide first so bounding rects resolve properly for canvas
        resultsSection.hidden = false;
        summarySection.hidden = false;
        
        data.results.forEach((apiData, i) => {
            const mapped = mapApiResponse(apiData);
            
            // Clone template
            const clone = resultBlockTemplate.content.cloneNode(true);
            const blockNode = clone.querySelector('.result-block-wrapper');
            
            // Get original file for thermal preview
            const row = batchSamplesContainer.querySelector(`input[name="sample_${mapped.sample_id}_thermal_image"]`).closest('.batch-sample-row');
            const fileObj = row.querySelector('input[name*="_thermal_image"]').files[0];
            
            populateResultBlock(mapped, blockNode, fileObj);
            resultsContainer.appendChild(clone);
            
            // Fix: Trigger fade-up for dynamically added elements since page-load observer missed them
            setTimeout(() => {
                const addedBlock = document.getElementById(`result-sample-${mapped.sample_id}`);
                if (addedBlock) {
                    addedBlock.querySelectorAll('.fade-up').forEach((el, index) => {
                        setTimeout(() => el.classList.add('visible'), index * 100);
                    });
                }
            }, 50);
            
            // Populate Summary Table Row
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border-subtle)';
            tr.innerHTML = `
                <td style="padding: 16px; font-weight:600; color:var(--text-primary);">Sample ${parseInt(mapped.sample_id) + 1}</td>
                <td style="padding: 16px;">${mapped.predicted_label}</td>
                <td style="padding: 16px;">${mapped.confidence}%</td>
                <td style="padding: 16px;">${mapped.features.amplitude.value.toFixed(4)}</td>
                <td style="padding: 16px;">${mapped.features.rms.value.toFixed(4)}</td>
                <td style="padding: 16px;">${mapped.features.power.value.toFixed(4)}</td>
                <td style="padding: 16px;">
                    <a href="#result-sample-${mapped.sample_id}" style="color:var(--accent); font-weight:600; font-family:'Plus Jakarta Sans', sans-serif;">View Details &rarr;</a>
                </td>
            `;
            summaryTableBody.appendChild(tr);
        });

        setStatus('');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (err) {
        setStatus(err.message || 'Analysis failed.', true);
    } finally {
        analyseBtn.disabled = false;
    }
});
