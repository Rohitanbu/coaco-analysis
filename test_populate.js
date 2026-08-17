const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

let js = fs.readFileSync('templatemo-quantix-analyser.js', 'utf-8');
let html = fs.readFileSync('analyser.html', 'utf-8');
html = html.replace('<script src="templatemo-quantix-script.js?v=2"></script>', '');
html = html.replace('<script src="templatemo-quantix-analyser.js?v=2"></script>', `<script>${js}</script>`);

const dom = new JSDOM(html, { runScripts: "dangerously" });

// Mock data
const apiData = {
    sample_id: "0",
    run_id: "1234",
    prediction: { label: "Ripe", confidence: 0.92 },
    probabilities: [
       { label: "Ripe", confidence: 0.92 },
       { label: "Unripe", confidence: 0.08 }
    ],
    features: { amplitude: 1.2, rms: 0.5, power: 0.8 },
    waveform: { sample_rate_hz: 12800, samples: [0, 0.5, 1] },
    audio_url: "/api/audio/1234.wav"
};

const window = dom.window;
const document = window.document;

// We need to simulate the batch submission
document.getElementById('addSampleBtn').click(); // ensures row exists
// We can't actually do file upload in jsdom easily, so we will manually mock mapApiResponse and populateResultBlock
window.eval(`
    const mapped = mapApiResponse(${JSON.stringify(apiData)});
    const clone = resultBlockTemplate.content.cloneNode(true);
    const blockNode = clone.querySelector('.result-block-wrapper');
    populateResultBlock(mapped, blockNode, null);
    resultsContainer.appendChild(clone);
`);

console.log("Prediction text:", document.querySelector('.predictedLabel').textContent);
console.log("Waveform canvas exists:", !!document.querySelector('.waveform-canvas'));
console.log("Features populated:", document.querySelector('.featuresGrid').innerHTML.trim());

