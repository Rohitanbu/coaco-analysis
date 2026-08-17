const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

let js = fs.readFileSync('templatemo-quantix-analyser.js', 'utf-8');
let html = fs.readFileSync('analyser.html', 'utf-8');
// Remove the external scripts
html = html.replace('<script src="templatemo-quantix-script.js"></script>', '');
html = html.replace('<script src="templatemo-quantix-analyser.js"></script>', '');
// Inline it
html = html.replace('</body>', `<script>${js}</script></body>`);

const virtualConsole = new jsdom.VirtualConsole();
virtualConsole.on("log", (...args) => console.log("LOG:", ...args));
virtualConsole.on("error", (...args) => console.log("ERROR:", ...args));

const dom = new JSDOM(html, {
  runScripts: "dangerously"
});

console.log("Rows count before click:", dom.window.document.getElementById('batchSamplesContainer').children.length);
const btn = dom.window.document.getElementById('addSampleBtn');
btn.click();
console.log("Rows count after click:", dom.window.document.getElementById('batchSamplesContainer').children.length);

