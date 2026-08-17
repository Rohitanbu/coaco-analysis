const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

let js = fs.readFileSync('templatemo-quantix-analyser.js', 'utf-8');
js = js.replace(/function createSampleRow\(\) \{/, 'function createSampleRow() { console.log("createSampleRow called!");');
fs.writeFileSync('test-analyser.js', js);

let html = fs.readFileSync('analyser.html', 'utf-8');
html = html.replace('templatemo-quantix-analyser.js', 'test-analyser.js');

const virtualConsole = new jsdom.VirtualConsole();
virtualConsole.on("log", (...args) => console.log("LOG:", ...args));
virtualConsole.on("error", (...args) => console.log("ERROR:", ...args));
virtualConsole.on("warn", (...args) => console.log("WARN:", ...args));
virtualConsole.on("info", (...args) => console.log("INFO:", ...args));

const dom = new JSDOM(html, {
  runScripts: "dangerously",
  resources: "usable",
  virtualConsole
});

dom.window.addEventListener("load", () => {
    console.log("LOADED");
    setTimeout(() => {
        const btn = dom.window.document.getElementById('addSampleBtn');
        if (btn) {
            btn.click();
            const container = dom.window.document.getElementById('batchSamplesContainer');
            console.log("Rows count:", container.children.length);
        }
    }, 1000);
});
