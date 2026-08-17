const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

const html = fs.readFileSync('analyser.html', 'utf-8');

const virtualConsole = new jsdom.VirtualConsole();
virtualConsole.on("log", (...args) => console.log("LOG:", ...args));
virtualConsole.on("error", (...args) => console.log("ERROR:", ...args));
virtualConsole.on("warn", (...args) => console.log("WARN:", ...args));
virtualConsole.on("info", (...args) => console.log("INFO:", ...args));
virtualConsole.on("dir", (...args) => console.log("DIR:", ...args));

const dom = new JSDOM(html, {
  runScripts: "dangerously",
  resources: "usable",
  virtualConsole
});

dom.window.addEventListener("load", () => {
    console.log("LOADED");
    setTimeout(() => {
        const btn = dom.window.document.getElementById('addSampleBtn');
        console.log("addSampleBtn exists:", !!btn);
        if (btn) {
            btn.click();
            const container = dom.window.document.getElementById('batchSamplesContainer');
            console.log("Rows count:", container.children.length);
        }
    }, 1000);
});
