const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Create a local server to serve the static files
  const { exec } = require('child_process');
  const server = exec('python3 -m http.server 8080');
  
  await new Promise(resolve => setTimeout(resolve, 2000));

  await page.goto('http://localhost:8080/analyser.html');
  
  // Fill the form
  await page.evaluate(() => {
     // create dummy files
     const dt = new DataTransfer();
     const file = new File(['dummy content'], 'dummy.jpg', {type: 'image/jpeg'});
     dt.items.add(file);
     document.querySelector('input[name="sample_0_thermal_image"]').files = dt.files;
     
     const dt2 = new DataTransfer();
     const file2 = new File(['dummy content'], 'dummy.csv', {type: 'text/csv'});
     dt2.items.add(file2);
     document.querySelector('input[name="sample_0_acoustic_file"]').files = dt2.files;
  });

  // Since we don't have the backend running on 8080, fetch will fail.
  // We can mock the fetch response.
  await page.route('/api/predict/batch', async route => {
    const json = {
      results: [
        {
          sample_id: "0",
          run_id: "1234",
          prediction: { label: "Ripe", confidence: 0.92 },
          probabilities: [
             { label: "Ripe", confidence: 0.92 },
             { label: "Unripe", confidence: 0.08 }
          ],
          features: { amplitude: 1.2, rms: 0.5, power: 0.8 },
          waveform: { sample_rate_hz: 12800, samples: [0, 0.5, 1, 0.5, 0, -0.5, -1] },
          audio_url: "/api/audio/1234.wav"
        }
      ]
    };
    await route.fulfill({ json });
  });

  await page.click('#analyseBtn');
  
  await page.waitForTimeout(2000);
  
  await page.screenshot({ path: 'screenshot.png', fullPage: true });
  
  server.kill();
  await browser.close();
})();
