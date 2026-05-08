#!/usr/bin/env node
/**
 * Flux Kontext Image Editor
 *
 * Uses FAL's flux-pro/kontext for precise image editing.
 * Accepts a local image file + text instruction.
 *
 * Usage:
 *   node kontext-edit.cjs --image ./input.jpg --prompt "Fix the hands" --output ./out.jpg
 *
 * Environment:
 *   FAL_KEY - Required
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const FAL_KEY = process.env.FAL_KEY;
const ENDPOINT = 'fal-ai/flux-pro/kontext';

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    image: '',
    prompt: '',
    output: './kontext_edited.jpg',
    aspect: '1:1',
    safety: '2',
    seed: null
  };
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':   opts.image   = args[++i]; break;
      case '--prompt':  opts.prompt  = args[++i]; break;
      case '--output':  opts.output  = args[++i]; break;
      case '--aspect':  opts.aspect  = args[++i]; break;
      case '--safety':  opts.safety  = args[++i]; break;
      case '--seed':    opts.seed    = parseInt(args[++i]); break;
    }
  }
  return opts;
}

function imageToDataUri(imagePath) {
  const buffer = fs.readFileSync(imagePath);
  const ext = path.extname(imagePath).toLowerCase();
  const mime = ext === '.png' ? 'image/png' : 'image/jpeg';
  return `data:${mime};base64,${buffer.toString('base64')}`;
}

function request(opts) {
  return new Promise((resolve, reject) => {
    const module_ = opts.protocol === 'http:' ? http : https;
    const req = module_.request(opts, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        if (!body.trim()) { resolve({ status: res.statusCode, body: {} }); return; }
        try { resolve({ status: res.statusCode, body: JSON.parse(body) }); }
        catch (e) { reject(new Error(`Parse error: ${body.substring(0, 300)}`)); }
      });
    });
    req.on('error', reject);
    if (opts.data) req.write(opts.data);
    req.end();
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function run(opts) {
  if (!FAL_KEY) { console.error('FAL_KEY not set'); process.exit(1); }
  if (!opts.image || !fs.existsSync(opts.image)) { console.error('--image required and must exist'); process.exit(1); }
  if (!opts.prompt) { console.error('--prompt required'); process.exit(1); }

  console.log(`🎨 Flux Kontext editing...`);
  console.log(`   Input:  ${opts.image}`);
  console.log(`   Prompt: ${opts.prompt.substring(0, 80)}...`);

  const imageDataUri = imageToDataUri(opts.image);

  const input = {
    prompt: opts.prompt,
    image_url: imageDataUri,
    aspect_ratio: opts.aspect,
    safety_tolerance: opts.safety,
    output_format: 'jpeg',
    num_images: 1
  };
  if (opts.seed) input.seed = opts.seed;

  const submitData = JSON.stringify(input);

  // Submit to queue
  const submitRes = await request({
    hostname: 'queue.fal.run',
    path: `/${ENDPOINT}`,
    method: 'POST',
    headers: {
      'Authorization': `Key ${FAL_KEY}`,
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(submitData)
    },
    data: submitData
  });

  if (submitRes.status !== 200) {
    console.error('Submit failed:', JSON.stringify(submitRes.body));
    process.exit(1);
  }

  const requestId = submitRes.body.request_id;
  console.log(`   Request ID: ${requestId}`);

  // Poll status
  let attempts = 0;
  while (attempts < 60) {
    await sleep(3000);
    const statusRes = await request({
      hostname: 'queue.fal.run',
      path: `/fal-ai/flux-pro/requests/${requestId}/status`,
      method: 'GET',
      headers: { 'Authorization': `Key ${FAL_KEY}` }
    });

    const status = statusRes.body.status;
    process.stdout.write(`\r   Status: ${status} (${++attempts * 3}s)`);

    if (status === 'COMPLETED') break;
    if (status === 'FAILED') {
      console.error('\n❌ Generation failed:', JSON.stringify(statusRes.body));
      process.exit(1);
    }
  }
  console.log('');

  // Get result
  const resultRes = await request({
    hostname: 'queue.fal.run',
    path: `/fal-ai/flux-pro/requests/${requestId}`,
    method: 'GET',
    headers: { 'Authorization': `Key ${FAL_KEY}` }
  });

  const images = resultRes.body.images || resultRes.body.output?.images;
  if (!images || images.length === 0) {
    console.error('No images in response:', JSON.stringify(resultRes.body).substring(0, 500));
    process.exit(1);
  }

  // Download result image
  const imageUrl = images[0].url;
  console.log(`   Downloading from: ${imageUrl}`);

  const outputDir = path.dirname(opts.output);
  if (outputDir && !fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  await new Promise((resolve, reject) => {
    const urlObj = new URL(imageUrl);
    const mod = urlObj.protocol === 'https:' ? https : http;
    mod.get(imageUrl, (res) => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        fs.writeFileSync(opts.output, Buffer.concat(chunks));
        resolve();
      });
    }).on('error', reject);
  });

  // Save metadata
  const meta = {
    source_image: opts.image,
    prompt: opts.prompt,
    endpoint: ENDPOINT,
    request_id: requestId,
    output_url: imageUrl,
    timestamp: new Date().toISOString(),
    seed: resultRes.body.seed || null
  };
  fs.writeFileSync(opts.output.replace(/\.[^.]+$/, '_meta.json'), JSON.stringify(meta, null, 2));

  console.log(`✅ Saved to: ${opts.output}`);
  console.log(JSON.stringify({ success: true, output: opts.output, url: imageUrl }));
}

run(parseArgs());
