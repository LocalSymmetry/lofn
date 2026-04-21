#!/usr/bin/env node
/**
 * FAL image generator
 *
 * Usage:
 *   node fal-generate.cjs --prompt "..." --aspect "1:1" --output ./out.png
 *   node fal-generate.cjs --endpoint "fal-ai/lightning-models" --model-name "Lykon/dreamshaper-xl-lightning" --prompt "..." --aspect "1:1" --output ./out.png
 *
 * Environment:
 *   FAL_KEY - Required API key
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const FAL_KEY = process.env.FAL_KEY;
const DEFAULT_ENDPOINT = 'fal-ai/flux-pro/v1.1-ultra';

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    prompt: '',
    aspect: '9:16',
    output: './output.png',
    format: 'png',
    seed: null,
    safety: 2,
    raw: false,
    enhance: false,
    endpoint: DEFAULT_ENDPOINT,
    modelName: ''
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--prompt': opts.prompt = args[++i]; break;
      case '--aspect': opts.aspect = args[++i]; break;
      case '--output': opts.output = args[++i]; break;
      case '--format': opts.format = args[++i]; break;
      case '--seed': opts.seed = parseInt(args[++i]); break;
      case '--safety': opts.safety = parseInt(args[++i]); break;
      case '--raw': opts.raw = true; break;
      case '--enhance': opts.enhance = true; break;
      case '--endpoint': opts.endpoint = args[++i]; break;
      case '--model-name': opts.modelName = args[++i]; break;
    }
  }
  return opts;
}

function aspectToImageSize(aspect) {
  switch (aspect) {
    case '1:1': return { width: 1024, height: 1024 };
    case '16:9': return { width: 1344, height: 768 };
    case '9:16': return { width: 768, height: 1344 };
    case '4:3': return { width: 1152, height: 896 };
    case '3:4': return { width: 896, height: 1152 };
    case '3:2': return { width: 1216, height: 832 };
    case '2:3': return { width: 832, height: 1216 };
    case '21:9': return { width: 1536, height: 640 };
    case '9:21': return { width: 640, height: 1536 };
    default: return { width: 1024, height: 1024 };
  }
}

async function httpRequest(options, data = null) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        resolve({ status: res.statusCode, body, headers: res.headers });
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function submitAndWait(endpoint, input) {
  const data = JSON.stringify(input);

  const res = await httpRequest({
    hostname: 'fal.run',
    path: `/${endpoint}`,
    method: 'POST',
    headers: {
      'Authorization': `Key ${FAL_KEY}`,
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data)
    }
  }, data);

  if (res.status !== 200) {
    throw new Error(`FAL API error ${res.status}: ${res.body}`);
  }

  return JSON.parse(res.body);
}

async function downloadImage(url, outputPath) {
  if (url.startsWith('data:')) {
    const match = url.match(/^data:([^;,]+)?(;base64)?,(.*)$/);
    if (!match) throw new Error('Unsupported data URL format');
    const isBase64 = Boolean(match[2]);
    const payload = match[3];
    const buffer = isBase64
      ? Buffer.from(payload, 'base64')
      : Buffer.from(decodeURIComponent(payload), 'utf8');
    fs.writeFileSync(outputPath, buffer);
    return;
  }

  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outputPath);
    https.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        https.get(res.headers.location, (res2) => {
          res2.pipe(file);
          file.on('finish', () => { file.close(); resolve(); });
        }).on('error', reject);
      } else {
        res.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
      }
    }).on('error', reject);
  });
}

async function generate(opts) {
  if (!FAL_KEY) {
    console.error('Error: FAL_KEY environment variable not set');
    process.exit(1);
  }

  if (!opts.prompt) {
    console.error('Error: --prompt is required');
    process.exit(1);
  }

  console.log(`🎨 Generating with FAL...`);
  console.log(`   Endpoint: ${opts.endpoint}`);
  if (opts.modelName) console.log(`   Model: ${opts.modelName}`);
  console.log(`   Aspect: ${opts.aspect}`);
  console.log(`   Prompt: ${opts.prompt.substring(0, 100)}...`);

  let input;
  if (opts.endpoint === 'fal-ai/lightning-models') {
    input = {
      prompt: opts.prompt,
      image_size: aspectToImageSize(opts.aspect),
      format: opts.format,
      num_images: 1,
      guidance_scale: 2,
      num_inference_steps: 5,
      enable_safety_checker: true,
      sync_mode: true
    };
    if (opts.modelName) input.model_name = opts.modelName;
    if (opts.seed) input.seed = opts.seed;
  } else {
    input = {
      prompt: opts.prompt,
      aspect_ratio: opts.aspect,
      output_format: opts.format,
      safety_tolerance: String(opts.safety),
      num_images: 1
    };
    if (opts.seed) input.seed = opts.seed;
    if (opts.raw) input.raw = true;
    if (opts.enhance) input.enable_safety_checker = false;
  }

  console.log(`   Submitting to FAL...`);
  const result = await submitAndWait(opts.endpoint, input);

  if (!result.images || result.images.length === 0) {
    console.error('❌ No images returned:', JSON.stringify(result));
    process.exit(1);
  }

  const imageUrl = result.images[0].url;
  const seed = result.seed;

  console.log(`   Got image, downloading...`);

  const outputDir = path.dirname(opts.output);
  if (outputDir && !fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  await downloadImage(imageUrl, opts.output);
  console.log(`✅ Saved to: ${opts.output}`);

  const metaPath = opts.output.replace(/\.[^.]+$/, '.json');
  const metadata = {
    prompt: opts.prompt,
    aspect_ratio: opts.aspect,
    seed: seed,
    model: opts.modelName || opts.endpoint,
    endpoint: opts.endpoint,
    provider: 'fal',
    timestamp: new Date().toISOString(),
    image_url: imageUrl.startsWith('data:') ? '[inline-data-url]' : imageUrl,
    has_nsfw: result.has_nsfw_concepts?.[0] || false
  };
  fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
  console.log(`📋 Metadata: ${metaPath}`);

  console.log(JSON.stringify({ success: true, output: opts.output, seed, metadata }));
}

generate(parseArgs()).catch(err => {
  console.error('❌ Error:', err.message);
  process.exit(1);
});
