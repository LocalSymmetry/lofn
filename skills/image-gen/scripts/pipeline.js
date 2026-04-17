#!/usr/bin/env node
/**
 * Full Image Generation Pipeline
 * 
 * 1. Generate with FAL Flux Pro 1.1 Ultra
 * 2. Optionally refine with Gemini
 * 
 * Usage:
 *   node pipeline.js --prompt "..." --aspect "9:16" --refine "Fix hands" --output ./final.png
 * 
 * Environment:
 *   FAL_KEY - For generation
 *   GEMINI_API_KEY - For refinement
 *   OPENAI_API_KEY - Fallback if FAL unavailable
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    prompt: '',
    aspect: '9:16',
    output: './output.png',
    refine: null,
    seed: null,
    safety: 2,
    provider: 'auto' // auto, fal, openai
  };
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--prompt': opts.prompt = args[++i]; break;
      case '--aspect': opts.aspect = args[++i]; break;
      case '--output': opts.output = args[++i]; break;
      case '--refine': opts.refine = args[++i]; break;
      case '--seed': opts.seed = args[++i]; break;
      case '--safety': opts.safety = args[++i]; break;
      case '--provider': opts.provider = args[++i]; break;
    }
  }
  return opts;
}

function runScript(scriptPath, args) {
  const fullPath = path.join(__dirname, scriptPath);
  const cmd = `node "${fullPath}" ${args.map(a => `"${a}"`).join(' ')}`;
  
  try {
    const output = execSync(cmd, { 
      encoding: 'utf8',
      env: process.env,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    return { success: true, output };
  } catch (err) {
    return { success: false, error: err.message, stderr: err.stderr };
  }
}

async function generateWithOpenAI(prompt, outputPath) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY not set');
  }
  
  const https = require('https');
  
  return new Promise((resolve, reject) => {
    const requestBody = {
      model: 'dall-e-3',
      prompt: prompt,
      n: 1,
      size: '1024x1792', // Portrait aspect
      quality: 'hd',
      response_format: 'url'
    };
    
    const data = JSON.stringify(requestBody);
    
    const req = https.request({
      hostname: 'api.openai.com',
      path: '/v1/images/generations',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    }, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', async () => {
        try {
          const result = JSON.parse(body);
          if (result.error) {
            reject(new Error(result.error.message));
            return;
          }
          
          const imageUrl = result.data[0].url;
          
          // Ensure output directory exists
          const outputDir = path.dirname(outputPath);
          if (outputDir && !fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
          }
          
          // Download image
          https.get(imageUrl, (imgRes) => {
            const chunks = [];
            imgRes.on('data', chunk => chunks.push(chunk));
            imgRes.on('end', () => {
              const buffer = Buffer.concat(chunks);
              fs.writeFileSync(outputPath, buffer);
              resolve({ url: imageUrl, revised_prompt: result.data[0].revised_prompt });
            });
          }).on('error', reject);
          
        } catch (e) {
          reject(new Error(`Parse error: ${body.substring(0, 500)}`));
        }
      });
    });
    
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function pipeline(opts) {
  if (!opts.prompt) {
    console.error('Error: --prompt is required');
    process.exit(1);
  }
  
  console.log('🚀 Starting Lofn Image Pipeline');
  console.log('================================');
  console.log(`Prompt: ${opts.prompt.substring(0, 100)}...`);
  console.log(`Aspect: ${opts.aspect}`);
  console.log(`Output: ${opts.output}`);
  if (opts.refine) console.log(`Refine: ${opts.refine}`);
  console.log('');
  
  // Determine provider
  let provider = opts.provider;
  if (provider === 'auto') {
    provider = process.env.FAL_KEY ? 'fal' : 'openai';
  }
  
  const tempOutput = opts.refine 
    ? opts.output.replace(/(\.[^.]+)$/, '_raw$1')
    : opts.output;
  
  // Step 1: Generate
  console.log(`📸 Step 1: Generate (${provider})`);
  
  let generateSuccess = false;
  
  if (provider === 'fal') {
    const args = [
      '--prompt', opts.prompt,
      '--aspect', opts.aspect,
      '--output', tempOutput
    ];
    if (opts.seed) args.push('--seed', opts.seed);
    if (opts.safety) args.push('--safety', opts.safety);
    
    const result = runScript('fal-generate.cjs', args);
    generateSuccess = result.success;
    
    if (!generateSuccess) {
      console.log('⚠️ FAL failed, trying OpenAI fallback...');
      provider = 'openai';
    }
  }
  
  if (provider === 'openai' || !generateSuccess) {
    try {
      console.log('   Using OpenAI DALL-E 3...');
      const result = await generateWithOpenAI(opts.prompt, tempOutput);
      console.log(`✅ Generated: ${tempOutput}`);
      generateSuccess = true;
      
      // Save metadata
      const metaPath = tempOutput.replace(/\.[^.]+$/, '.json');
      fs.writeFileSync(metaPath, JSON.stringify({
        prompt: opts.prompt,
        revised_prompt: result.revised_prompt,
        model: 'dall-e-3',
        provider: 'openai',
        timestamp: new Date().toISOString()
      }, null, 2));
      
    } catch (err) {
      console.error('❌ OpenAI failed:', err.message);
      process.exit(1);
    }
  }
  
  // Step 2: Refine (if requested)
  if (opts.refine && generateSuccess) {
    console.log('');
    console.log('🔧 Step 2: Refine (Gemini)');
    
    const args = [
      '--image', tempOutput,
      '--instruction', opts.refine,
      '--output', opts.output
    ];
    
    const result = runScript('gemini-edit.js', args);
    
    if (!result.success) {
      console.log('⚠️ Refinement failed, using raw output');
      if (tempOutput !== opts.output) {
        fs.copyFileSync(tempOutput, opts.output);
      }
    }
  }
  
  console.log('');
  console.log('================================');
  console.log(`✅ Pipeline complete: ${opts.output}`);
  
  // Final output for automation
  console.log(JSON.stringify({
    success: true,
    output: opts.output,
    provider: provider,
    refined: !!opts.refine
  }));
}

pipeline(parseArgs()).catch(err => {
  console.error('❌ Pipeline error:', err.message);
  process.exit(1);
});
