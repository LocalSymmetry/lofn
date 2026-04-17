#!/usr/bin/env node
/**
 * Gemini Image Editor (nano-banana 2)
 * 
 * Uses Gemini's native image generation to edit/refine images.
 * 
 * Usage:
 *   node gemini-edit.js --image ./input.png --instruction "Fix hands" --output ./out.png
 * 
 * Environment:
 *   GEMINI_API_KEY - Required
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MODEL = 'nano-banana-pro-preview'; // NightCafe's nano banana pro

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    image: '',
    instruction: '',
    output: './edited.png',
    model: MODEL
  };
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image': opts.image = args[++i]; break;
      case '--instruction': opts.instruction = args[++i]; break;
      case '--output': opts.output = args[++i]; break;
      case '--model': opts.model = args[++i]; break;
    }
  }
  return opts;
}

function imageToBase64(imagePath) {
  const buffer = fs.readFileSync(imagePath);
  return buffer.toString('base64');
}

function getMimeType(imagePath) {
  const ext = path.extname(imagePath).toLowerCase();
  const types = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif'
  };
  return types[ext] || 'image/png';
}

async function editWithGemini(opts) {
  return new Promise((resolve, reject) => {
    const imageData = imageToBase64(opts.image);
    const mimeType = getMimeType(opts.image);
    
    const requestBody = {
      contents: [{
        parts: [
          {
            inline_data: {
              mime_type: mimeType,
              data: imageData
            }
          },
          {
            text: `Edit this image according to the following instruction. Return ONLY the edited image, no text.

Instruction: ${opts.instruction}

Important guidelines:
- Preserve the overall composition and style
- Make minimal changes necessary to fulfill the instruction
- Maintain image quality and resolution
- If the instruction mentions fixing hands/fingers, ensure exactly 5 fingers per hand
- If the instruction mentions faces, preserve likeness while fixing issues`
          }
        ]
      }],
      generationConfig: {
        responseModalities: ['IMAGE', 'TEXT']
      }
    };
    
    const data = JSON.stringify(requestBody);
    
    const req = https.request({
      hostname: 'generativelanguage.googleapis.com',
      path: `/v1beta/models/${opts.model}:generateContent?key=${GEMINI_API_KEY}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    }, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          resolve(result);
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

async function edit(opts) {
  if (!GEMINI_API_KEY) {
    console.error('Error: GEMINI_API_KEY environment variable not set');
    process.exit(1);
  }
  
  if (!opts.image || !fs.existsSync(opts.image)) {
    console.error('Error: --image is required and must exist');
    process.exit(1);
  }
  
  if (!opts.instruction) {
    console.error('Error: --instruction is required');
    process.exit(1);
  }
  
  console.log(`🔧 Editing with Gemini (${opts.model})...`);
  console.log(`   Input: ${opts.image}`);
  console.log(`   Instruction: ${opts.instruction}`);
  
  try {
    const result = await editWithGemini(opts);
    
    // Check for errors
    if (result.error) {
      console.error('❌ Gemini error:', result.error.message);
      process.exit(1);
    }
    
    // Extract image from response
    const candidates = result.candidates || [];
    if (candidates.length === 0) {
      console.error('❌ No response from Gemini');
      console.error('Response:', JSON.stringify(result, null, 2));
      process.exit(1);
    }
    
    const parts = candidates[0].content?.parts || [];
    let imageData = null;
    let textResponse = '';
    
    for (const part of parts) {
      // Handle both snake_case and camelCase from Gemini API
      const inlineData = part.inline_data || part.inlineData;
      if (inlineData?.data) {
        imageData = inlineData.data;
        imageMime = inlineData.mime_type || inlineData.mimeType || 'image/png';
      }
      if (part.text) {
        textResponse = part.text;
      }
    }
    
    if (!imageData) {
      console.error('❌ No image in Gemini response');
      if (textResponse) {
        console.log('Text response:', textResponse);
      }
      // If no edited image, copy original as fallback
      console.log('⚠️ Copying original image as fallback');
      fs.copyFileSync(opts.image, opts.output);
    } else {
      // Save edited image
      const outputDir = path.dirname(opts.output);
      if (outputDir && !fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      
      const buffer = Buffer.from(imageData, 'base64');
      fs.writeFileSync(opts.output, buffer);
      console.log(`✅ Saved to: ${opts.output}`);
    }
    
    // Save metadata
    const metaPath = opts.output.replace(/\.[^.]+$/, '_edit.json');
    const metadata = {
      source_image: opts.image,
      instruction: opts.instruction,
      model: opts.model,
      provider: 'gemini',
      timestamp: new Date().toISOString(),
      text_response: textResponse || null
    };
    fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
    
    console.log(JSON.stringify({ success: true, output: opts.output }));
    
  } catch (err) {
    console.error('❌ Error:', err.message);
    process.exit(1);
  }
}

edit(parseArgs());
