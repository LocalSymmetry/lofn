import React, { useState } from 'react';
import type { GlobalState, ConceptRequest, PromptRequest, ImageGenerationRequest, GeneratedConcept } from '../types';
import { streamRequest } from '../api';
import { Send, Image as ImageIcon, Sparkles, Loader, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

interface ImageGeneratorProps {
  state: GlobalState;
  setState: React.Dispatch<React.SetStateAction<GlobalState>>;
}

export const ImageGenerator: React.FC<ImageGeneratorProps> = ({ state, setState }) => {
  const [activeStep, setActiveStep] = useState<number>(0);

  const handleGenerateConcepts = async () => {
    if (!state.input.trim()) return;

    setState(s => ({ ...s, isGenerating: true, logs: [], concepts: [] }));
    setActiveStep(1);

    const request: ConceptRequest = {
      input_text: state.input,
      model: state.selectedModel,
      temperature: state.temperature,
      max_retries: state.maxRetries,
      style_axes: state.styleAxes,
      creativity_spectrum: state.creativitySpectrum,
      medium: 'image',
      reasoning_level: state.reasoningLevel
    };

    await streamRequest<any[]>(
      '/api/generate/concepts',
      request,
      (log) => setState(s => ({ ...s, logs: [...s.logs, log] })),
      (result) => {
        // Result is a list of concepts
        setState(s => ({ ...s, concepts: result, isGenerating: false }));
      },
      (error) => {
        setState(s => ({ ...s, logs: [...s.logs, { type: 'error', content: error }], isGenerating: false }));
      }
    );
  };

  const handleGeneratePrompts = async (concept: GeneratedConcept) => {
    setState(s => ({ ...s, isGenerating: true }));
    setActiveStep(2);

    const request: PromptRequest = {
      input_text: state.input,
      concept: concept.concept,
      medium: concept.medium,
      model: state.selectedModel, // Should be Prompt Model? Using selectedModel for simplicity or we can add promptModel selector
      temperature: state.temperature,
      max_retries: state.maxRetries,
      style_axes: state.styleAxes,
      creativity_spectrum: state.creativitySpectrum,
      reasoning_level: state.reasoningLevel
    };

    await streamRequest<any[]>(
      '/api/generate/prompts',
      request,
      (log) => setState(s => ({ ...s, logs: [...s.logs, log] })),
      (result) => {
        // Result is list of prompt objects (revised, synthesized)
        // Adjust based on actual API response structure which returns list of objects with Keys
        setState(s => ({ ...s, prompts: result, isGenerating: false }));
      },
      (error) => {
        setState(s => ({ ...s, logs: [...s.logs, { type: 'error', content: error }], isGenerating: false }));
      }
    );
  };

  const handleGenerateImage = async (prompt: string) => {
    setState(s => ({ ...s, isGenerating: true }));
    setActiveStep(3);

    const request: ImageGenerationRequest = {
      prompt: prompt,
      image_model: state.selectedImageModel,
      num_images: state.numImages,
      image_size: state.imageSize,
      extra_params: {}
    };

    await streamRequest<string[]>(
      '/api/generate/image',
      request,
      (log) => setState(s => ({ ...s, logs: [...s.logs, log] })),
      (result) => {
        // Result is list of image URLs/Paths
        setState(s => ({ ...s, images: [...s.images, ...result], isGenerating: false }));
      },
      (error) => {
        setState(s => ({ ...s, logs: [...s.logs, { type: 'error', content: error }], isGenerating: false }));
      }
    );
  };

  return (
    <div className="space-y-8 pb-32">
      {/* Input Section */}
      <section className="space-y-4">
        <h2 className="text-2xl font-light text-slate-100 flex items-center gap-2">
          <Sparkles className="text-cyan-400" />
          <span className="font-serif italic">Concept Origin</span>
        </h2>
        <div className="relative">
          <textarea
            value={state.input}
            onChange={(e) => setState(s => ({ ...s, input: e.target.value }))}
            placeholder="Describe the essence of the art you wish to generate..."
            className="w-full h-32 bg-slate-900 border border-slate-800 rounded-lg p-4 text-slate-200 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all resize-none"
          />
          <button
            onClick={handleGenerateConcepts}
            disabled={state.isGenerating || !state.input.trim()}
            className="absolute bottom-4 right-4 bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-md flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {state.isGenerating && activeStep === 1 ? <Loader className="animate-spin" size={16} /> : <Send size={16} />}
            Generate Concepts
          </button>
        </div>
      </section>

      {/* Concepts Grid */}
      {state.concepts.length > 0 && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-light text-slate-300">Concepts & Mediums</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {state.concepts.map((c, i) => (
              <div key={i} className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-cyan-500/50 transition-colors cursor-pointer group"
                   onClick={() => handleGeneratePrompts(c)}>
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs font-mono text-cyan-500 bg-cyan-950/30 px-2 py-1 rounded">Concept {i+1}</span>
                  <ArrowRight size={16} className="text-slate-600 group-hover:text-cyan-400 transition-colors" />
                </div>
                <h3 className="font-semibold text-slate-200 mb-1">{c.concept}</h3>
                <p className="text-sm text-slate-400 italic">{c.medium}</p>
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* Prompts Section */}
      {state.prompts.length > 0 && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-light text-slate-300">Refined Prompts</h2>
          <div className="space-y-6">
            {state.prompts.map((p: any, i) => ( // Using any because API returns complex objects sometimes, need to align with schema
              <div key={i} className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
                <div className="grid grid-cols-1 gap-4">
                  <div className="space-y-2">
                    <span className="text-xs font-bold text-slate-500 uppercase">Synthesized Prompt</span>
                    <p className="text-sm text-slate-300 font-mono bg-black/20 p-3 rounded">{p['Synthesized Prompts'] || p['synthesized_prompt']}</p>
                    <button
                      onClick={() => handleGenerateImage(p['Synthesized Prompts'] || p['synthesized_prompt'])}
                      className="text-xs flex items-center gap-1 text-cyan-400 hover:text-cyan-300"
                    >
                      <ImageIcon size={12} /> Generate Image
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* Gallery */}
      {state.images.length > 0 && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-light text-slate-300">Gallery</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {state.images.map((img, i) => (
              <div key={i} className="aspect-square bg-slate-900 rounded-lg overflow-hidden relative group">
                <img src={img.startsWith('/') ? `/api${img}` : img} alt="Generated" className="w-full h-full object-cover" /> {/* Handle local paths proxying */}
                {img.startsWith('/') && <img src={img} alt="Generated" className="w-full h-full object-cover absolute top-0 left-0" onError={(e) => e.currentTarget.style.display = 'none'} />}
                {/* Fallback logic for serving static files: if it starts with /, it's relative to root. We mounted static files at root but images are at /images. API serves static files from frontend/dist.
                    We need to expose /images via FastAPI as well.
                */}
              </div>
            ))}
          </div>
        </motion.section>
      )}
    </div>
  );
};
