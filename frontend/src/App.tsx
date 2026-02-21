import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { GlassBox } from './components/GlassBox';
import { ImageGenerator } from './features/ImageGenerator';
import type { GlobalState } from './types';
import { getModels } from './api';

const DEFAULT_STYLE_AXES = {
  "Abstraction vs. Realism": 50,
  "Emotional Valence": 50,
  "Color Intensity": 50,
  "Symbolic Density": 50,
  "Compositional Complexity": 50,
  "Textural Richness": 50,
  "Symmetry vs. Asymmetry": 50,
  "Novelty": 50,
  "Figure-Ground Relationship": 50,
  "Dynamic vs. Static": 50
};

const App: React.FC = () => {
  const [state, setState] = useState<GlobalState>({
    mode: 'Image Generation',
    selectedModel: 'gpt-4o',
    selectedPromptModel: 'gpt-4o',
    selectedImageModel: 'dall-e-3', // Default, will update from API
    styleAxes: DEFAULT_STYLE_AXES,
    creativitySpectrum: { literal: 33, inventive: 33, transformative: 34 },
    temperature: 0.7,
    maxRetries: 3,
    reasoningLevel: 'medium',
    imageSize: '1024x1024',
    numImages: 1,
    input: '',
    concepts: [],
    prompts: [],
    images: [],
    logs: [],
    isGenerating: false,
    currentStep: 'idle'
  });

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [models, setModels] = useState<any>({});

  useEffect(() => {
    getModels().then(data => {
      setModels(data);
      if (data.llm_models?.length) setState(s => ({ ...s, selectedModel: data.llm_models[0] }));
      if (data.image_models?.length) setState(s => ({ ...s, selectedImageModel: data.image_models[0] }));
    }).catch(console.error);
  }, []);

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500/30 overflow-hidden">
      <Sidebar
        state={state}
        setState={setState}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        models={models}
      />

      <main className={`flex-1 overflow-y-auto h-screen transition-all duration-300 ease-in-out ${sidebarCollapsed ? 'ml-20' : 'ml-80'}`}>
        <div className="max-w-6xl mx-auto p-8 animate-fade-in">
          <ImageGenerator state={state} setState={setState} />
        </div>
      </main>

      <GlassBox logs={state.logs} isVisible={true} />

      {/* Background Ambience */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-[-1] overflow-hidden">
        <div className="absolute -top-[20%] -right-[10%] w-[800px] h-[800px] bg-purple-900/10 rounded-full blur-[120px]" />
        <div className="absolute top-[40%] left-[20%] w-[600px] h-[600px] bg-cyan-900/5 rounded-full blur-[100px]" />
      </div>
    </div>
  );
};

export default App;
