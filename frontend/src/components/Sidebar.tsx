import React from 'react';
import type { GlobalState, StyleAxes } from '../types';
import { Settings, ChevronLeft, ChevronRight } from 'lucide-react';

interface SidebarProps {
  state: GlobalState;
  setState: React.Dispatch<React.SetStateAction<GlobalState>>;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  models: any;
}

export const Sidebar: React.FC<SidebarProps> = ({ state, setState, collapsed, setCollapsed, models }) => {
  const toggleCollapsed = () => setCollapsed(!collapsed);

  const updateStyleAxis = (key: keyof StyleAxes, value: number) => {
    setState(prev => ({
      ...prev,
      styleAxes: { ...prev.styleAxes, [key]: value }
    }));
  };

  if (collapsed) {
    return (
      <div className="fixed left-0 top-0 h-full w-20 bg-slate-950 border-r border-slate-800 flex flex-col items-center py-8 z-50">
        <button onClick={toggleCollapsed} className="p-2 bg-slate-900 rounded-full hover:bg-slate-800 transition-colors">
          <ChevronRight size={20} />
        </button>
        <div className="mt-8">
          <Settings size={24} className="text-slate-400" />
        </div>
      </div>
    );
  }

  return (
    <div className="fixed left-0 top-0 h-full w-80 bg-slate-950 border-r border-slate-800 flex flex-col z-50 overflow-y-auto scrollbar-thin">
      <div className="p-6 flex justify-between items-center border-b border-slate-900">
        <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Lofn Studio
        </h1>
        <button onClick={toggleCollapsed} className="p-1 hover:bg-slate-900 rounded">
          <ChevronLeft size={20} className="text-slate-400" />
        </button>
      </div>

      <div className="p-6 space-y-8">
        {/* Model Selection */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Models</h3>

          <div className="space-y-2">
            <label className="text-xs text-slate-500">Reasoning Model</label>
            <select
              value={state.selectedModel}
              onChange={(e) => setState(s => ({ ...s, selectedModel: e.target.value }))}
              className="w-full bg-slate-900 border border-slate-800 rounded p-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              {models?.llm_models?.map((m: string) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-500">Image Model</label>
            <select
              value={state.selectedImageModel}
              onChange={(e) => setState(s => ({ ...s, selectedImageModel: e.target.value }))}
              className="w-full bg-slate-900 border border-slate-800 rounded p-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            >
               {models?.image_models?.map((m: string) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </div>

        {/* Style Axes */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Style Axes</h3>
          <div className="space-y-4">
            {Object.entries(state.styleAxes).map(([key, value]) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">{key}</span>
                  <span className="text-cyan-400">{value}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={value}
                  onChange={(e) => updateStyleAxis(key as keyof StyleAxes, parseInt(e.target.value))}
                  className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Creativity Spectrum */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Creativity Spectrum</h3>
          {(['literal', 'inventive', 'transformative'] as const).map(key => (
             <div key={key} className="space-y-1">
             <div className="flex justify-between text-xs">
               <span className="text-slate-400 capitalize">{key}</span>
               <span className="text-purple-400">{state.creativitySpectrum[key]}%</span>
             </div>
             <input
               type="range"
               min="0"
               max="100"
               value={state.creativitySpectrum[key]}
               onChange={(e) => setState(s => ({
                 ...s,
                 creativitySpectrum: { ...s.creativitySpectrum, [key]: parseInt(e.target.value) }
               }))}
               className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
             />
           </div>
          ))}
        </div>
      </div>
    </div>
  );
};
