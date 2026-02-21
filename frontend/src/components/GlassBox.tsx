import React, { useEffect, useRef } from 'react';
import type { LogMessage } from '../types';
import { Terminal, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface GlassBoxProps {
  logs: LogMessage[];
  isVisible: boolean;
}

export const GlassBox: React.FC<GlassBoxProps> = ({ logs, isVisible }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-0 right-0 w-96 h-[32rem] bg-slate-900 border-l border-t border-slate-800 shadow-2xl flex flex-col z-40 transition-all duration-300">
      <div className="p-3 bg-slate-950 border-b border-slate-800 flex items-center space-x-2">
        <Terminal size={16} className="text-cyan-500" />
        <span className="text-xs font-mono font-bold text-slate-300 uppercase">Tree of Thoughts / System Log</span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-xs">
        <AnimatePresence>
          {logs.map((log, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex gap-3"
            >
              <div className="pt-1 shrink-0">
                {log.type === 'error' && <AlertCircle size={14} className="text-red-500" />}
                {log.type === 'success' && <CheckCircle size={14} className="text-green-500" />}
                {log.type === 'status' && <div className="w-3 h-3 rounded-full border border-cyan-500 border-t-transparent animate-spin" />}
                {log.type === 'info' || log.type === 'write' && <Info size={14} className="text-slate-600" />}
              </div>
              <div className="flex-1">
                {log.label && <div className="text-cyan-400 font-bold mb-1">{log.label}</div>}
                <div className={`whitespace-pre-wrap ${log.type === 'error' ? 'text-red-400' : 'text-slate-300'}`}>
                  {log.content}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};
