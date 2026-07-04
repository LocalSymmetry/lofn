// Local-first Monaco: bundle the editor worker (no CDN fetch). Markdown/YAML
// highlighting is built into monaco's basic languages; only the core editor
// worker is needed.
import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import { loader } from "@monaco-editor/react";

(self as any).MonacoEnvironment = {
  getWorker() {
    return new EditorWorker();
  },
};

loader.config({ monaco });

export { monaco };
