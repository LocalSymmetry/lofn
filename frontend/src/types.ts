export interface StyleAxes {
  "Abstraction vs. Realism": number;
  "Emotional Valence": number;
  "Color Intensity": number;
  "Symbolic Density": number;
  "Compositional Complexity": number;
  "Textural Richness": number;
  "Symmetry vs. Asymmetry": number;
  "Novelty": number;
  "Figure-Ground Relationship": number;
  "Dynamic vs. Static": number;
}

export interface CreativitySpectrum {
  literal: number;
  inventive: number;
  transformative: number;
}

export interface ConceptRequest {
  input_text: string;
  model: string;
  temperature: number;
  max_retries: number;
  style_axes?: Partial<StyleAxes>; // Use string keys to match Python alias
  creativity_spectrum?: CreativitySpectrum;
  medium: string;
  reasoning_level: string;
  input_images?: string[];
}

export interface PromptRequest {
  input_text: string;
  concept: string;
  medium: string;
  model: string;
  temperature: number;
  max_retries: number;
  style_axes?: Partial<StyleAxes>;
  creativity_spectrum?: CreativitySpectrum;
  reasoning_level: string;
  input_images?: string[];
}

export interface ImageGenerationRequest {
  prompt: string;
  image_model: string;
  num_images: number;
  image_size: string;
  extra_params: Record<string, any>;
}

export interface LogMessage {
  type: string;
  content: string;
  label?: string;
  state?: string;
}

export interface GeneratedConcept {
  concept: string;
  medium: string;
}

export interface GeneratedPrompt {
  revised: string;
  synthesized: string;
}

export interface GlobalState {
  mode: string;
  selectedModel: string;
  selectedPromptModel: string;
  selectedImageModel: string;
  styleAxes: StyleAxes;
  creativitySpectrum: CreativitySpectrum;
  temperature: number;
  maxRetries: number;
  reasoningLevel: string;
  imageSize: string;
  numImages: number;
  input: string;
  concepts: GeneratedConcept[];
  prompts: GeneratedPrompt[];
  images: string[];
  logs: LogMessage[];
  isGenerating: boolean;
  currentStep: string;
}
