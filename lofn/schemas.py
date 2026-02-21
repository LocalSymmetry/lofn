from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

class StyleAxes(BaseModel):
    abstraction_vs_realism: int = Field(50, alias="Abstraction vs. Realism")
    emotional_valence: int = Field(50, alias="Emotional Valence")
    color_intensity: int = Field(50, alias="Color Intensity")
    symbolic_density: int = Field(50, alias="Symbolic Density")
    compositional_complexity: int = Field(50, alias="Compositional Complexity")
    textural_richness: int = Field(50, alias="Textural Richness")
    symmetry_vs_asymmetry: int = Field(50, alias="Symmetry vs. Asymmetry")
    novelty: int = Field(50, alias="Novelty")
    figure_ground_relationship: int = Field(50, alias="Figure-Ground Relationship")
    dynamic_vs_static: int = Field(50, alias="Dynamic vs. Static")

    class Config:
        populate_by_name = True

class CreativitySpectrum(BaseModel):
    literal: int = 33
    inventive: int = 33
    transformative: int = 34

class ConceptRequest(BaseModel):
    input_text: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_retries: int = 3
    style_axes: Optional[Dict[str, int]] = None
    creativity_spectrum: Optional[CreativitySpectrum] = None
    medium: str = "image" # image, video, music, story
    reasoning_level: str = "medium"
    input_images: Optional[List[str]] = None

class ConceptResponse(BaseModel):
    concepts: List[Dict[str, str]]
    style_axes: Optional[Dict[str, int]]
    creativity_spectrum: Optional[Dict[str, int]]

class PromptRequest(BaseModel):
    input_text: str
    concept: str
    medium: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_retries: int = 3
    style_axes: Optional[Dict[str, int]] = None
    creativity_spectrum: Optional[CreativitySpectrum] = None
    reasoning_level: str = "medium"
    input_images: Optional[List[str]] = None

class PromptResponse(BaseModel):
    prompts: List[str] # List of final prompts (revised + synthesized)
    raw_prompts: Dict[str, List[str]] # Detailed dictionary

class ImageGenerationRequest(BaseModel):
    prompt: str
    image_model: str
    num_images: int = 1
    image_size: str = "1024x1024"
    # Additional params as a dict to be flexible
    extra_params: Dict[str, Any] = {}

class LogMessage(BaseModel):
    type: str # "info", "warning", "error", "success", "status", "write"
    content: str
    label: Optional[str] = None # For status updates
    state: Optional[str] = None # "running", "complete", "error"
