import React, { useState, useEffect } from 'react';
import {
  Box, Grid, TextField, Select, MenuItem, FormControl, InputLabel,
  Slider, Typography, Button, Paper, CircularProgress, Chip,
  Card, CardContent, CardActions, Switch, FormControlLabel
} from '@mui/material';
import { getConfig, generateConcepts, generatePrompts, selectBestPairs } from '../api';

const STYLE_AXES_DEFS = {
  image: [
    "Abstraction vs. Realism", "Emotional Valence", "Color Intensity",
    "Symbolic Density", "Compositional Complexity", "Textural Richness",
    "Symmetry vs. Asymmetry", "Novelty", "Figure-Ground Relationship",
    "Dynamic vs. Static"
  ],
  video: [
    "Narrative Complexity", "Emotional Intensity", "Symbolism",
    "Pacing (Energy Level)", "Hook Intensity", "Aesthetic Stylization",
    "Lighting Mood", "Perspective & Lensing", "Motion Quality",
    "Surrealism vs. Realism (Physics)"
  ],
  music: [
    "Tempo", "Mood", "Instrumentation Complexity", "Lyrical Depth",
    "Genre Fusion", "Vocal Style", "Rhythmic Complexity",
    "Melodic Emphasis", "Harmonic Richness", "Production Style"
  ]
};

export default function CompetitionMode() {
  const [config, setConfig] = useState({ models: [], personalities: [], panels: [] });
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0); // 0: Input, 1: Concepts, 2: Prompts

  // Inputs
  const [input, setInput] = useState('');
  const [mediumType, setMediumType] = useState('image');
  const [model, setModel] = useState('');
  const [personality, setPersonality] = useState('LLM Generated');
  const [panel, setPanel] = useState('LLM Generated');
  const [numResults, setNumResults] = useState(3);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [styleAxes, setStyleAxes] = useState({});
  const [autoStyle, setAutoStyle] = useState(true);

  // Outputs
  const [concepts, setConcepts] = useState([]);
  const [metaPrompt, setMetaPrompt] = useState('');
  const [generatedPrompts, setGeneratedPrompts] = useState([]);

  useEffect(() => {
    getConfig().then(data => {
      setConfig(data);
      if (data.models.length > 0) setModel(data.models[0]);
    });
  }, []);

  useEffect(() => {
    // Reset style axes when medium changes
    const defaults = {};
    const axes = STYLE_AXES_DEFS[mediumType] || [];
    axes.forEach(ax => defaults[ax] = 50);
    setStyleAxes(defaults);
  }, [mediumType]);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files).slice(0, 5);
    Promise.all(files.map(file => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    })).then(images => setUploadedImages(images));
  };

  const handleGenerateConcepts = async () => {
    setLoading(true);
    try {
      // Find prompt text for selected personality/panel
      const selPers = config.personalities.find(p => p.name === personality);
      const selPanel = config.panels.find(p => p.name === panel);

      const res = await generateConcepts({
        input_text: input,
        medium_type: mediumType,
        model: model,
        competition_mode: true, // Always true per requirements
        personality_prompt: selPers ? selPers.prompt : null,
        panel_prompt: selPanel ? selPanel.prompt : null,
        images: uploadedImages,
        style_axes: autoStyle ? null : styleAxes,
        max_retries: 3
      });

      setConcepts(res.concepts);
      setMetaPrompt(res.meta_prompt);
      setStep(1);
    } catch (e) {
      console.error(e);
      alert("Error generating concepts");
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePrompts = async () => {
    setLoading(true);
    try {
      // If competition mode, we select best pairs then generate
      // Or user selects? Requirement: "The user can select the number of results sets from 1 to 12."
      // Let's assume we filter the top N pairs.

      // First, get best pairs if we have many
      let pairsToUse = concepts;
      if (concepts.length > numResults) {
          // Use API to select best pairs?
          // "The user can supply a prompt and trigger the full Lofn... system will do the rest."
          // But user also wants to select number of results.
          // Let's use the API to select best pairs
          const best = await selectBestPairs({
              input_text: input,
              pairs: concepts,
              num_best_pairs: numResults,
              model: model
          });
          pairsToUse = best;
      }

      // Now generate prompts for each pair
      const allPrompts = [];
      for (const pair of pairsToUse) {
          const prompts = await generatePrompts({
              input_text: input,
              concept: pair.concept,
              medium: pair.medium,
              medium_type: mediumType,
              model: model,
              style_axes: autoStyle ? null : styleAxes,
              images: uploadedImages
          });
          // Attach pair info
          if (Array.isArray(prompts)) {
              prompts.forEach(p => allPrompts.push({...p, concept: pair.concept, medium: pair.medium}));
          } else if (prompts.revised_prompts) {
             // Music structure
             prompts.revised_prompts.forEach(p => allPrompts.push({...p, concept: pair.concept, medium: pair.medium, type: 'Revised'}));
             prompts.synthesized_prompts.forEach(p => allPrompts.push({...p, concept: pair.concept, medium: pair.medium, type: 'Synthesized'}));
          }
      }
      setGeneratedPrompts(allPrompts);
      setStep(2);
    } catch (e) {
       console.error(e);
       alert("Error generating prompts");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Left Panel: Controls */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Configuration</Typography>

            <FormControl fullWidth margin="normal">
              <InputLabel>Medium</InputLabel>
              <Select value={mediumType} label="Medium" onChange={e => setMediumType(e.target.value)}>
                <MenuItem value="image">Image</MenuItem>
                <MenuItem value="video">Video</MenuItem>
                <MenuItem value="music">Music</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth margin="normal">
              <InputLabel>Model</InputLabel>
              <Select value={model} label="Model" onChange={e => setModel(e.target.value)}>
                {config.models.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl fullWidth margin="normal">
              <InputLabel>Personality</InputLabel>
              <Select value={personality} label="Personality" onChange={e => setPersonality(e.target.value)}>
                <MenuItem value="LLM Generated"><em>LLM Generated</em></MenuItem>
                {config.personalities.map(p => <MenuItem key={p.name} value={p.name}>{p.name}</MenuItem>)}
              </Select>
            </FormControl>

             <FormControl fullWidth margin="normal">
              <InputLabel>Panel</InputLabel>
              <Select value={panel} label="Panel" onChange={e => setPanel(e.target.value)}>
                <MenuItem value="LLM Generated"><em>LLM Generated</em></MenuItem>
                {config.panels.map(p => <MenuItem key={p.name} value={p.name}>{p.name}</MenuItem>)}
              </Select>
            </FormControl>

            <Typography gutterBottom sx={{ mt: 2 }}>Number of Results: {numResults}</Typography>
            <Slider
              value={numResults}
              min={1} max={12} step={1} marks
              onChange={(e, v) => setNumResults(v)}
            />

            <FormControlLabel
              control={<Switch checked={autoStyle} onChange={e => setAutoStyle(e.target.checked)} />}
              label="Auto Style Axes"
            />

            {!autoStyle && (
                <Box sx={{ mt: 2 }}>
                    {STYLE_AXES_DEFS[mediumType]?.map(ax => (
                        <Box key={ax} sx={{ mb: 1 }}>
                            <Typography variant="caption">{ax}</Typography>
                            <Slider
                                size="small"
                                value={styleAxes[ax] || 50}
                                onChange={(e, v) => setStyleAxes({...styleAxes, [ax]: v})}
                            />
                        </Box>
                    ))}
                </Box>
            )}

            <Button variant="contained" component="label" fullWidth sx={{ mt: 2 }}>
                Upload Reference Images
                <input type="file" hidden multiple accept="image/*" onChange={handleImageUpload} />
            </Button>
            {uploadedImages.length > 0 && <Typography variant="caption">{uploadedImages.length} images selected</Typography>}

          </Paper>
        </Grid>

        {/* Right Panel: Input & Results */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, minHeight: '80vh' }}>
             {/* Input Section */}
             <Box sx={{ mb: 4 }}>
                 <TextField
                    fullWidth
                    multiline
                    rows={4}
                    label="Describe your idea..."
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    variant="outlined"
                    sx={{ backgroundColor: '#222' }}
                 />
                 <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                     <Button
                        variant="contained"
                        color="primary"
                        size="large"
                        onClick={handleGenerateConcepts}
                        disabled={loading || !input}
                     >
                        {loading ? <CircularProgress size={24} /> : "RUN ENGINE"}
                     </Button>
                     {step >= 1 && (
                         <Button variant="outlined" onClick={handleGeneratePrompts} disabled={loading}>
                             Generate Prompts ({concepts.length})
                         </Button>
                     )}
                 </Box>
             </Box>

             {/* Results Section */}
             {step === 1 && (
                 <Box>
                     <Typography variant="h5" gutterBottom>Generated Concepts</Typography>
                     <Grid container spacing={2}>
                         {concepts.map((c, i) => (
                             <Grid item xs={12} sm={6} key={i}>
                                 <Card variant="outlined">
                                     <CardContent>
                                         <Typography variant="subtitle1" color="primary">{c.concept}</Typography>
                                         <Typography variant="caption" color="text.secondary">{c.medium}</Typography>
                                     </CardContent>
                                 </Card>
                             </Grid>
                         ))}
                     </Grid>
                 </Box>
             )}

             {step === 2 && (
                 <Box>
                     <Typography variant="h5" gutterBottom>Final Prompts</Typography>
                     <Grid container spacing={2}>
                         {generatedPrompts.map((p, i) => (
                             <Grid item xs={12} key={i}>
                                 <Card variant="outlined" sx={{ borderColor: '#00ff9d' }}>
                                     <CardContent>
                                         <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                             <Chip label={`${p.medium}`} size="small" />
                                             <Typography variant="caption">{p.type || 'Prompt'}</Typography>
                                         </Box>

                                         {mediumType === 'music' ? (
                                             <>
                                                <Typography variant="h6">{p.title}</Typography>
                                                <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 1, whiteSpace: 'pre-wrap' }}>
                                                    {p.music_prompt}
                                                </Typography>
                                                <Typography variant="caption" display="block" sx={{ mt: 1 }}>Lyrics Prompt:</Typography>
                                                <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#aaa' }}>
                                                    {p.lyrics_prompt}
                                                </Typography>
                                             </>
                                         ) : (
                                             <>
                                                 <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 1 }}>
                                                    {p.Revised_Prompts || p['Revised Prompts'] || p.Synthesized_Prompts || p['Synthesized Prompts']}
                                                 </Typography>
                                             </>
                                         )}
                                     </CardContent>
                                     <CardActions>
                                         <Button size="small" onClick={() => navigator.clipboard.writeText(JSON.stringify(p, null, 2))}>Copy JSON</Button>
                                     </CardActions>
                                 </Card>
                             </Grid>
                         ))}
                     </Grid>
                 </Box>
             )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
