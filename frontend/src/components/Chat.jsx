import React, { useState, useEffect, useRef } from 'react';
import {
  Box, TextField, Button, Paper, Typography, Avatar,
  List, ListItem, ListItemText, ListItemAvatar, Select, MenuItem,
  FormControl, InputLabel, CircularProgress
} from '@mui/material';
import { chat, getConfig } from '../api';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [personalities, setPersonalities] = useState([]);
  const [selectedPersonality, setSelectedPersonality] = useState('');
  const [model, setModel] = useState('');
  const [config, setConfig] = useState(null);
  const [uploadedImages, setUploadedImages] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    getConfig().then(data => {
      setConfig(data);
      setPersonalities(data.personalities);
      if (data.personalities.length > 0) setSelectedPersonality(data.personalities[0].name);
      if (data.models.length > 0) setModel(data.models[0]);
    });
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!input.trim() && uploadedImages.length === 0) return;

    const userMsg = { role: 'user', content: input, images: uploadedImages };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput('');
    setUploadedImages([]);
    setLoading(true);

    try {
      const persPrompt = personalities.find(p => p.name === selectedPersonality)?.prompt || '';
      const res = await chat({
        message: userMsg.content,
        history: messages.map(m => ({role: m.role, content: m.content})), // Simplified history for API
        personality_prompt: persPrompt,
        model: model,
        images: userMsg.images
      });

      setMessages([...newHistory, { role: 'assistant', content: res.response }]);
    } catch (e) {
      console.error(e);
      setMessages([...newHistory, { role: 'assistant', content: "Error: Could not reach personality." }]);
    } finally {
      setLoading(false);
    }
  };

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

  return (
    <Box sx={{ height: '80vh', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl fullWidth>
                <InputLabel>Personality</InputLabel>
                <Select
                    value={selectedPersonality}
                    label="Personality"
                    onChange={e => setSelectedPersonality(e.target.value)}
                >
                    {personalities.map(p => <MenuItem key={p.name} value={p.name}>{p.name}</MenuItem>)}
                </Select>
            </FormControl>
             <FormControl fullWidth>
                <InputLabel>Model</InputLabel>
                <Select value={model} label="Model" onChange={e => setModel(e.target.value)}>
                    {config?.models.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                </Select>
            </FormControl>
        </Box>
      </Paper>

      <Paper sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2, backgroundColor: '#111' }}>
        <List>
          {messages.map((msg, index) => (
            <ListItem key={index} alignItems="flex-start">
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: msg.role === 'user' ? '#00ff9d' : '#ff0055', color: '#000' }}>
                    {msg.role === 'user' ? 'U' : 'AI'}
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={
                    <Box component="span" sx={{ color: msg.role === 'user' ? '#fff' : '#ccc' }}>
                        {msg.role === 'user' ? 'You' : selectedPersonality}
                    </Box>
                }
                secondary={
                  <React.Fragment>
                    <Typography
                      component="span"
                      variant="body1"
                      color="text.primary"
                      sx={{ whiteSpace: 'pre-wrap', color: '#ddd' }}
                    >
                      {msg.content}
                    </Typography>
                    {msg.images && (
                        <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            {msg.images.map((img, i) => (
                                <img key={i} src={img} alt="upload" style={{ height: 100, borderRadius: 4 }} />
                            ))}
                        </Box>
                    )}
                  </React.Fragment>
                }
              />
            </ListItem>
          ))}
          <div ref={messagesEndRef} />
        </List>
      </Paper>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button component="label" variant="outlined">
            IMG
            <input type="file" hidden multiple accept="image/*" onChange={handleImageUpload} />
        </Button>
        <TextField
            fullWidth
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
        />
        <Button variant="contained" onClick={handleSend} disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Send"}
        </Button>
      </Box>
      {uploadedImages.length > 0 && <Typography variant="caption" sx={{ mt: 1 }}>{uploadedImages.length} images attached</Typography>}
    </Box>
  );
}
