import axios from 'axios';

// Use relative path so it works when served by the same origin
const API_URL = '/api';

export const getConfig = async () => {
    const res = await axios.get(`${API_URL}/config`);
    return res.data;
};

export const generateConcepts = async (data) => {
    const res = await axios.post(`${API_URL}/generate/concepts`, data);
    return res.data;
};

export const generatePrompts = async (data) => {
    const res = await axios.post(`${API_URL}/generate/prompts`, data);
    return res.data;
};

export const selectBestPairs = async (data) => {
    const res = await axios.post(`${API_URL}/generate/best_pairs`, data);
    return res.data;
};

export const chat = async (data) => {
    const res = await axios.post(`${API_URL}/chat`, data);
    return res.data;
};
