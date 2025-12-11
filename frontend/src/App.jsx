import React, { useState } from 'react';
import { Container, AppBar, Toolbar, Typography, Button, Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import CompetitionMode from './components/CompetitionMode';
import Chat from './components/Chat';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff9d',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  const [tab, setTab] = useState('create');

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <AppBar position="static" elevation={0} sx={{ borderBottom: '1px solid #333' }}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#00ff9d', fontWeight: 'bold', letterSpacing: 2 }}>
              LOFN /// ENGINE
            </Typography>
            <Button color="inherit" onClick={() => setTab('create')} sx={{ borderBottom: tab === 'create' ? '2px solid #00ff9d' : 'none', borderRadius: 0 }}>
              Competition Mode
            </Button>
            <Button color="inherit" onClick={() => setTab('chat')} sx={{ borderBottom: tab === 'chat' ? '2px solid #00ff9d' : 'none', borderRadius: 0 }}>
              Personality Chat
            </Button>
          </Toolbar>
        </AppBar>
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
          {tab === 'create' && <CompetitionMode />}
          {tab === 'chat' && <Chat />}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
