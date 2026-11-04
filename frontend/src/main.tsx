import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import '@fontsource-variable/geist/wght.css';
import '@fontsource-variable/geist-mono/wght.css';
import '@fontsource/geist-pixel/400.css';
import './style.css';
import './theme.css';

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
