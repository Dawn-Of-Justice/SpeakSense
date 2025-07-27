#!/usr/bin/env node
/**
 * Simple frontend startup script
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting SpeakSense Frontend...');
console.log('📁 Frontend directory:', __dirname);

// Start the Next.js development server
const npmProcess = spawn('npm', ['run', 'dev'], {
    cwd: __dirname,
    stdio: 'inherit',
    shell: true
});

npmProcess.on('error', (error) => {
    console.error('❌ Failed to start frontend:', error);
});

npmProcess.on('close', (code) => {
    console.log(`\n🛑 Frontend process exited with code ${code}`);
});

// Handle Ctrl+C
process.on('SIGINT', () => {
    console.log('\n🛑 Stopping frontend server...');
    npmProcess.kill('SIGINT');
    process.exit(0);
});

console.log('🌐 Frontend will be available at: http://localhost:3000');
console.log('💡 Press Ctrl+C to stop the server');
