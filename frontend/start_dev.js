const { spawn } = require('child_process');
const path = require('path');

console.log('Starting Next.js development server...');

const nextBin = path.join(__dirname, 'node_modules', '.bin', 'next.cmd');
const child = spawn('cmd', ['/c', nextBin, 'dev', '--port', '3001'], {
  stdio: 'inherit',
  cwd: __dirname
});

child.on('error', (error) => {
  console.error('Failed to start server:', error);
});

child.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
});
