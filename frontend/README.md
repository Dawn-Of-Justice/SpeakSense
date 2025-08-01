# Frontend Configuration

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

Create a `.env.local` file with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## Available Scripts

- `npm run dev`: Starts the development server with Turbopack
- `npm run build`: Builds the application for production
- `npm start`: Starts the production server
- `npm run lint`: Runs ESLint for code linting

## Tech Stack

- **Framework**: Next.js 15.4.4
- **React**: 19.1.0
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **Animation**: Framer Motion
- **WebSocket**: Socket.IO Client
- **Icons**: Lucide React

## Directory Structure

- `app/`: Next.js app directory with pages and layouts
- `public/`: Static assets
- Configuration files for TypeScript, ESLint, PostCSS, and Tailwind
