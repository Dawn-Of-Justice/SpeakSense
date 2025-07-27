"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Camera, CameraOff, Activity, Brain, Waves, Volume2, Wifi, WifiOff } from 'lucide-react';

const AITranscriptionInterface = () => {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionText, setTranscriptionText] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [isAddressingRobot, setIsAddressingRobot] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [displayText, setDisplayText] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  
  // WebSocket connection setup
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket('ws://localhost:8765/ws');
        
        wsRef.current.onopen = () => {
          console.log('WebSocket connected');
          setIsConnected(true);
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
              case 'video_frame':
                setVideoFrame(`data:image/jpeg;base64,${data.data}`);
                break;
              case 'transcription':
                setTranscriptionText(data.text);
                break;
              case 'ai_response':
                setAiResponse(data.text);
                break;
              case 'ai_speaking':
                setIsAiSpeaking(data.is_speaking);
                break;
              case 'addressing_status':
                setIsAddressingRobot(data.is_addressing);
                break;
              case 'classification':
                setIsAddressingRobot(data.is_addressing_robot);
                setConfidence(data.confidence);
                break;
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        wsRef.current.onclose = () => {
          console.log('WebSocket disconnected');
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };
        
        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  // Simulate typing effect for transcription
  useEffect(() => {
    if (transcriptionText) {
      let index = 0;
      setDisplayText('');
      const timer = setInterval(() => {
        if (index < transcriptionText.length) {
          setDisplayText(prev => prev + transcriptionText[index]);
          index++;
        } else {
          clearInterval(timer);
        }
      }, 50);
      return () => clearInterval(timer);
    }
  }, [transcriptionText]);

  // Display video frames from WebSocket
  useEffect(() => {
    if (videoFrame && canvasRef.current) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
        }
      };
      img.src = videoFrame;
    }
  }, [videoFrame]);
  
  const toggleTranscription = () => {
    const newState = !isTranscribing;
    setIsTranscribing(newState);
    
    // Send command to WebSocket server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: newState ? 'start_transcription' : 'stop_transcription'
      }));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 animate-pulse"></div>
        <motion.div 
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360]
          }}
          transition={{ 
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1.2, 1, 1.2],
            rotate: [360, 180, 0]
          }}
          transition={{ 
            duration: 15,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      </div>

      <div className="relative z-10 min-h-screen p-6">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            AI Transcription System
          </h1>
          <p className="text-slate-300">Real-time speech recognition with intelligent response generation</p>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Left Side - Video Stream */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {/* Video Container with Liquid Glass Effect */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-300"></div>
              <div className="relative backdrop-blur-xl bg-white/10 rounded-3xl border border-white/20 overflow-hidden shadow-2xl">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                      <Camera className="w-5 h-5 text-blue-400" />
                      Live Video Stream
                    </h2>
                    <div className="flex items-center gap-2">
                      <motion.div 
                        className={`w-3 h-3 rounded-full ${isConnected ? 'bg-red-500' : 'bg-gray-500'}`}
                        animate={isConnected ? { opacity: [1, 0.3, 1] } : { opacity: 0.5 }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                      <span className="text-sm text-slate-300">
                        {isConnected ? 'LIVE' : 'OFFLINE'}
                      </span>
                      {isConnected ? (
                        <Wifi className="w-4 h-4 text-green-400" />
                      ) : (
                        <WifiOff className="w-4 h-4 text-red-400" />
                      )}
                    </div>
                  </div>
                  
                  <div className="relative aspect-video bg-slate-800 rounded-2xl overflow-hidden">
                    <canvas
                      ref={canvasRef}
                      className="w-full h-full object-cover"
                    />
                    
                    {/* Active Speaker Detection Overlay */}
                    <AnimatePresence>
                      {isAddressingRobot && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="absolute inset-0 border-4 border-green-400 rounded-2xl"
                        >
                          <div className="absolute top-4 left-4 bg-green-500/90 text-white px-3 py-1 rounded-full text-sm font-medium">
                            Addressing AI • {(confidence * 100).toFixed(0)}%
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>

            {/* Control Panel */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-slate-500/20 to-slate-600/20 rounded-2xl blur-xl"></div>
              <div className="relative backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-green-400" />
                  System Status
                </h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={toggleTranscription}
                    disabled={!isConnected}
                    className={`flex items-center justify-center gap-2 p-4 rounded-xl transition-all ${
                      !isConnected 
                        ? 'bg-gray-500/50 cursor-not-allowed'
                        : isTranscribing 
                        ? 'bg-red-500/80 hover:bg-red-500' 
                        : 'bg-green-500/80 hover:bg-green-500'
                    } text-white font-medium`}
                  >
                    {isTranscribing ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                    {isTranscribing ? 'Stop' : 'Start'}
                  </motion.button>
                  
                  <div className={`flex items-center justify-center gap-2 p-4 rounded-xl ${
                    isAiSpeaking ? 'bg-purple-500/80' : 'bg-slate-600/50'
                  } text-white font-medium`}>
                    <Volume2 className="w-5 h-5" />
                    {isAiSpeaking ? 'Speaking' : 'Listening'}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right Side - Transcription & AI Response */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            {/* Transcription Panel */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-3xl blur-xl"></div>
              <div className="relative backdrop-blur-xl bg-white/10 rounded-3xl border border-white/20 overflow-hidden shadow-2xl">
                <div className="p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Waves className="w-5 h-5 text-blue-400" />
                    <h2 className="text-xl font-semibold text-white">Live Transcription</h2>
                    {isTranscribing && (
                      <motion.div
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="w-2 h-2 bg-blue-400 rounded-full ml-2"
                      />
                    )}
                  </div>
                  
                  <div className="min-h-[200px] bg-slate-800/50 rounded-2xl p-4 border border-slate-700/50">
                    <motion.p 
                      className="text-white text-lg leading-relaxed"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {displayText || (
                        <span className="text-slate-400 italic">
                          {isConnected 
                            ? (isTranscribing ? "Listening for speech..." : "Start transcription to begin")
                            : "Connecting to server..."
                          }
                        </span>
                      )}
                      {displayText && (
                        <motion.span
                          animate={{ opacity: [1, 0, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          className="inline-block w-1 h-6 bg-blue-400 ml-1"
                        />
                      )}
                    </motion.p>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Response Panel */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-3xl blur-xl"></div>
              <div className="relative backdrop-blur-xl bg-white/10 rounded-3xl border border-white/20 overflow-hidden shadow-2xl">
                <div className="p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Brain className="w-5 h-5 text-purple-400" />
                    <h2 className="text-xl font-semibold text-white">AI Response</h2>
                    {isAiSpeaking && (
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                        className="w-2 h-2 bg-purple-400 rounded-full ml-2"
                      />
                    )}
                  </div>
                  
                  <div className="min-h-[200px] bg-slate-800/50 rounded-2xl p-4 border border-slate-700/50">
                    <AnimatePresence>
                      {aiResponse ? (
                        <motion.p 
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="text-white text-lg leading-relaxed"
                        >
                          {aiResponse}
                        </motion.p>
                      ) : (
                        <motion.p 
                          className="text-slate-400 italic text-lg"
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 2, repeat: Infinity }}
                        >
                          AI is ready to respond...
                        </motion.p>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>

            {/* Processing Status */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/20 to-teal-500/20 rounded-2xl blur-xl"></div>
              <div className="relative backdrop-blur-xl bg-white/10 rounded-2xl border border-white/20 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-white font-medium">Processing Status</span>
                  <div className="flex items-center gap-2">
                    <motion.div 
                      className={`w-3 h-3 rounded-full ${
                        isAddressingRobot ? 'bg-green-400' : 'bg-slate-400'
                      }`}
                      animate={isAddressingRobot ? { 
                        boxShadow: ['0 0 0 0 rgba(34, 197, 94, 0.7)', '0 0 0 10px rgba(34, 197, 94, 0)'] 
                      } : {}}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                    <span className="text-sm text-slate-300">
                      {isAddressingRobot ? 'Active' : 'Standby'}
                    </span>
                  </div>
                </div>
                
                {confidence > 0 && (
                  <div className="mt-2">
                    <div className="flex justify-between text-sm text-slate-300 mb-1">
                      <span>Confidence</span>
                      <span>{(confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-2">
                      <motion.div 
                        className="bg-gradient-to-r from-green-400 to-emerald-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default AITranscriptionInterface;