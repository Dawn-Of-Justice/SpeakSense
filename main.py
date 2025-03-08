import threading
# from Live_transcription.Transcription import WhisperRealtimeTranscriber
# from Live_transcription.Transcription2 import RealTimeTranscriber
from Live_transcription.OnlineTranscription import RealtimeTranscriber
from audio_model.Classifier import AddressClassifier
import time
from LIGHT_ASD.realtime import RealtimeASD, main
from LLM import AI

# A state variable as of now set to True, but will be modifed by the ASD function thread based on ASD Models prediction
state = True 
result = None

chat = AI()
def transcription():
    global result
    
    with RealtimeTranscriber(prompt="This is a tech discussion, listen to english only") as transcriber:
        print("Speak now! (Press Ctrl+C to stop)")
        try:
            while True:
                result = transcriber.get_transcription()
                # if result:
                #     print(f"Transcription: {result}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
def ASD():
    # ASD realtime code goes here
    main()
                
def Addressing():
    # The Classifier model works here
    global state, result
    classifier = AddressClassifier()
    context = ""
    while True:
        
        if result:
            context += result
            
            if state:
                out = classifier.classify_text(context)
                if out["is_addressing_robot"] and out['confidence'] > 0.6:
                    print(f"Context: {context}")
                    response = chat.generate_response(prompt=f"Input: {context}", 
                        system_message =  "You are an AI model who gives to Live transcription message when you are addressed\n\ninput: I was reading about the use of AI in agriculture. What do you think,\noutput: <response>")
                    print(f"AI:{response}")
                    context = ""
                    
            if len(context) >= 100:
                context = ""
            
if __name__ == "__main__":
    transcription_thread = threading.Thread(target=transcription)
    Addressing_thread = threading.Thread(target=Addressing)
    # ASD_thread = threading.Thread(target=main)
    # Threads are getting started
    
    
    transcription_thread.start()
    Addressing_thread.start()