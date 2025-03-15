import pyttsx3
import playsound

engine = pyttsx3.Engine()
engine.save_to_file("Hi how are you", "sample.wav")
engine.runAndWait()
playsound.playsound("./sample.wav")


# import threading
# import time
# import playsound

# lock = threading.Condition()
# # lock = threading.Lock()

# def transcription_thread():
#     """Simulates the transcription process but waits if locked."""
#     while True:
#         with lock:
#             print("Transcription running...")
#             time.sleep(1)

# def ai_response_thread(audio_file):
#     """Plays AI response audio while locking transcription."""
#     while True:
#         print("AI Response: Pausing Transcription")
#         with lock:
#             playsound.playsound(audio_file)  # This will block until playback is done
#         print("AI Response: Resuming Transcription")

# t1 = threading.Thread(target=transcription_thread, daemon=False)
# t1.start()

# time.sleep(3)
# t2 = threading.Thread(target=ai_response_thread, args=("sample.wav",))
# t2.start()

# # t2.join()
