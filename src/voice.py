"""
Voice Guidance Module
Provides non-blocking, thread-safe spoken auditory guidance for navigation events.
"""

import queue
import threading
import time


class VoiceAssistant:
    """Handles text-to-speech for navigation instructions."""
    
    def __init__(self, speech_rate=160, volume=1.0):
        self.enabled = False
        self.instruction_queue = queue.Queue()
        self.last_instruction = None
        self.last_instruction_time = 0
        self.min_instruction_interval = 3.0  # Minimum seconds between duplicate instructions
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', speech_rate)
            self.engine.setProperty('volume', volume)
            self.enabled = True
            
            # Start the TTS worker thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
        except Exception as e:
            print(f"      ℹ Audio voice assistant disabled (TTS initialization notice: {e})")
            self.engine = None
    
    def _tts_worker(self):
        """Worker thread for background TTS processing."""
        while self.enabled:
            try:
                instruction = self.instruction_queue.get(timeout=1.0)
                if instruction is None:
                    break
                
                if self.engine:
                    self.engine.say(instruction)
                    self.engine.runAndWait()
                
                self.instruction_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
                time.sleep(0.1)
    
    def speak_instruction(self, instruction, force=False):
        """Queue a spoken navigation instruction."""
        if not self.enabled or not instruction:
            return
        
        current_time = time.time()
        
        # Avoid rapid repeated identical instructions unless forced
        if not force and instruction == self.last_instruction:
            if current_time - self.last_instruction_time < self.min_instruction_interval:
                return
        
        self.last_instruction = instruction
        self.last_instruction_time = current_time
        self.instruction_queue.put(instruction)
    
    def stop(self):
        """Stop the voice assistant worker."""
        if self.enabled:
            self.enabled = False
            self.instruction_queue.put(None)
