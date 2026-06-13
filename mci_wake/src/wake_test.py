import random
import time
import winsound
import os
import sys
import msvcrt

class WakeTest:
    """
    A class for testing 'Close' wakeword detection by listening to keystrokes.
    Functions similarly to discrete_test.py but beeps on 'Close' predictions
    and does not run the model locally. Instead, it expects predictions 
    (keystrokes) from an external script like run_discrete.py.
    """
    def __init__(self, num_trials=25):
        self.classes = ['Close', 'Extension', 'Flexion', 'Open', 'Pinch']
        # Mapping keystrokes (sent by libemg.DiscreteControl) to class labels
        self.key_map = {
            'c': 'Close',
            'e': 'Extension',
            'f': 'Flexion',
            'o': 'Open',
            'p': 'Pinch'
        }
        self.num_trials = num_trials
        # Track trials per class to match discrete_test.py behavior
        self.trials = [0] * len(self.classes)
        self.next_trial = True
        self.time = -1
        self.spawn_time = random.uniform(1, 2)
        self.target_id = -1
        self.target_shown = False

    def run(self):
        print("\n=== Wake Test (Console Listener) ===")
        print("Functioning like discrete_test.py, but with sound and no GUI.")
        print("Predictions are received as keystrokes (e.g., from run_discrete.py).")
        print("Beep will trigger ONLY when 'Close' is predicted.")
        print("-" * 40)
        print("INSTRUCTIONS:")
        print("1. Run run_discrete.py in another terminal.")
        print("2. Focus THIS terminal window.")
        print("3. Perform the prompted gestures.")
        print("Press Ctrl+C to exit at any time.")
        print("-" * 40)

        try:
            while sum(self.trials) < len(self.classes) * self.num_trials:
                if self.next_trial:
                    # Pick a gesture that hasn't reached num_trials
                    self.target_id = random.randint(0, len(self.classes) - 1)
                    while self.trials[self.target_id] >= self.num_trials:
                        self.target_id = random.randint(0, len(self.classes) - 1)
                    
                    self.spawn_time = random.uniform(1, 2)
                    self.time = time.time()
                    self.next_trial = False
                    self.target_shown = False
                    
                    total_done = sum(self.trials)
                    total_needed = self.num_trials * len(self.classes)
                    print(f"\n[Progress: {total_done}/{total_needed}] Waiting for next prompt...")

                # Prompt logic (spawning)
                if not self.target_shown and (time.time() - self.time >= self.spawn_time):
                    print(f"\n*** PERFORM {self.classes[self.target_id].upper()} ***")
                    self.target_shown = True

                # Listen for global-ish keys (requires console focus)
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8').lower()
                    
                    if char in self.key_map:
                        predicted_label = self.key_map[char]
                        # "it makes a sound whenever Close is predicted"
                        if predicted_label == 'Close':
                            print(">> Beep! 'Close' predicted.")
                            winsound.Beep(1000, 250)
                        else:
                            print(f">> Prediction received: {predicted_label}")
                        
                        # Check if the prediction matches the target gesture
                        # Mapping: 'c'->'Close', 'e'->'Extension', etc.
                        target_char = self.classes[self.target_id][0].lower()
                        if char == target_char:
                            print(f"SUCCESS: {predicted_label} correctly detected.")
                            self.trials[self.target_id] += 1
                            self.next_trial = True
                            time.sleep(1.0) # Reset period
                        else:
                            print(f"MISMATCH: Expected {self.classes[self.target_id]}, but got {predicted_label}.")
                    
                    elif char == '\x03': # Ctrl+C manual handle
                        raise KeyboardInterrupt
                
                time.sleep(0.01)

            print("\n=== Wake Test Complete ===")
            print(f"Finished {sum(self.trials)} trials successfully.")

        except KeyboardInterrupt:
            print("\nStopping wake test...")

if __name__ == "__main__":
    # Default to evaluation mode (25 trials per class)
    wt = WakeTest(num_trials=25)
    wt.run()
