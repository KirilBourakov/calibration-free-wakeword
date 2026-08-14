import sys
import time
import msvcrt
import numpy as np
from pydantic import TypeAdapter
from libemg.streamers import myo_streamer
from mci_wake.data_handler.online import CompatibleOnlineDataHandler
from mci_wake.data_handler.recording import RecordingFileContents, RecordingFileRegions

output_filename = r"D:\Coding\calibration-free-wakeword\mci_wake\recordings\pinchfirst\shake1.json"

def main():
    streamer, sm = myo_streamer()
    odh = CompatibleOnlineDataHandler(sm)

    emg_data = []
    timestamps = []
    regions: list[RecordingFileRegions] = []
    active_start_time = None

    print("Streaming live EMG data...")
    print("  Press 'r' - Mark region start / end time")
    print("  Press 'q' - Stop recording and save to JSON\n")

    last_count = 0
    start_time = time.time()
    running = True

    while running:
        # Listen for key press ('r' to mark time, 'q' to quit)
        if sys.platform == "win32" and msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            if key == "r":
                now = time.time() - start_time
                if active_start_time is None:
                    active_start_time = now
                    print(f"[MARK START] t = {active_start_time:.3f}s")
                else:
                    regions.append(RecordingFileRegions(start=active_start_time, end=now))
                    print(f"[MARK END]   t = {now:.3f}s (duration: {now - active_start_time:.3f}s)")
                    active_start_time = None
            elif key in ["q", "\x1b"]:
                running = False
                break

        # Read new EMG data from streamer
        dh_out = odh.get_data(N=0, filter=False)
        new_count = dh_out.count
        num_new = new_count - last_count
        if num_new > 0:
            new_samples = dh_out.emg[:num_new, :][::-1]
            now = time.time() - start_time
            emg_data.extend(new_samples.tolist())
            timestamps.extend([now] * num_new)
            last_count = new_count

        time.sleep(0.005)

    # Stop streamer process
    streamer.stop()

    # Save recording and marked regions into a single JSON file

    recording_data = RecordingFileContents(
        emg=np.asarray(emg_data, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        regions=regions,
    )

    with open(output_filename, "w") as f:
        f.write(TypeAdapter(RecordingFileContents).dump_json(recording_data, indent=2).decode())


    print(f"\nRecording finished. Saved to '{output_filename}' ({len(emg_data)} samples, {len(regions)} marked regions).")


if __name__ == "__main__":
    main()


