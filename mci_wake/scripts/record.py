import sys
import time
import json
import msvcrt
from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler


def main():
    # Initialize libemg streamer and data handler
    streamer, sm = myo_streamer()
    odh = OnlineDataHandler(sm)

    emg_data = []
    timestamps = []
    regions = []
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
                    regions.append({"start_time": active_start_time, "end_time": now})
                    print(f"[MARK END]   t = {now:.3f}s (duration: {now - active_start_time:.3f}s)")
                    active_start_time = None
            elif key in ["q", "\x1b"]:
                running = False
                break

        # Read new EMG data from streamer
        val, count = odh.get_data(N=0, filter=False)
        new_count = count["emg"][0, 0]
        num_new = new_count - last_count
        if num_new > 0:
            new_samples = val["emg"][:num_new, :][::-1]
            now = time.time() - start_time
            emg_data.extend(new_samples.tolist())
            timestamps.extend([now] * num_new)
            last_count = new_count

        time.sleep(0.005)

    # Stop streamer process
    if hasattr(streamer, "stop"):
        try:
            streamer.stop()
        except Exception:
            pass

    # Save recording and marked regions into a single JSON file
    output_filename = "emg_recording.json"
    data = {
        "emg": emg_data,
        "timestamps": timestamps,
        "regions": regions,
    }

    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nRecording finished. Saved to '{output_filename}' ({len(emg_data)} samples, {len(regions)} marked regions).")


if __name__ == "__main__":
    main()
