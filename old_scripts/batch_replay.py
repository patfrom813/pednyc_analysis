"""
batch_replay.py — Automates loading multiple .replay files in com.farlab.StrangeLand

SETUP (run once in a terminal with the game NOT open):
    pip install pyautogui pygetwindow

USAGE:
    1. Launch com.farlab.StrangeLand.exe and wait for it to fully load
    2. Edit the CONFIG section below (REPLAY_FOLDER, GAME_WINDOW_TITLE)
    3. Run: python batch_replay.py
    4. Click into the game window once, then don't touch the mouse/keyboard

HOW IT WORKS:
    - Reads replay duration directly from each .replay file header
    - Presses 'O' to open the in-game file browser
    - Types the full file path into the filename field and presses Enter
    - Waits for the replay to finish (duration + buffer), then loads the next file
    - Also watches the Unity Player.log for the NullReferenceException that
      signals a replay has ended — whichever comes first triggers the next file
"""

import os
import sys
import struct
import time
import glob
import threading

try:
    import pyautogui
    import pygetwindow as gw
except ImportError:
    print("Missing dependencies. Run:  pip install pyautogui pygetwindow")
    sys.exit(1)

# Disable pyautogui's fail-safe (moving mouse to screen corner would otherwise crash the script)
pyautogui.FAILSAFE = False

# ─────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────

# Root folder containing all scenario subfolders
REPLAY_FOLDER = r"C:\Users\FARLAB\Desktop\Studies_by_scenario"

# Which scenario folders to run — leave empty [] to run ALL scenarios.
# Example to run just two scenarios:
#   SCENARIOS = ["Ped-3", "Ped-7"]
SCENARIOS = ["Ped-21"]

# Run only a subset of the matched files (1-based index).
# Useful for resuming after a crash, e.g. if it quit at 15/30 set START_FROM = 15.
# Set to 1 and 0 to run everything.
START_FROM = 1   # first file to run (1 = start from the beginning)
END_AT     = 0   # last file to run  (0 = run to the end)

# Part of the game window title (enough to uniquely identify it)
GAME_WINDOW_TITLE = "StrangeLand"

# Extra seconds to wait after a replay ends before loading the next one
BUFFER_SECONDS = 5

# How many seconds to wait for the file browser to open after pressing 'O'
FILE_BROWSER_OPEN_WAIT = 2.0

# Set to True to skip files already processed (based on a log file)
SKIP_ALREADY_PROCESSED = False

# Between-scenario automation:
# After the last file in each scenario, the script will stop your Snipping Tool
# recording, then pause so you can adjust the quad view and start a new recording
# before the next scenario begins.
STOP_RECORDING_BETWEEN_SCENARIOS = True

# Unity Player.log location (auto-detected, but you can override)
UNITY_LOG_PATH = os.path.expandvars(
    r"%APPDATA%\..\LocalLow\farlab\com.farlab.StrangeLand\Player.log"
)

# ─────────────────────────────────────────────────────────────────


def get_replay_duration(replay_path: str) -> float:
    """Read replay duration (seconds) from the UR20 file header."""
    with open(replay_path, "rb") as f:
        header = f.read(50)
    # UR20 format: duration stored as little-endian float at byte offset 32
    magic = header[6:10]
    if magic != b"UR20":
        print(f"  WARNING: {os.path.basename(replay_path)} doesn't look like a UR20 replay file")
        return 60.0  # fallback
    duration = struct.unpack_from("<f", header, 32)[0]
    if duration <= 0 or duration > 3600:
        return 60.0  # fallback if header value is implausible
    return duration


def get_game_window():
    """Find and return the game window, or None if not found."""
    matches = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
    if not matches:
        # Try broader search
        all_windows = gw.getAllTitles()
        matches = [gw.getWindowsWithTitle(t)[0] for t in all_windows
                   if GAME_WINDOW_TITLE.lower() in t.lower() and t.strip()]
    return matches[0] if matches else None


def wait_for_replay_end_via_log(log_path: str, log_position: int, timeout: float) -> bool:
    """
    Monitor Unity Player.log for signs that a replay playback has finished.
    Looks for 'Stopped Recording' or 'NullReferenceException' appearing AFTER
    a 'RecordingToFile' line — meaning the playback completed and the app
    went idle. Returns True if detected within timeout, False otherwise.
    """
    deadline = time.time() + timeout
    # Wait a few seconds before starting to monitor, to let playback begin
    time.sleep(3.0)
    while time.time() < deadline:
        time.sleep(1.5)
        try:
            with open(log_path, "r", errors="ignore") as f:
                f.seek(log_position)
                new_content = f.read()
                # "Stopped Recording" followed by NullReference = playback ended
                if "Stopped Recording" in new_content and "NullReferenceException" in new_content:
                    return True
        except Exception:
            pass
    return False


def set_clipboard(text: str):
    """Copy text to Windows clipboard using clip.exe (no extra dependencies)."""
    import subprocess
    subprocess.run("clip", input=text.encode("utf-16le"), check=True)


def close_dev_console():
    """
    Close the Development Console that pops up after a replay ends.
    Clicks the Close button in the bottom-left console panel.
    """
    win = get_game_window()
    if win is None:
        return
    try:
        win.activate()
    except Exception:
        pass
    time.sleep(0.3)
    # "Close" button sits on the right edge of the console panel, near the bottom-left
    # Approximately 53% across, 96% down the window
    close_x = win.left + int(win.width * 0.53)
    close_y = win.top + int(win.height * 0.93)
    pyautogui.click(close_x, close_y)
    time.sleep(0.5)


def stop_snipping_tool_recording(save_name: str = None):
    """
    Stop the active Snipping Tool screen recording and save it with a custom name.
    save_name: filename to use (without extension), e.g. "ped3_all".
               If None, accepts whatever default name the dialog suggests.
    """
    toolbar = None
    for title in gw.getAllTitles():
        if title.strip().lower() == "recording toolbar":
            wins = gw.getWindowsWithTitle(title)
            if wins:
                toolbar = wins[0]
                break

    if not toolbar:
        print("  WARNING: Snipping Tool window not found — stop the recording manually.")
        return

    print(f"  Stopping recording — clicking Stop button...")
    try:
        toolbar.activate()
    except Exception:
        pass
    time.sleep(0.5)
    # Stop button is in the first quarter of the toolbar
    stop_x = toolbar.left + toolbar.width // 4
    stop_y = toolbar.top + toolbar.height // 2
    pyautogui.click(stop_x, stop_y)

    if save_name:
        time.sleep(1.5)  # Wait for the recording to fully stop
        # Press Ctrl+S to open the Save As dialog
        pyautogui.hotkey("ctrl", "s")
        time.sleep(2.0)  # Wait for the Save As dialog to appear
        # Filename field should be auto-focused — clear it and type the new name
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.typewrite(save_name, interval=0.05)
        time.sleep(0.2)
        pyautogui.press("enter")
        print(f"  Saved as: {save_name}")
        time.sleep(1.0)
    else:
        time.sleep(1.0)


def load_replay_file(replay_path: str) -> bool:
    """
    Bring the game window to focus, press 'O' to open the file browser,
    then paste the full path via clipboard and press Enter.
    Using clipboard paste (Ctrl+V) instead of typing to avoid the game's
    input manager intercepting individual keystrokes as game controls.
    """
    win = get_game_window()
    if win is None:
        print("  ERROR: Game window not found. Is the game running?")
        return False

    folder = os.path.dirname(replay_path)
    filename = os.path.basename(replay_path)

    # Copy folder path to clipboard BEFORE focusing the game window
    set_clipboard(folder)

    # Bring game window to foreground
    try:
        win.activate()
    except Exception:
        pass
    time.sleep(0.5)

    # Press 'O' to open file browser
    pyautogui.press("o")
    print(f"  Pressed 'O', waiting {FILE_BROWSER_OPEN_WAIT}s for file browser...")
    time.sleep(FILE_BROWSER_OPEN_WAIT)

    # Step 1: Click the top path bar (~55% x, ~21% y), select all, paste folder, Enter
    path_bar_x = win.left + int(win.width * 0.55)
    path_bar_y = win.top + int(win.height * 0.21)
    pyautogui.click(path_bar_x, path_bar_y)
    time.sleep(0.5)           # longer wait to ensure the field has focus
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(2.0)           # wait for folder to load and file list to populate

    # Step 2: Click the bottom filename field (~80% down), paste just the filename
    set_clipboard(filename)
    field_x = win.left + win.width // 2
    field_y = win.top + int(win.height * 0.80)
    pyautogui.click(field_x, field_y)
    time.sleep(0.4)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.4)

    # Step 3: Click the "Select" button (~73% across, ~87% down)
    select_x = win.left + int(win.width * 0.73)
    select_y = win.top + int(win.height * 0.87)
    pyautogui.click(select_x, select_y)

    return True


def run_batch(replay_files: list):
    """Load each replay file in sequence, waiting for each one to finish."""
    total = len(replay_files)
    # Build a lookup so we know when the last file of each scenario is reached
    scenario_last_idx = {}
    for i, path in enumerate(replay_files):
        scenario_last_idx[os.path.basename(os.path.dirname(path))] = i
    log_path = os.path.normpath(UNITY_LOG_PATH)
    log_exists = os.path.exists(log_path)

    if not log_exists:
        print(f"Note: Unity log not found at {log_path}")
        print("      Will use duration-based timing only.\n")

    processed_log = "batch_replay_progress.txt"

    for i, replay_path in enumerate(replay_files, 1):
        fname = os.path.basename(replay_path)
        scenario = os.path.basename(os.path.dirname(replay_path))
        rel_path = f"{scenario}/{fname}"

        # Optional: skip already-processed files
        if SKIP_ALREADY_PROCESSED and os.path.exists(processed_log):
            with open(processed_log) as pf:
                if rel_path in pf.read():
                    print(f"[{i}/{total}] Skipping (already processed): {rel_path}")
                    continue

        duration = get_replay_duration(replay_path)
        total_wait = duration + BUFFER_SECONDS
        mins = int(duration // 60)
        secs = int(duration % 60)
        print(f"\n[{i}/{total}] Loading: {rel_path}")
        print(f"          Duration: {mins}m {secs}s  (will wait {total_wait:.0f}s total)")

        # Get current log file size so we only look at new output
        log_position = 0
        if log_exists:
            try:
                log_position = os.path.getsize(log_path)
            except Exception:
                pass

        # Automate loading the file
        if not load_replay_file(replay_path):
            print("  Skipping due to window error.")
            continue

        print(f"  Replay started. Waiting up to {total_wait:.0f}s...")

        # Wait: log-based detection OR duration timeout, whichever comes first
        if log_exists:
            ended = wait_for_replay_end_via_log(log_path, log_position, total_wait)
            if ended:
                print(f"  Replay ended (detected via log). Waiting {BUFFER_SECONDS}s buffer...")
                time.sleep(BUFFER_SECONDS)
            else:
                print(f"  Duration elapsed ({total_wait:.0f}s). Moving on.")
        else:
            # No log available — just sleep for the full duration + buffer
            time.sleep(total_wait)

        # Close the Development Console that pops up after replay ends
        print("  Closing Development Console...")
        close_dev_console()

        # Record progress
        with open(processed_log, "a") as pf:
            pf.write(rel_path + "\n")

        # After the last file in a scenario, stop recording and pause for next scenario
        is_last_in_scenario = (i - 1 == scenario_last_idx[scenario])
        is_last_overall = (i == total)
        if STOP_RECORDING_BETWEEN_SCENARIOS and is_last_in_scenario and not is_last_overall:
            print(f"\n  ── End of scenario: {scenario} ──")
            print("  Stopping Snipping Tool recording...")
            save_name = scenario.lower().replace("-", "") + "_all"  # e.g. "ped3_all"
            stop_snipping_tool_recording(save_name)
            print("\n  ┌─────────────────────────────────────────────────┐")
            print(f"  │  Scenario '{scenario}' complete.                ")
            print("  │  1. Save your screen recording                  ")
            print("  │  2. Adjust the quad view for the next scenario  ")
            print("  │  3. Start a new Snipping Tool recording         ")
            print("  └─────────────────────────────────────────────────┘")
            input("  Press Enter when ready to continue to the next scenario...")
            print("  Resuming in 3 seconds...")
            time.sleep(3)

    # Stop recording after the very last file too
    if STOP_RECORDING_BETWEEN_SCENARIOS:
        last_scenario = os.path.basename(os.path.dirname(replay_files[-1]))
        save_name = last_scenario.lower().replace("-", "") + "_all"
        print(f"\n  Stopping final Snipping Tool recording...")
        stop_snipping_tool_recording(save_name)

    print(f"\n✓ Batch complete. Processed {total} replay files.")


def print_file_list(replay_files: list):
    """Print the list of files grouped by scenario folder."""
    print(f"\nFound {len(replay_files)} replay files in: {REPLAY_FOLDER}")
    total_duration = 0
    current_scenario = None
    for f in replay_files:
        scenario = os.path.basename(os.path.dirname(f))
        if scenario != current_scenario:
            print(f"\n  [{scenario}]")
            current_scenario = scenario
        d = get_replay_duration(f)
        total_duration += d
        mins, secs = int(d // 60), int(d % 60)
        print(f"    {os.path.basename(f):<55}  {mins}m {secs}s")
    total_mins = int(total_duration // 60)
    total_secs = int(total_duration % 60)
    buffer_total = BUFFER_SECONDS * len(replay_files)
    print(f"\n  Total replay time:   {total_mins}m {total_secs}s")
    print(f"  Total with buffers:  ~{int((total_duration + buffer_total) // 60)}m {int((total_duration + buffer_total) % 60)}s")


def natural_sort_key(path: str):
    """Sort paths so Ped-3 comes before Ped-12 (numeric-aware)."""
    import re
    parts = re.split(r'(\d+)', path)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def main():
    # Collect .replay files — either from specified scenarios or all subfolders
    if SCENARIOS:
        replay_files = []
        for scenario in SCENARIOS:
            folder = os.path.join(REPLAY_FOLDER, scenario)
            if not os.path.isdir(folder):
                print(f"WARNING: Scenario folder not found, skipping: {folder}")
                continue
            found = sorted(glob.glob(os.path.join(folder, "*.replay")), key=natural_sort_key)
            replay_files.extend(found)
        if not replay_files:
            print(f"No .replay files found for scenarios: {SCENARIOS}")
            sys.exit(1)
    else:
        pattern = os.path.join(REPLAY_FOLDER, "**", "*.replay")
        replay_files = sorted(glob.glob(pattern, recursive=True), key=natural_sort_key)
        if not replay_files:
            print(f"No .replay files found under: {REPLAY_FOLDER}")
            print("Make sure REPLAY_FOLDER is set correctly at the top of this script.")
            sys.exit(1)

    # Apply START_FROM / END_AT subset (convert 1-based to 0-based slice)
    start_idx = max(0, START_FROM - 1)
    end_idx   = END_AT if END_AT > 0 else len(replay_files)
    replay_files = replay_files[start_idx:end_idx]

    if not replay_files:
        print(f"No files in range START_FROM={START_FROM}, END_AT={END_AT}.")
        sys.exit(1)

    if START_FROM > 1 or END_AT > 0:
        print(f"Running subset: files {START_FROM}–{END_AT if END_AT > 0 else 'end'}")

    print_file_list(replay_files)

    print("\n─────────────────────────────────────────────────")
    print("Before starting:")
    print("  1. Make sure com.farlab.StrangeLand.exe is open and fully loaded")
    print("  2. Click into the game window once")
    print("  3. Do NOT move the mouse or type anything during the batch run")
    print("─────────────────────────────────────────────────")

    input("\nPress Enter to start the batch run (Ctrl+C to cancel at any time)...")

    # 3-second countdown so you can switch to the game window
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    run_batch(replay_files)


if __name__ == "__main__":
    main()