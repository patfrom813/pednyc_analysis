import pandas as pd
import subprocess
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis")
SCRIPT_DIR = Path(__file__).resolve().parent

# Use the labeled timestamp/frame-by-frame CSV
csv_path = PROJECT_DIR / r"feature_outputs\features_PedNYC1_scenario3_smoothed_accel_ped_behavior_labeled.csv"

video_path = PROJECT_DIR / r"feature_outputs\plots\PedNYC1_scenario3_final_synced_4sec.mp4"

# Save outputs inside 4frame_view
ass_path = SCRIPT_DIR / "PedNYC1_scenario3_ped_behavior_overlay.ass"
output_path = SCRIPT_DIR / "PedNYC1_scenario3_final_synced_4sec_with_ped_behavior.mp4"

# Change this only if the overlay is shifted.
# Positive means behavior text appears later.
# Negative means behavior text appears earlier.
TIME_OFFSET = 0.0


def sec_to_ass_time(seconds):
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))

    if cs == 100:
        s += 1
        cs = 0

    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean_ass_text(text):
    text = str(text).strip()
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\n", " ")
    return text


df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print("CSV columns found:")
print(df.columns.tolist())

# Pick the time column
if "ScenarioTime" in df.columns:
    time_col = "ScenarioTime"
elif "scenario_time" in df.columns:
    time_col = "scenario_time"
elif "time" in df.columns:
    time_col = "time"
elif "start_time" in df.columns:
    time_col = "start_time"
else:
    raise ValueError("Could not find a time column. Expected ScenarioTime, scenario_time, time, or start_time.")

# Pick behavior column
if "ped_behavior_inferred" in df.columns:
    behavior_col = "ped_behavior_inferred"
elif "ped_behavior" in df.columns:
    behavior_col = "ped_behavior"
else:
    raise ValueError("Could not find pedestrian behavior column. Expected ped_behavior_inferred or ped_behavior.")

# Pick confidence column if it exists
if "ped_behavior_confidence" in df.columns:
    confidence_col = "ped_behavior_confidence"
else:
    confidence_col = None

df = df.sort_values(time_col).reset_index(drop=True)

# Remove rows where behavior is empty
df[behavior_col] = df[behavior_col].astype(str).str.strip()
df = df[df[behavior_col].notna()]
df = df[df[behavior_col] != ""]
df = df[df[behavior_col].str.lower() != "nan"]

# Create start/end time for each row.
# End time is the next row's timestamp.
df["start_time_overlay"] = df[time_col].astype(float)
df["end_time_overlay"] = df["start_time_overlay"].shift(-1)

# Last row ends at the video end / scenario end
df.loc[df.index[-1], "end_time_overlay"] = df["start_time_overlay"].iloc[-1] + 0.05

# Merge consecutive rows with the same behavior and confidence.
segments = []

current_behavior = None
current_confidence = None
current_start = None
current_end = None

for _, row in df.iterrows():
    behavior = clean_ass_text(row[behavior_col])
    confidence = clean_ass_text(row[confidence_col]) if confidence_col else ""

    start = float(row["start_time_overlay"])
    end = float(row["end_time_overlay"])

    if current_behavior is None:
        current_behavior = behavior
        current_confidence = confidence
        current_start = start
        current_end = end
        continue

    same_behavior = behavior == current_behavior
    same_confidence = confidence == current_confidence

    if same_behavior and same_confidence:
        current_end = end
    else:
        segments.append((current_start, current_end, current_behavior, current_confidence))
        current_behavior = behavior
        current_confidence = confidence
        current_start = start
        current_end = end

segments.append((current_start, current_end, current_behavior, current_confidence))

ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BehaviorStyle,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,1,0,0,0,100,100,0,0,3,2,0,7,25,25,25,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

events = []

for start, end, behavior, confidence in segments:
    start_ass = sec_to_ass_time(start + TIME_OFFSET)
    end_ass = sec_to_ass_time(end + TIME_OFFSET)

    if confidence:
        text = f"Ped behavior: {behavior}\\NConfidence: {confidence}"
    else:
        text = f"Ped behavior: {behavior}"

    events.append(
        f"Dialogue: 0,{start_ass},{end_ass},BehaviorStyle,,0,0,0,,{text}"
    )

ass_path.write_text(ass_header + "\n".join(events), encoding="utf-8")

print(f"\nCreated ASS overlay file:\n{ass_path}")
print(f"Number of behavior overlay segments created: {len(segments)}")

# Important Windows fix:
# Use only the ASS filename and run FFmpeg inside SCRIPT_DIR.
cmd = [
    "ffmpeg",
    "-y",
    "-i", str(video_path),
    "-vf", f"ass={ass_path.name}",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(output_path)
]

print("\nRunning FFmpeg...")
subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)

print(f"\nDone. Saved video here:\n{output_path}")