import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\feature_outputs\features_PedNYC1_scenario3_smoothed_accel.csv"
)

OUTPUT_DIR = CSV_PATH.parent / "unsupervised_autoencoder_outputs" / "PedNYC1_scenario3_w30_s5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
STRIDE = 5

KERNEL_SIZES = [3, 5, 9]   # multi-scale filters
LATENT_DIM = 16
N_CLUSTERS = 5

EPOCHS = 150
BATCH_SIZE = 16
RANDOM_STATE = 42

USE_YAW_SIN_COS = True


# ============================================================
# COLUMNS TO IGNORE
# ============================================================

IGNORE_COLUMNS = {
    "ped_stopped",
    "ped_walking",
    "car_slowing",
    "close_interaction",
    "ped_possible_noise",
    "car_possible_noise",
}

# These columns are useful for time/output, but should not be model features.
TIME_COLUMNS = {
    "ScenarioTime",
    "GameTime",
    "Frame Number",
    "FrameRate",
    "FrameRate-XRDevice",
}

# Candidate movement features.
# The script will only use columns that actually exist in the CSV.
CANDIDATE_FEATURE_COLUMNS = [
    "dt",

    "car_x",
    "car_z",
    "car_yaw",
    "A_velocity_raw",
    "A_accel_raw",
    "A_steering",
    "A_indicators",
    "A_horn",

    "ped_x",
    "ped_z",
    "ped_yaw",

    "ped_avatar_x",
    "ped_avatar_z",
    "ped_avatar_yaw",

    "car_speed_xz",
    "ped_speed_xz",
    "car_speed_xz_smooth",
    "ped_speed_xz_smooth",

    "car_accel_xz",
    "ped_accel_xz",

    "car_ped_distance_xz",
    "distance_change",
    "distance_closing",

    "avatar_vr_gap_xz",

    "car_accel_xz_raw_unsmoothed",
    "ped_accel_xz_raw_unsmoothed",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# LOAD + CLEAN DATA
# ============================================================

print(f"Reading CSV:\n{CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Clean header whitespace
df.columns = df.columns.str.strip()

if "ScenarioTime" not in df.columns:
    raise ValueError("ScenarioTime column is required for time outputs.")

# Remove unreliable labels if present
df = df.drop(columns=[c for c in IGNORE_COLUMNS if c in df.columns], errors="ignore")

# Build usable feature list
feature_cols = [c for c in CANDIDATE_FEATURE_COLUMNS if c in df.columns]

if len(feature_cols) == 0:
    raise ValueError("No usable feature columns found. Check CSV headers.")

print("\nInitial selected feature columns:")
for c in feature_cols:
    print(" -", c)


# ============================================================
# OPTIONAL: CONVERT YAW ANGLES TO SIN/COS
# ============================================================

# Raw yaw angles can jump from 359 to 0 degrees, which looks like a huge
# artificial movement to a neural network. sin/cos avoids that wraparound issue.
if USE_YAW_SIN_COS:
    yaw_cols = [c for c in ["car_yaw", "ped_yaw", "ped_avatar_yaw"] if c in feature_cols]

    for col in yaw_cols:
        radians = np.deg2rad(pd.to_numeric(df[col], errors="coerce"))
        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)

    feature_cols = [c for c in feature_cols if c not in yaw_cols]
    feature_cols += [f"{c}_sin" for c in yaw_cols]
    feature_cols += [f"{c}_cos" for c in yaw_cols]


# ============================================================
# NUMERIC CONVERSION + MISSING VALUE HANDLING
# ============================================================

feature_df = pd.DataFrame(index=df.index)

for col in feature_cols:
    s = df[col]

    # Handle possible boolean-looking strings
    if s.dtype == "object":
        s = s.replace({
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "YES": 1,
            "NO": 0,
            "yes": 1,
            "no": 0,
        })

    feature_df[col] = pd.to_numeric(s, errors="coerce")

# Drop columns that are completely empty after numeric conversion
all_nan_cols = feature_df.columns[feature_df.isna().all()].tolist()
if all_nan_cols:
    print("\nDropping all-empty/non-numeric feature columns:")
    for c in all_nan_cols:
        print(" -", c)
    feature_df = feature_df.drop(columns=all_nan_cols)

# Interpolate missing values inside the time-series
feature_df = feature_df.interpolate(method="linear", limit_direction="both")

# Fill anything remaining
feature_df = feature_df.ffill().bfill().fillna(0)

final_feature_cols = feature_df.columns.tolist()

print("\nFinal model feature columns:")
for c in final_feature_cols:
    print(" -", c)

print(f"\nNumber of rows: {len(feature_df)}")
print(f"Number of features: {len(final_feature_cols)}")


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()
scaled_values = scaler.fit_transform(feature_df.values)

scaler_params = pd.DataFrame({
    "feature": final_feature_cols,
    "mean": scaler.mean_,
    "scale": scaler.scale_,
})
scaler_params.to_csv(OUTPUT_DIR / "feature_scaler_params.csv", index=False)


# ============================================================
# CREATE SLIDING WINDOWS
# ============================================================

def make_windows(values, times, window_size, stride):
    X = []
    window_records = []

    n = len(values)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = values[start:end]

        start_time = times[start]
        end_time = times[end - 1]
        mid_time = times[start + window_size // 2]

        X.append(window)

        window_records.append({
            "window_id": len(window_records),
            "start_row": start,
            "end_row": end - 1,
            "start_time": start_time,
            "end_time": end_time,
            "mid_time": mid_time,
        })

    return np.array(X, dtype=np.float32), pd.DataFrame(window_records)


times = pd.to_numeric(df["ScenarioTime"], errors="coerce").interpolate().ffill().bfill().values

X, window_df = make_windows(
    scaled_values,
    times,
    WINDOW_SIZE,
    STRIDE
)

if len(X) < N_CLUSTERS:
    print(f"\nWarning: only {len(X)} windows found. Reducing clusters.")
    N_CLUSTERS = max(1, len(X))

print(f"\nWindow shape: {X.shape}")
print(f"Number of windows: {len(X)}")


# ============================================================
# BUILD MULTI-SCALE 1D CONV AUTOENCODER
# ============================================================

def build_multiscale_conv_autoencoder(window_size, n_features, kernel_sizes, latent_dim):
    input_layer = layers.Input(shape=(window_size, n_features), name="input_window")

    branches = []

    for k in kernel_sizes:
        if k <= window_size:
            branch = layers.Conv1D(
                filters=32,
                kernel_size=k,
                padding="same",
                activation="relu",
                name=f"encoder_conv_k{k}"
            )(input_layer)

            branch = layers.Conv1D(
                filters=32,
                kernel_size=k,
                padding="same",
                activation="relu",
                name=f"encoder_conv_k{k}_second"
            )(branch)

            branches.append(branch)

    if len(branches) == 0:
        raise ValueError("No valid kernel sizes. Use smaller kernel sizes.")

    if len(branches) > 1:
        x = layers.Concatenate(name="multi_scale_concat")(branches)
    else:
        x = branches[0]

    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="encoder_mixing_conv"
    )(x)

    x = layers.GlobalAveragePooling1D(name="temporal_global_pool")(x)

    latent = layers.Dense(
        latent_dim,
        activation="relu",
        name="latent_embedding"
    )(x)

    # Decoder
    x = layers.Dense(
        window_size * 64,
        activation="relu",
        name="decoder_dense"
    )(latent)

    x = layers.Reshape((window_size, 64), name="decoder_reshape")(x)

    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="decoder_conv_1"
    )(x)

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="decoder_conv_2"
    )(x)

    output_layer = layers.Conv1D(
        filters=n_features,
        kernel_size=1,
        padding="same",
        activation="linear",
        name="reconstructed_window"
    )(x)

    autoencoder = models.Model(
        inputs=input_layer,
        outputs=output_layer,
        name="multiscale_1d_conv_autoencoder"
    )

    encoder = models.Model(
        inputs=input_layer,
        outputs=latent,
        name="encoder_model"
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return autoencoder, encoder


n_features = X.shape[2]

autoencoder, encoder = build_multiscale_conv_autoencoder(
    window_size=WINDOW_SIZE,
    n_features=n_features,
    kernel_sizes=KERNEL_SIZES,
    latent_dim=LATENT_DIM
)

autoencoder.summary()


# ============================================================
# TRAIN ON WHOLE CSV — NO VALIDATION SPLIT
# ============================================================

early_stop = callbacks.EarlyStopping(
    monitor="loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

history = autoencoder.fit(
    X,
    X,
    epochs=EPOCHS,
    batch_size=min(BATCH_SIZE, len(X)),
    shuffle=True,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# ============================================================
# SAVE TRAINING LOSS PLOT
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training reconstruction loss")
plt.title("Autoencoder Training Loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "autoencoder_training_loss.png", dpi=200)
plt.close()


# ============================================================
# LATENT EMBEDDINGS
# ============================================================

embeddings = encoder.predict(X)

embedding_cols = [f"latent_{i}" for i in range(embeddings.shape[1])]
embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
embedding_df.insert(0, "window_id", window_df["window_id"].values)
embedding_df.insert(1, "mid_time", window_df["mid_time"].values)
embedding_df.to_csv(OUTPUT_DIR / "window_latent_embeddings.csv", index=False)


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

X_reconstructed = autoencoder.predict(X)

reconstruction_error = np.mean(
    np.square(X - X_reconstructed),
    axis=(1, 2)
)

window_df["reconstruction_error"] = reconstruction_error


# ============================================================
# CLUSTER LATENT EMBEDDINGS
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

cluster_ids = kmeans.fit_predict(embeddings)

window_df["cluster_id"] = cluster_ids

# Requested output #1
window_output = window_df[
    [
        "window_id",
        "start_time",
        "end_time",
        "mid_time",
        "cluster_id",
        "reconstruction_error",
        "start_row",
        "end_row",
    ]
]

window_output.to_csv(
    OUTPUT_DIR / "window_behavior_clusters.csv",
    index=False
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

summary_basic = window_df.groupby("cluster_id").agg(
    n_windows=("window_id", "count"),
    first_mid_time=("mid_time", "min"),
    last_mid_time=("mid_time", "max"),
    mean_reconstruction_error=("reconstruction_error", "mean"),
    median_reconstruction_error=("reconstruction_error", "median"),
    max_reconstruction_error=("reconstruction_error", "max"),
).reset_index()

# Add interpretable feature averages using the center row of each window
mid_rows = []
for _, row in window_df.iterrows():
    mid_row = int((row["start_row"] + row["end_row"]) // 2)
    mid_rows.append(mid_row)

mid_feature_df = feature_df.iloc[mid_rows].reset_index(drop=True)
mid_feature_df["cluster_id"] = cluster_ids

summary_features = mid_feature_df.groupby("cluster_id").mean(numeric_only=True).reset_index()

cluster_summary = summary_basic.merge(
    summary_features,
    on="cluster_id",
    how="left"
)

# Requested output #2
cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_summary.csv",
    index=False
)


# ============================================================
# PLOT 1: CLUSTERS OVER TIME
# ============================================================

plt.figure(figsize=(11, 5))
plt.scatter(
    window_df["mid_time"],
    window_df["cluster_id"],
    c=window_df["cluster_id"],
    s=45
)
plt.xlabel("ScenarioTime")
plt.ylabel("Cluster ID")
plt.title("Behavior Clusters Over Time")
plt.yticks(sorted(window_df["cluster_id"].unique()))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "clusters_over_time.png", dpi=200)
plt.close()


# ============================================================
# PLOT 2: RECONSTRUCTION ERROR OVER TIME
# ============================================================

plt.figure(figsize=(11, 5))
plt.plot(
    window_df["mid_time"],
    window_df["reconstruction_error"],
    marker="o"
)
plt.xlabel("ScenarioTime")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error Over Time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "reconstruction_error_over_time.png", dpi=200)
plt.close()


# ============================================================
# PLOT 3: PCA OF LATENT EMBEDDINGS
# ============================================================

if embeddings.shape[1] >= 2 and len(embeddings) >= 2:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame({
        "window_id": window_df["window_id"],
        "mid_time": window_df["mid_time"],
        "pca_1": pca_xy[:, 0],
        "pca_2": pca_xy[:, 1],
        "cluster_id": cluster_ids,
        "reconstruction_error": reconstruction_error,
    })

    pca_df.to_csv(OUTPUT_DIR / "behavior_embedding_pca.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_df["pca_1"],
        pca_df["pca_2"],
        c=pca_df["cluster_id"],
        s=55
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Latent Behavior Embeddings PCA")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "behavior_embedding_pca.png", dpi=200)
    plt.close()

else:
    print("Skipping PCA plot because there are not enough embeddings.")


# ============================================================
# SAVE MODEL
# ============================================================

autoencoder.save(OUTPUT_DIR / "multiscale_conv_autoencoder.keras")
encoder.save(OUTPUT_DIR / "encoder_model.keras")


# ============================================================
# DONE
# ============================================================

print("\nDone.")
print(f"Outputs saved to:\n{OUTPUT_DIR}")

print("\nMain outputs:")
print(" - window_behavior_clusters.csv")
print(" - cluster_summary.csv")
print(" - clusters_over_time.png")
print(" - reconstruction_error_over_time.png")
print(" - behavior_embedding_pca.png")
print(" - autoencoder_training_loss.png")
print(" - window_latent_embeddings.csv")
print(" - feature_scaler_params.csv")