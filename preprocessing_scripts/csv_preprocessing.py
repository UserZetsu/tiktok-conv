import os
import pandas as pd


# define folders with their labels
task_folders = [
    ('./rat_dance_csv/train', 1),
    ('./rat_dance_csv/val', 1),
    ('./rat_dance_csv/test', 1),
    ('./neg_control_csv/train', 0),
    ('./neg_control_csv/val', 0),
    ('./neg_control_csv/test', 0)
]

# column headers excluding "label"
columns = [
    "frame", "x_pose_0", "y_pose_0", "visibility_pose_0", "x_pose_1", "y_pose_1", "visibility_pose_1",
    "x_pose_2", "y_pose_2", "visibility_pose_2", "x_pose_3", "y_pose_3", "visibility_pose_3",
    "x_pose_4", "y_pose_4", "visibility_pose_4", "x_pose_5", "y_pose_5", "visibility_pose_5",
    "x_pose_6", "y_pose_6", "visibility_pose_6", "x_pose_7", "y_pose_7", "visibility_pose_7",
    "x_pose_8", "y_pose_8", "visibility_pose_8", "x_pose_9", "y_pose_9", "visibility_pose_9",
    "x_hand1_0", "y_hand1_0", "z_hand1_0", "x_hand1_1", "y_hand1_1", "z_hand1_1",
    "x_hand1_2", "y_hand1_2", "z_hand1_2", "x_hand1_3", "y_hand1_3", "z_hand1_3",
    "x_hand1_4", "y_hand1_4", "z_hand1_4", "x_hand1_5", "y_hand1_5", "z_hand1_5",
    "x_hand2_0", "y_hand2_0", "z_hand2_0", "x_hand2_1", "y_hand2_1", "z_hand2_1",
    "x_hand2_2", "y_hand2_2", "z_hand2_2", "x_hand2_3", "y_hand2_3", "z_hand2_3",
    "x_hand2_4", "y_hand2_4", "z_hand2_4", "x_hand2_5", "y_hand2_5", "z_hand2_5"
]

def adjust_csv_length(file_path):
    """ ensures each CSV has exactly 355 rows cutting or padding with zeros if needed """
    df = pd.read_csv(file_path)

    # ensure correct headers (excluding label)
    if list(df.columns)[-len(columns):] != columns:
        print(f"skipping {file_path} column mismatch")
        return

    # drop label column if it exists (to re-add later)
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # adjust frame column values
    df["frame"] = range(len(df))

    if len(df) > 355:
        df = df.iloc[:355]  # trim extra rows
    elif len(df) < 355:
        num_missing = 355 - len(df)
        new_rows = pd.DataFrame(0, index=range(num_missing), columns=columns)
        new_rows["frame"] = range(len(df), 355)  # ensure frame continues counting
        df = pd.concat([df, new_rows], ignore_index=True)

    return df

def add_label_column(csv_folder, label):
    """ adds a label column to all CSV files in folder after adjusting row count """
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            df = adjust_csv_length(file_path)  # adjust rows first

            if df is not None:
                df.insert(0, "label", label)  # first column
                df.to_csv(file_path, index=False)
                print(f"updated {file_path}: 355 rows with label {label}")

# process each folder
for folder, label in task_folders:
    if os.path.exists(folder):
        add_label_column(folder, label)
    else:
        print(f"folder {folder} does not exist")