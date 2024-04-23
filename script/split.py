import os
import shutil
import random
import pandas as pd

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    class_folders = sorted(os.listdir(dataset_path))
    columns = ["path", "num_frame", "label"]
    train_csv = []
    val_csv = []
    test_csv = []
    

    for label, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        files = sorted(os.listdir(class_path))
        video_folders = []
        for file in files:
            if not file.endswith(".tform.mat"):
                video_folders.append(file)

        num_videos = len(video_folders)
        num_train = int(num_videos * train_ratio)
        num_val = int(num_videos * val_ratio)
        num_test = num_videos - num_train - num_val

        random.shuffle(video_folders)

        train_videos = video_folders[:num_train]
        val_videos = video_folders[num_train:num_train + num_val]
        test_videos = video_folders[num_train + num_val:]

        for video_folder in train_videos:
            src_path = os.path.join(class_path, video_folder)
            num_frame = len(os.listdir(src_path))
            path = src_path.replace("../dataset/", "./dataset/")
            train_csv.append([f"{path}", num_frame, label])
            # dest_path = os.path.join("train", class_folder, video_folder)
            # shutil.move(src_path, dest_path)

        for video_folder in val_videos:
            src_path = os.path.join(class_path, video_folder)
            num_frame = len(os.listdir(src_path))
            path = src_path.replace("../dataset/", "./dataset/")
            val_csv.append([f"{path}", num_frame, label])
            # dest_path = os.path.join("val", class_folder, video_folder)
            # shutil.move(src_path, dest_path)

        for video_folder in test_videos:
            src_path = os.path.join(class_path, video_folder)
            num_frame = len(os.listdir(src_path))
            path = src_path.replace("../dataset/", "./dataset/")
            test_csv.append([f"{path}", num_frame, label])
            # dest_path = os.path.join("test", class_folder, video_folder)
            # shutil.move(src_path, dest_path)
    train_df = pd.DataFrame(train_csv, columns=columns)
    val_df = pd.DataFrame(val_csv, columns=columns)
    test_df = pd.DataFrame(test_csv, columns=columns)

    return train_df, val_df, test_df

if __name__ == "__main__":
    dataset_path = "../dataset/HMDB51/video_imgs"
    train_ratio = 0.8
    val_ratio = 0
    test_ratio = 0.2
    train_path = "../dataset/HMDB51/train.csv"
    val_path = "../dataset/HMDB51/val.csv"
    test_path = "../dataset/HMDB51/test.csv"

    train_df, val_df, test_df = split_dataset(dataset_path, train_ratio,
                                               val_ratio, test_ratio)
    train_df.to_csv(path_or_buf=train_path, index=False)
    val_df.to_csv(path_or_buf=val_path, index=False)
    test_df.to_csv(path_or_buf=test_path, index=False)
    