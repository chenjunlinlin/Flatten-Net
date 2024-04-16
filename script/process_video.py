import subprocess
import tqdm
import os


video_root = ['train']
out_root = ['../dataset/HMDB51/videos_img']
suffix = '.jpg'
flow_root = ['../dataset/train_flow']
flow_face_root = ['../dataset/trainface_flow']


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise

def extract_frames(filepath, suffix, index):

    if(os.path.isfile(filepath)):
        if(filepath.lower().endswith('.avi')):
            path, filename = os.path.split(filepath)
            filename = filename.replace(".avi","")
            destpath_rgb =  "/raid5/chenjunlin/code/flatten-net/dataset/HMDB51/videos_img"
            output_filedir = os.path.join(destpath_rgb, path, 
                                          filename).replace("/video", "/video_img")
            mkdir_p(output_filedir)
            command = f"ffmpeg -i {filepath} {output_filedir}/image_%05d{suffix}"
            subprocess.call(command, shell=True)
    else:
        allfiles = [f for f in os.listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            extract_frames(os.path.join(filepath, anyfile), suffix=suffix, index=index)

def main():
    for i , dir in enumerate(video_root):
        videos_path = "/raid5/chenjunlin/code/flatten-net/dataset/HMDB51/videos"
        for video_path in tqdm.tqdm(os.listdir(videos_path), desc=videos_path):
            if video_path != '__MACOSX' and video_path != '.ipynb_checkpoints':
                extract_frames(os.path.join(videos_path, video_path), suffix, i)


if __name__ == '__main__':
    main()
    print("finish!!!!!!!!!!!!!!!!!!")