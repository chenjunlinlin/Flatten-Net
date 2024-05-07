import os
import pandas as pd
from PIL import Image

list_file = "/raid5/chenjunlin/code/flatten-net/dataset/HMDB51/val.txt"
columns = ["path", "num_frame", "label"]
csv_path = "/raid5/chenjunlin/code/flatten-net/dataset/HMDB51/val_off.csv"
root_path = "/raid5/chenjunlin/code/flatten-net/dataset/HMDB51/video_imgs/"
image_tmpl='image_{:05d}.jpg'

if __name__=="__main__":
    tmp = [x.strip().split(' ') for x in open(list_file)]
    count=0
    for ind, video in enumerate(tmp):
        try:
            Image.open(os.path.join(root_path, video[0], image_tmpl.format(1))).convert('RGB')
        except Exception:
            count += 1
            print(f"------- {tmp[ind]} -------------")
            del tmp[ind]
            print('error loading image:', os.path.join(root_path, video[0]))

    csv = pd.DataFrame(tmp, columns=columns)
    
    print(f"count: {count}")
    csv.to_csv(path_or_buf=csv_path, index=False)