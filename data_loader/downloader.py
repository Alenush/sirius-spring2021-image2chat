from argparse import ArgumentParser
import urllib.request
import os
import pathlib
import time
#accepts as input the address of the file with the hashes of the images to be uploaded and
#the number of the part(0...9) to be uploaded
#the log file contains the addresses of the images that could not be uploaded
parser = ArgumentParser()
parser.add_argument('path_to_hashes')
parser.add_argument('item', type=int)
parser.add_argument('--images', default='./images/train/', help='path where save images')
args = parser.parse_args()

f = open('log.txt', 'w')
f.close()

url = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/'
start = time.time()
n = 186782
portion = n // 10
downloaded = len(os.listdir(args.images))

with open(args.path_to_hashes) as hashes_file:
    for i, current_hash in enumerate(hashes_file):
        if i < downloaded:
            continue
        if args.item * portion < i <= (args.item + 1) * portion:
            current_hash = current_hash.strip()
            folder1, folder2 = current_hash[:3], current_hash[3:6]
            folder_path = os.path.join(args.images, folder1, folder2)
            image_path = os.path.join(folder_path, current_hash) + ".jpg"
            if not os.path.isfile(image_path):
                if not os.path.isdir(folder_path):
                    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
                full_url = os.path.join(url, folder1, folder2, current_hash) + ".jpg"
                print("Download " + full_url, "in", image_path)
                try:
                    urllib.request.urlretrieve (full_url, image_path)
                except BaseException:
                    f = open('log.txt', 'a')
                    f.write(full_url+'\n')

                    f.close()
                    continue
print("total time", time.time() - start)
