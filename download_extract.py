from __future__ import print_function

from converter import WaymoToKITTI
import argparse
import os
import glob
from multiprocessing import Pool
import subprocess

import time




if __name__ == '__main__':




    parser = argparse.ArgumentParser()

    parser.add_argument('split', help='training, validation')
    parser.add_argument('tmp_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--num_proc', default=1, type=int, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = WaymoToKITTI(args.tmp_dir, os.path.join(args.save_dir, args.split))


    url = f'gs://waymo_open_dataset_v_1_2_0_individual_files/{args.split}/'

    files = subprocess.check_output(f'gsutil ls {url}', shell=True, text=True)

    files = files.split("\n")

    print(f"Found {len(files)} Records...")

    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)

    def convert_file(file, tmp_dir, save_dir):
        converter = WaymoToKITTI(tmp_dir, save_dir)

        print("convert : ", file)
        converter.convert_file(file)
        os.system(f'rm {file}')
        os.system(f'touch {file}.done')

        print("... done : ", file)

    with Pool(args.num_proc) as pool:

        for filepath in files:

            filename = filepath.split("/")[-1]
            local_filepath = os.path.join(args.tmp_dir, filename)

            print(local_filepath)

            if os.path.isfile(local_filepath+".done"):
                print("... already done")
                continue

            if not os.path.isfile(local_filepath):
                flag = os.system('gsutil cp ' + filepath + ' ' + args.tmp_dir)
                assert flag == 0, 'Failed to download segment %d. Make sure gsutil is installed'%filepath

            pool.apply_async(convert_file, (local_filepath, args.tmp_dir, os.path.join(args.save_dir, args.split)))
            #convert_file(local_filepath, args.tmp_dir, os.path.join(args.save_dir, args.split))