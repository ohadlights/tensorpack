import os
import argparse
from shutil import copy
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def copy_file(file, source_dir, target_dir):
    copy(os.path.join(source_dir, file),
         os.path.join(target_dir, file))


def main(args):
    files = os.listdir(args.source_dir)
    func = partial(copy_file, source_dir=args.source_dir, target_dir=args.target_dir)
    with Pool(args.num_processes) as p:
        for _ in tqdm(p.imap(func, files), total=len(files), desc='Copy files'):
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source_dir', required=True)
    p.add_argument('--target_dir', required=True)
    p.add_argument('--num_processes', type=int, default=10)
    main(p.parse_args())
