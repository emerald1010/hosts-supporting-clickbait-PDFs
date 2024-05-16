import tqdm as tqdm

from Utilities.Files.IO import filename_path_decode
import shutil
import os
from pathlib import Path
from Utilities.Files.Screenshots import PDFPPM, produce_thumbnail
from multiprocessing import Pool
import functools
from Confs.Configs import Configs
from Datalake.Phishing.Phishing import PhishingDataLake


def process_sample(filehash,root_folder,screenshots_files_endpoint,thumbnail_screenshot_entrypoint, tmp_old):
    screenshot_destination_folder_path = filename_path_decode(filehash, phishing_datalake.screenshots_files_endpoint)
    screenshot_destination_path = os.path.join(screenshot_destination_folder_path, filehash + '.png')

    if Path(screenshot_destination_path).exists():
        shutil.move(screenshot_destination_path, os.path.join(tmp_old, filehash + '.png'))

    create_screenshot(filehash,root_folder,screenshots_files_endpoint,thumbnail_screenshot_entrypoint)


def create_screenshot(filehash, root_folder,screenshots_files_endpoint,thumbnail_screenshot_entrypoint):
    pdf_dest_folder_path = filename_path_decode(filehash, root_folder)
    pdf_sample_path = os.path.join(pdf_dest_folder_path, filehash)

    screenshot_destination_folder_path = filename_path_decode(filehash, screenshots_files_endpoint)
    screenshot_destination_path = os.path.join(screenshot_destination_folder_path, filehash + '.png')

    thumbnail_destination_folder_path = filename_path_decode(filehash,
                                                             thumbnail_screenshot_entrypoint)
    thumbnail_destination_path = os.path.join(thumbnail_destination_folder_path, filehash + '.png')

    if not Path(screenshot_destination_path).exists():

        ppm = PDFPPM(pdf_sample_path, '/tmp/tmp-screenshots/', filehash)
        if hasattr(ppm, 'images') and len(ppm.images) > 0:
            ppm_img_path = ppm.images[0]

            if not Path(screenshot_destination_folder_path).exists():
                os.makedirs(screenshot_destination_folder_path)

            if not Path(screenshot_destination_path).exists():
                shutil.move(ppm_img_path, screenshot_destination_path)


    if not Path(thumbnail_destination_path).exists() and Path(screenshot_destination_path).exists():
        produce_thumbnail(screenshot_destination_path, thumbnail_destination_path)


config = Configs("config.yaml", debug_folder=False)

phishing_datalake = PhishingDataLake.prepare_datalake(config, autocommit=False)

cur = phishing_datalake._db_cursor
cur.execute("""
                select distinct is2.filehash from
                imported_samples is2
                left join imported_samples is3 on is2.filehash =is3.filehash and is3.provider = 'FromUrl'
                inner join samplecluster s on is2.filehash =s.filehash
                where is2.provider <> 'FromUrl' and is3.provider is not null
            """)

filehashes = [f[0] for f in cur.fetchall()]

print(f"Samples to test: {len(filehashes)}")

tmp_old = "/tmp/old_screenshots_2"
if not Path(tmp_old).exists():
    os.makedirs(tmp_old)

with Pool(10) as p:
    r = list(tqdm.tqdm(
        p.imap(functools.partial(process_sample, root_folder=phishing_datalake.root_folder,screenshots_files_endpoint=phishing_datalake.screenshots_files_endpoint,thumbnail_screenshot_entrypoint=phishing_datalake.thumbnail_screenshot_entrypoint, tmp_old=tmp_old), filehashes),
        total=len(filehashes)))
