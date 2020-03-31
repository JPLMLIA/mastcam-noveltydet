import argparse
import os
import urllib.request
from zipfile import ZipFile

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Zenoto datasets')
    parser.add_argument('dest', help='Data Destination', type=dir_path)
    parser.add_argument('-f','--force', help='Force Redownload',action='store_true')
    args = parser.parse_args()

    files = ["test_novel","test_typical","train_typical","validation_typical"]
    record = 3732485
    for file in files:
        file_path = os.path.join(args.dest,file)
        zip_path = file_path + ".zip"
        print("Checking %s" % file_path)
        if os.path.isdir(file_path) and not args.force:
            print("Found %s... skipping download" % file_path)
            continue
        print("Downloading %s.zip from Zenoto" % file)
        url = "https://zenodo.org/record/%i/files/%s.zip" % (record,file)

        urllib.request.urlretrieve(url,zip_path)
        print("Download complete... extracting to %s" % file_path)
        with ZipFile(zip_path,'r') as zipObj:
            zipObj.extractall(args.dest)
        print("Extraction complete \n")
