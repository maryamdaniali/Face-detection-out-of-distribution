import os
import sys
import csv
from shutil import copy
from pathlib import Path
import numpy as np
from tqdm import tqdm
import requests
import zipfile

def unzip_file(filename, path = Path()):
    ## unzip file: filename in a directly with the same name
    fname, suffix = filename.split('.')
    dist_path = path/fname
    mkdir_p(dist_path)
    with zipfile.ZipFile(path/filename,"r") as zip_ref:
        zip_ref.extractall(dist_path)

def download_file_from_google_drive(id, destination):
    ## download files from google drive using file id and destination filename
    ## this function skips the scanning if the file is bigger than google scanner can work with, called download anyway on UI
    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_file(file_id, destination, path = Path()):
    ## takes file id, destination name/address and proccess through google drive downloader
    download_file_from_google_drive(file_id, destination)
    if destination.endswith(".zip"):
        unzip_file(destination, path)

def load_csv(filepath):
    ## reads CVS files to extract ethnicity and gender categories of images in FairFace data set
    with open(filepath) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            gender_dict = {}
            race_dict = {}
            
            # read file data
            for row in reader:
                if row:
                    gender_dict[row[0]] = row[2]
                    race_dict[row[0]] = row[3]

    return gender_dict,race_dict

def mkdir_p(mypath):
    ## Creates a directory. equivalent to using mkdir -p on the command line
    ## Ref: https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
def copy_to_folders(data_type, feature, data_dict, path):
        ## data_type: "train" or "val"
        ## feature: "gender" or "race"
        ## data_dict: dictionary output of load_csv
        ## path: directory of the train or val set
   
    for file_name in data_dict:
        source_file = Path(path/file_name)
        dist_path = path/feature/data_type/f"{data_dict[file_name]}/"
        mkdir_p(dist_path)
        copy(source_file,dist_path)

def copy_to_folders_mixed_features(data_type, data_dict_1, data_dict_2, path):
    ''' data_type: "train" or "val"
        data_dict: dictionary output of load_csv
        path: directory of the train or val set
    '''    
    for file_name in data_dict_1:
        source_file = Path(path/file_name)
        dist_path = path/'mixed_features'/data_type/f"{data_dict_1[file_name]}"/f"{data_dict_2[file_name]}"
        mkdir_p(dist_path)
        copy(source_file,dist_path)

if __name__ == "__main__":

    ## Desired path to get the data in, etc.
    path= Path()
    ## Download dataset files from google drive, including FairFace images and their labels
    ## for more info visit https://github.com/joojs/fairface
    dict_file_info = {} #dict contaning filename, file id for each item
    dict_file_info['fairface-img-margin025-trainval.zip'] = ['1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86']
    dict_file_info['fairface_label_train.csv'] = [ '1wOdja-ezstMEp81tX1a-EYkFebev4h7D']
    dict_file_info['fairface_label_val.csv'] = [ '1wOdja-ezstMEp81tX1a-EYkFebev4h7D']
    ## download each file
    for file in dict_file_info:
        print('Downloading file: ', file)
        get_file(dict_file_info[file], file, path)

    ## by defaults fairface data will be soted at path/'fairface-img-margin025-trainval'
    data_folder = path/'fairface-img-margin025-trainval'
    
    ## read and process train set
    train_folder = data_folder/'train'
    ## read train CSV file
    train_label_file = path/'fairface_label_train.csv'
    train_gender_dict,train_race_dict = load_csv(train_label_file)
    ## copy files to distination folders
    copy_to_folders_mixed_features('train', train_race_dict,train_gender_dict, data_folder) #dispatch files based on both ethnicity and gender
    # copy_to_folders('train', 'gender', train_gender_dict, data_folder) # dispatch files based on gender
    # copy_to_folders('train', 'race', train_race_dict, data_folder) # dispatch files based on race/ethnicity
    
    ## read and process validation set
    val_folder = data_folder/'val'
    ## read val CSV file
    val_label_file = path/'fairface_label_val.csv'
    val_gender_dict,val_race_dict = load_csv(val_label_file)
    ##copy files to distination folders
    copy_to_folders_mixed_features('val', val_race_dict,val_gender_dict, data_folder)
    # copy_to_folders('val', 'gender', val_gender_dict, data_folder)
    # copy_to_folders('val', 'race', val_race_dict, data_folder)

    print("Done")