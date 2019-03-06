from zipfile import ZipFile
import numpy as np
import os
import requests
import time

GOODLE_DRIVE_WEIGHTS_ID = '1Iz5bXqK8P98k8XU6QGWiF3_XTaUAzw7D'

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass

    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

def weights_download():
    if not os.path.exists("weights/"):
        print("Downloading Weights......")
        response = requests.get('https://drive.google.com/uc?export=download&id='+GOODLE_DRIVE_WEIGHTS_ID)
        with open('weights.zip','wb') as f:
            for data in response.iter_content(100000):
                f.write(data)
        print("Download Completed")
        #time.sleep(5)
        print("unzipping weights")
        zip = ZipFile("weights.zip",'r')
        zip.extractall('./')
        zip.close()
        print("Weights Unzipped Successfully!")
