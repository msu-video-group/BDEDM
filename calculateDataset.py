from glob import glob
import os
import gc
import ioYUV
import pickle
from tqdm import tqdm
from extractFeatures import uncompressedFeatureExtractor, compressedFeatureExtractor

def calcUncompr(path, GTFolder='GT', targetBitDepth=10, multiframe=False, videoMetaInfo={'W':1920, 'H':1080}):
    folders = glob(os.path.join(path, '*'))
    X = []
    Y = []
    for i, folder in enumerate(tqdm(folders)):
        vids = glob(os.path.join(folder, '*'))
        for vid in tqdm(vids):
            for frame in ioYUV.read_frame(vid, w=videoMetaInfo.W, h=videoMetaInfo.H, bit_depth=targetBitDepth):
                features = uncompressedFeatureExtractor(frame[:,:,0])
                features.extend(uncompressedFeatureExtractor(frame[:,:,1]))
                features.extend(uncompressedFeatureExtractor(frame[:,:,2]))
                X.append(features)

                if os.path.split(folder)[-1] == GTFolder:
                    Y.append(0)
                else:
                    Y.append(1)
                    
                if not multiframe:
                    break
            gc.collect()  
    return X, Y



def calcCompr(path, modelSVMpath, modelGBMpath, preset='fast', GTFolder='GT', targetBitDepth=10):
    folders = glob(os.path.join(path, '*'))
    X = []
    Y = []
    
    modelSVM = pickle.load(open(modelSVMpath, 'rb'))
    modelGBM = pickle.load(open(modelGBMpath, 'rb'))

    for folder in tqdm(folders):
        vids = glob(os.path.join(folder, '*'))
        for vid in tqdm(vids):
                   
            X.append(compressedFeatureExtractor(vid, modelSVM, modelGBM, targetBitDepth=targetBitDepth, preset=preset))

            if os.path.split(folder)[-1] == GTFolder:
                Y.append(0)
            else:
                Y.append(1)

    return X, Y
        

    