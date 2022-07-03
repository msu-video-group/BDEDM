import numpy as np
import subprocess
import json
import os
import cv2
from scipy import stats
import skvideo.io as vio
from tqdm import tqdm

def getFrameTypeStructure(path, ffmpegPath=''):
    fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    with open(fname, "w") as f:
        subprocess.call([ffmpegPath + 'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_frames', path], stdout=f)
    with open(fname) as f:
        frames = json.load(f)['frames']
    types = []
    sz = []
    for frame in frames:
        types.append(frame['pict_type'])
        sz.append(int(frame['pkt_size']))
    os.remove(fname)
    return types, sz

def uncompressedFeatureExtractor(frame, winSize=16, bins=100):
    features = []
    lsb = frame % 4
    msb = frame // 4

    iimg, iimg2 = cv2.integral2(msb)
    mean = (iimg[winSize:, winSize:] + iimg[:-winSize, :-winSize] - iimg[:-winSize, winSize:] - iimg[winSize:, :-winSize]) / winSize / winSize
    mean2 = (iimg2[winSize:, winSize:] + iimg2[:-winSize, :-winSize] - iimg2[:-winSize, winSize:] - iimg2[winSize:, :-winSize]) / winSize / winSize
    flat = (mean2  - mean**2).flatten()

    iimg, iimg2 = cv2.integral2(lsb)
    mean = (iimg[winSize:, winSize:] + iimg[:-winSize, :-winSize] - iimg[:-winSize, winSize:] - iimg[winSize:, :-winSize]) / winSize / winSize
    mean2 = (iimg2[winSize:, winSize:] + iimg2[:-winSize, :-winSize] - iimg2[:-winSize, winSize:] - iimg2[winSize:, :-winSize]) / winSize / winSize
    std = np.log1p(mean2  - mean**2).flatten()
    std = std[np.where(flat != 0)]
    std = std - np.mean(std)
    std = std / np.max(np.abs(std))

    iimg = cv2.integral(msb)
    mean = ((iimg[winSize:, winSize:] + iimg[:-winSize, :-winSize] - iimg[:-winSize, winSize:] - iimg[winSize:, :-winSize]) / winSize / winSize).flatten()
    mean = mean[np.where(flat != 0)]
    mean = mean - np.mean(mean)
    mean = mean / np.max(np.abs(mean))

    bin_means, bin_edges, binnumber = stats.binned_statistic(mean, std, 'mean', bins=bins)
    bin_means = np.nan_to_num(bin_means) 
    features.append(np.mean(bin_means))                                                       
    features.append(np.std(bin_means))                                                         
    outliers = bin_means[np.where(np.abs(bin_means - np.mean(bin_means)) > 3 * np.std(bin_means))]  
    features.append(outliers.shape[0])
    
    bin_means, bin_edges, binnumber = stats.binned_statistic(mean, std, 'std', bins=bins)
    bin_means = np.nan_to_num(bin_means)
    features.append(np.mean(bin_means))
    features.append(np.std(bin_means))
    outliers = bin_means[np.where(np.abs(bin_means - np.mean(bin_means)) > 3 * np.std(bin_means))]
    features.append(outliers.shape[0])
    
    bin_max, bin_edges, binnumber = stats.binned_statistic(mean, std, 'max', bins=bins)
    bin_min, bin_edges, binnumber = stats.binned_statistic(mean, std, 'min', bins=bins)
    
    dist = bin_max - bin_min + np.finfo(float).eps
    features.append(np.sum(dist * np.log2(dist)))
 
    return features
    
def modelFeature(frame, clf1, clf2):
    features = uncompressedFeatureExtractor(frame[:, :, 0])
    features.extend(uncompressedFeatureExtractor(frame[:, :, 1]))
    features.extend(uncompressedFeatureExtractor(frame[:, :, 2]))
    return (clf1.predict_proba(np.asarray([features]))[0] + clf2.predict_proba(np.asarray([features]))[0]) / 2
    
def compressedFeatureExtractor(videoPath, modelSVM, modelGBM, targetBitDepth=10, preset='fast'):
    compFeatures = []
    reader = vio.FFmpegReader(videoPath,
                    inputdict={},
                    outputdict={'-pix_fmt': 'yuv444p16le'})

    frameType, frameSz = getFrameTypeStructure(videoPath)

    feature = []
    usedFrameTypes = {}
    for i, frame in tqdm(enumerate(reader)):
        frame = (frame // 2**(16 - targetBitDepth))
        frameFeature= [np.std(frame[:, :, 0] % 4), np.mean(frame[:, :, 0] % 4), 
                        np.std(frame[:, :, 1] % 4), np.mean(frame[:, :, 1] % 4), 
                        np.std(frame[:, :, 2] % 4), np.mean(frame[:, :, 2] % 4)] 
        proba = modelFeature(frame, modelSVM, modelGBM)[0]
        if preset == 'fast' and not (ftype[i] in usedFrameTypes.keys()):
            frameFeature.append(proba)
            usedFrameTypes[ftype[i]] = proba
        elif preset == 'fast':
            frameFeature.append(usedFrameTypes[ftype[i]])
        else:
            frameFeature.append(proba)
        feature.append(frameFeature)

    
    feature = np.asarray(feature)
    frameSz = np.asarray(frameSz)
    frameType = np.asarray(frameType)
    
    compFeatures.extend([np.mean(frameSz),
                         np.mean(frameSz[np.where(frameType == 'I')]),
                         np.mean(frameSz[np.where(frameType == 'B')]),
                         np.mean(frameSz[np.where(frameType == 'P')]),
                         np.std(frameSz[np.where(frameType == 'B')]),
                         np.std(frameSz[np.where(frameType == 'P')])])
    
    I = feature[np.where(frameType == 'I'), :]
    I = I[0, :, :]
    P = feature[np.where(frameType == 'P'), :]
    P = P[0, :, :]
    B = feature[np.where(frameType == 'B'), :]
    B = B[0, :, :]
    
    for i in range(7):
        stat = stats.ttest_ind(P[:, i], B[:, i])
        simpleFeatures = [np.mean(I, axis=0).flatten()[i],
                          np.mean(B, axis=0).flatten()[i],
                          np.mean(P, axis=0).flatten()[i]]
        complexFeatures = [np.std(B, axis=0).flatten()[0],
                           np.std(P, axis=0).flatten()[0],
                           np.log1p(abs(stat[1])),
                           abs(stat[0]), 
                           stats.shapiro(B[:, i])[0],
                           stats.shapiro(B[:, i])[1],
                           stats.shapiro(P[:, i])[0],
                           stats.shapiro(P[:, i])[1],
                           stats.shapiro(feature[:, 1])[0],
                           stats.shapiro(feature[:, 1])[1]]
        compFeatures.extend(simpleFeatures)
        if i == 6 and preset == 'precise':
            compFeatures.extend(complexFeatures)
            
        
    stat = stats.ttest_ind(feature[:, 0], feature[:, 2])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0])) 
    
    stat = stats.ttest_ind(feature[:, 0], feature[:, 4])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0]))  

    stat = stats.ttest_ind(feature[:, 4], feature[:, 2])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0])) 
    
    stat = stats.ttest_ind(feature[:, 1], feature[:, 3])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0])) 

    stat = stats.ttest_ind(feature[:, 1], feature[:, 5])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0])) 

    stat = stats.ttest_ind(feature[:, 3], feature[:, 5])
    compFeatures.append(np.log1p(abs(stat[1])))
    compFeatures.append(abs(stat[0]))   
    
    return compFeatures
    
