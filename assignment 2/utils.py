import numpy as np
import pandas as pd
import os
import soundfile as sf
import librosa

def audioFilesWordsIds(dataFolder: str, subFolderExceptions: np.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Function to import audio files from a directory, excluding specified subfolders
    # and generating a DataFrame with file names, subfolder names, and unique subfolder IDs.

    files = []
    subFolderNames = []
    subFolderIds = []

    subFolderIdMap = {}
    currentId = 1

    for subfolder in os.scandir(dataFolder):
        if subfolder.is_dir() and subfolder.name not in subFolderExceptions:
            subFolderIdMap[subfolder.name] = currentId
            currentId += 1
            
            for file in os.listdir(subfolder.path):
                fullFilePath = os.path.join(subfolder.path, file)
                files.append(fullFilePath)
                subFolderNames.append(subfolder.name)
                subFolderIds.append(subFolderIdMap[subfolder.name])

    files = np.array(files)
    subFolderNames = np.array(subFolderNames)
    subFolderIds = np.array(subFolderIds)

    return files,subFolderNames,subFolderIds

def loadAudioFile(file, sampleRate) -> tuple[np.array, int]:
    y, sr = librosa.load(file, sr=sampleRate)
    return np.array(y), sr


def balanceClassSamples(trainingDf: pd.DataFrame) -> pd.DataFrame:
    countSamples = trainingDf['words'].value_counts().reset_index()
    countSamples.columns = ['words', 'counts']
    meanCount = int(countSamples['counts'].mean())

    # removing words that were more than mean
    wordsAboveMean = countSamples[countSamples['counts'] > meanCount]

    for index, row in wordsAboveMean.iterrows():
        word = row['words']
        excessCount = row['counts'] - meanCount

        samplesToRemove = trainingDf[trainingDf['words'] == word].sample(n=excessCount)
        trainingDf = trainingDf.drop(samplesToRemove.index)


    # adding words that were less than mean
    countSamples = trainingDf['words'].value_counts().reset_index()
    countSamples.columns = ['words', 'counts']

    uniqueWords = trainingDf['words'].unique()

    bootstrapSamples = []
    for word in uniqueWords:
        currentCount = countSamples[countSamples['words'] == word]['counts'].sum()
        
        if currentCount < meanCount:
            samplesNeeded = meanCount - currentCount

            existingSamples = trainingDf[trainingDf['words'] == word]
            bootstrappedSamples = existingSamples.sample(n=samplesNeeded, replace=True)
            bootstrapSamples.append(bootstrappedSamples)

    if bootstrapSamples:
        bootstrappedDf = pd.concat(bootstrapSamples, ignore_index=True)
        trainingDf = pd.concat([trainingDf, bootstrappedDf], ignore_index=True)

    trainingDf = trainingDf.sample(frac=1).reset_index(drop=True)

    return trainingDf

def frameAudio(amplitudes: np.array, frameLength: int, hopLength: int) -> list[np.array]:
    frames = []
    for start in range(0, len(amplitudes) - frameLength + 1, hopLength):
        frame = amplitudes[start:start + frameLength]
        frames.append(frame)
    return frames



def optimizedGaborTransform(framedAudio: np.array, sampleRate: int, sigma: float, frequency: float) -> np.array:
    # Get time vector
    t = np.arange(framedAudio.shape[1]) / sampleRate
    gaborFilter = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * frequency * t)

    return np.apply_along_axis(lambda frame: np.convolve(frame, gaborFilter, mode='same'), axis=1, arr=framedAudio)
