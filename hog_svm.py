###HOG descriptor + SVM###

import numpy as np
import cv2, time, sklearn, os, glob
from sklearn import svm, datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def initializeHOG():
    ###Parameters of HOG descriptor###
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog

def calculateHOG(hog, img):
    imgSize = img.shape[0]*img.shape[1]                 
    h = hog.compute(img)
    npNumber = np.zeros(h.shape[0], dtype=np.float)
    for i in range(0, h.shape[0]):
        npNumber[i]=h[i]
    return npNumber

def trainClassifier(hog, directoryTraining):
    Y = []
    X = []
    nSamples = 0

    for root, dirs, files in os.walk(os.fsencode(directoryTraining)):
        directory = os.fsencode(root)
        if os.path.basename(root):
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".ppm") or filename.endswith(".jpg"):
                    filePath = os.fsdecode(root)+"/"+filename
                    img = cv2.imread(filePath,0)
                    X.append(calculateHOG(hog, img))
                    Y.append(os.fsdecode(os.path.basename(root)))
                    nSamples+=1

    print ("Number of training samples: ", nSamples)
    clf = svm.SVC(gamma=1/10000, C=100.0, tol=1e-8, cache_size=600, kernel='linear')
    xScaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    xScalerTransf = xScaler.fit_transform(X)
    clf.fit(xScalerTransf,Y)

    return clf, xScaler

def saveClassifier(classifier, scaler):
    ###Saving the classifier configuration###
    joblib.dump(classifier,'classifier.pkl')
    joblib.dump(scaler,'scaler.pkl')

def loadClassifier():
    classifier = joblib.load('classifier.pkl')
    xScaler = joblib.load('scaler.pkl')

    return classifier, xScaler

def predictSign(image, classifier, xScaler, hog):
    hTest = calculateHOG(hog, image)
    transTest = xScaler.transform(hTest.reshape(1,-1))
    prediction = classifier.predict(transTest)
    return prediction

def checkPerformance(directoryTesting, classifier, xScaler, hog):
    nSamples = 0
    okPredictions = 0
    for root, dirs, files in os.walk(os.fsencode(directoryTesting)):
        directory = os.fsencode(root)
        if os.path.basename(root):
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".ppm") or filename.endswith(".jpg"):
                    filePath = os.fsdecode(root)+"/"+filename
                    img = cv2.imread(filePath,0)
                    prediction = predictSign(img, classifier, xScaler, hog)
                    sign = os.fsdecode(os.path.basename(root))

                    if prediction[0] == sign: okPredictions+=1
                    nSamples+=1

                    print("Sign: ", sign, " Prediction: ", prediction[0])
                    #cv2.imshow('test',img)
                    #cv2.waitKey(0)
                    cv2.destroyAllWindows()

    print ("Performance:",np.around(okPredictions/nSamples*100,1),"%")                

directoryTesting = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/SignRecognition/testing"
directoryTraining = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/SignRecognition/training/"

#hog = initializeHOG()
#clf, xScaler = trainClassifier(hog, directoryTraining)
#saveClassifier(clf, xScaler)
#clf, xScaler = loadClassifier()
#checkPerformance(directoryTesting, clf, xScaler, hog)

#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))