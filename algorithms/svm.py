import numpy as np
import cv2
import time
from skimage.feature import hog
from matplotlib import pyplot as plt
from skimage import exposure
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from algorithms.preprocessing import load_data
from parameters import train_test_dir, signs_dict, val_path


def hog_plot(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True, block_norm = 'L1')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Obraz ')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG')
    plt.show()

def initialize_hog():
    ###Parameters of HOG descriptor
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

def calculate_hog(hog, img):
    h = hog.compute(img)
    npNumber = np.zeros(h.shape[0], dtype=float)
    for i in range(0, h.shape[0]):
        npNumber[i]=h[i]
    return npNumber

def train(hog, train_data, train_labels):
    X = []
    start_time = time.time()
    for i in train_data:
        X.append(calculate_hog(hog, i))
        print ("Iteration :", i)

    print ("Number of training samples: ", len(X))

    parameters={'C':[0.1, 1, 10], 'tol':[1e-4, 1e-3, 1e-2], 'gamma':[1e-4, 1e-3, 1e-2]}
    clf = svm.SVC(C=1.95, tol=1.55, cache_size=100, kernel='rbf', probability=False)

    data_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    data_scalerTransf = data_scaler.fit_transform(X)
    clf.fit(data_scalerTransf,train_labels)

    #best = clf.best_params_
    #results = clf.cv_results_

    print ("Training time: ", time.time()-start_time)

    return clf, data_scaler

def save_clf(classifier, scaler):
    #Saving classifier configuration and data scaler
    dump(classifier,'svm_clf.pkl')
    dump(scaler,'svm_scaler.pkl')

def load_clf():
    #Loading classifier configuration and data scaler
    classifier = load('models\svm\svm_clf.pkl')
    data_scaler = load('models\svm\svm_scaler.pkl')

    return classifier, data_scaler

def predict(image, classifier, data_scaler, hog):
    img_hog = calculate_hog(hog, image)
    trans_data = data_scaler.transform(img_hog.reshape(1,-1))
    prediction = classifier.predict(trans_data)
    return prediction

def run_train_svm():
    data, labels = load_data(train_test_dir, show_hist=False, one_hot=False, type='svm')
    data, test_data, labels, test_labels = train_test_split(data, labels, test_size=0.25, shuffle=True)
    hog = initialize_hog()
    clf, scaler = train(hog, data, labels)
    save_clf(clf, scaler)

    clf, scaler = load_clf()

    eval_data, eval_labels = load_data(val_path, False, False, 'svm')
    evaluate_model(eval_data, eval_labels, True, clf, scaler, hog)

def evaluate_model(Xt, Yt, confusion_matrix=False, clf=None, scaler=None, hog=None):
    predictions = []
    n_samples = len(Xt)
    pass_number = 0
    
    start = time.time()

    for j in range(n_samples):
        img_hog = calculate_hog(hog, Xt[j])
        transf_data = scaler.transform(img_hog.reshape(1,-1))
        prediction = clf.predict(transf_data)[0]
        if(confusion_matrix): predictions.append(prediction)
        if (prediction == Yt[j]): pass_number += 1

    print ("Performance: ",np.around(pass_number/n_samples*100,1),"%") 
    print ("Number of test samples: ", n_samples)
    if (confusion_matrix):
        classes = []
        for i in signs_dict.keys():
            classes.append(int(i))
        conf_mx = metrics.confusion_matrix(y_true=Yt, y_pred=predictions, labels = np.asarray(classes))

        row_sums = conf_mx.sum(axis=1)
        plt.matshow(conf_mx, cmap=plt.cm.Blues)
        conf_mx = conf_mx.astype('float')
        for i in range(conf_mx.shape[0]):
            if row_sums[i] > 1: conf_mx[i] = conf_mx[i] / row_sums[i]
        plt.matshow(conf_mx, cmap=plt.cm.Blues)
        plt.show()