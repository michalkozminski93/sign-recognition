'''Functions for SVM classification, Tensorflow classification, HOG descriptor'''
import cv2, time, os, glob, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import svm, datasets, metrics
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from skimage import data, exposure, transform, util

train_test_dir = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/train_test/"
val_path = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/validation/"
model_dir = "C:/Users/Michau/Desktop/best/"
sign_desc = {'0':'20', '1':'30', '2':'40', '3':'50', '4':'60', '5':'70', '6':'80', '7':'90', '8':'100', '9':'110', '10':'120', '11':'Zakaz wyprz.', '12':'Ustap piersz.',
            '13':'Stop', '14':'Zakaz wjazdu', '15':'Zakaz ruchu', '16':'Zakaz w lewo', '17':'Zakaz w prawo', '18':'Niebezp. lewo', '19':'Niebezp. prawo', '20':'Zwezenie',
            '21':'Nakaz w lewo', '22':'Nakaz w prawo', '23':'Nakaz prosto', '24':'Z prawej', '25':'Z lewej', '26':'Rondo', '27':'Tlo'}

#########################################################################
def get_images_paths(path, random=True):
    filenames = []
    dir = os.fsencode(path)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".ppm"): 
            filenames.append(os.fsdecode(dir+file))
    if (random): shuffle(filenames)
    return filenames

def load_data(path, show_hist=False, one_hot=True, type='svm'):
    X = []
    Y = []
    data_hist = []
    index = 0
    global num_classes, img_size
    labels = np.zeros(num_classes, dtype = np.int64)

    for root, dirs, files in os.walk(os.fsencode(path)):
        if (index):
            dir = os.fsencode(root)
            if os.path.basename(root):
                for file in os.listdir(dir):
                    filename = os.fsdecode(file)
                    if filename.endswith(".ppm") or filename.endswith(".jpg") or filename.endswith(".png"):
                        filePath = os.fsdecode(root)+"/"+filename
                        img = cv2.imread(filePath,0)
                        if (type=='tf'):
                            X.append(np.asarray(cv2.resize(img,(img_size, img_size)), dtype="float32"))
                            #X.append(cv2.resize(img,(img_size, img_size)))
                        if (type=='svm'):
                            X.append(cv2.resize(img,(64,64)))
                        Y.append(int(os.fsdecode(os.path.basename(root))))
            print(dir, len(os.listdir(dir)))
            data_hist.append(len(os.listdir(dir)))
        else:
            for i in range (num_classes):
                labels[i] = os.fsdecode(dirs[i])
        index += 1
    if (show_hist):
        hist = np.zeros(num_classes, dtype = np.int64)
        for i in range(num_classes):
            hist[labels[i]] = data_hist[i] 
            labels[i]=i
        plt.bar(labels, hist, width=0.8, bottom=0.2)
        plt.title('Training data distribution')
        plt.xticks(labels, sign_desc.values(), rotation=90)
        plt.show()
    if(type=="tf" and one_hot):
        Y = np.asarray(Y)
        X = np.asarray(X)
        lb = LabelBinarizer().fit(labels)
        Y = lb.transform(Y)
    return X, Y

def flip_and_save(path, i):
    print (path)
    img = cv2.imread(path)
    img = cv2.flip(img, 1)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    cv2.imwrite('ll_'+str(i)+'.png', img)

def flip_signs():
    i=0
    filenames = []
    dir = os.fsencode('C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/train_test/19/')
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".ppm"): 
            filenames.append(os.fsdecode(dir+file))
    for path in filenames:
        flip_and_save(path, i)
        i += 1

def rotate_img(img):
    random_degree = random.uniform(-15,15)
    return transform.rotate(img, random_degree)

def add_noise(img):
    return util.random_noise(img)

def translate_img(img):
    rows, cols = img.shape
    tx = random.uniform(-5, 5)
    ty = random.uniform(-5, 5)
    M = np.float32([[1,0,int(tx)], [0,1,int(ty)]])
    img = cv2.warpAffine(img, M, (cols, rows))
    return img

filenames = get_images_paths('C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/train_test/7/')
transformation = {
    'rotation': rotate_img,
    'noise': add_noise,
    'translation': translate_img
    }
transformations_to_apply = random.randint(1, len(transformation))

for i in enumerate (100):
    rand_path = random.choice(filenames)
    img = cv2.imread(rand_path)
    img = cv2.resize(img, (32,32))
    shuffle(transformation)
    item = 0
    for j in range (transformations_to_apply):
        img = transformation[item](img)
    cv2.imwrite('C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/train_test/7/new'+str(i)+'.jpg', img)


#########################################################################

def preprocess(x_train, x_test, equalize_hist=False):
    for i in x_test:
        i /= 255
    for i in x_train:
        i /= 255
    if (equalize_hist):
        for i in range(x_test.shape[0]):
            x_test[i] = exposure.equalize_adapthist(x_test[i])
            print ("Proccessed image from test dataset: ", i)
        for i in range(x_train.shape[0]):
            x_train[i] = exposure.equalize_adapthist(x_train[i])
            print ("Proccessed image from train dataset: ", i)
    else:
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train /= std
        x_test /= std
        x_train -= mean
        x_test -= mean
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    print ('Preprocessing done.')
    return x_train, x_test

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
    ax2.set_title('Histogram zorientowanych gradientów')
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
    npNumber = np.zeros(h.shape[0], dtype=np.float)
    for i in range(0, h.shape[0]):
        npNumber[i]=h[i]
    return npNumber

def train_svm(hog, train_data, train_labels):
    X = []
    start_time = time.time()
    for i in train_data:
        X.append(calculate_hog(hog, i))
        print ("Iteration :", i)

    print ("Number of training samples: ", len(X))

    #parameters={'C':[0.1, 0.5, 1, 2, 4, 6, 8, 10, 100, 200], 'tol':[1e-4, 1e-3, 1e-2, 0.1, 1, 10], 'gamma':[1e-4, 1e-3, 1e-2, 0.1, 1, 10]}
    parameters={'C':[0.1, 1, 10], 'tol':[1e-4, 1e-3, 1e-2], 'gamma':[1e-4, 1e-3, 1e-2]}
    clf = svm.SVC(C=1.95, tol=1.55, cache_size=100, kernel='rbf', probability=False)
    #clf_search = svm.SVC(cache_size=100, kernel='rbf', probability=False)
    #clf = GridSearchCV(clf_search, parameters)

    data_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    data_scalerTransf = data_scaler.fit_transform(X)
    clf.fit(data_scalerTransf,train_labels)

    #best = clf.best_params_
    #results = clf.cv_results_

    print ("Training time: ", time.time()-start_time)

    return clf, data_scaler

def train_many_svm(hog, data, labels):
    data = []
    for i in dataImg:
        data.append(calculate_hog(hog, i))

    train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
    scores = []

    cList = (0.1, 0.5, 1, 2, 4, 6, 8, 10, 100, 200)
    X = np.arange(0.05, 2, 0.1)
    tolList = (1e-4, 1e-3, 1e-2, 0.1, 1, 10)
    Y = np.arange(0.05, 2, 0.1)
    max_score = 0
    max_params = ""
    print ("Number of training samples: ", len(train_data), "\nNumber of samples for evaluation: ", len(eval_data))
    for param1 in X.tolist():
        for param2 in Y.tolist(): 
            start = time.time()
            clf = svm.SVC(C=param1, tol=1.55, cache_size=100, kernel='rbf')
            data_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            data_scalerTransf = data_scaler.fit_transform(train_data)
            clf.fit(data_scalerTransf,train_labels)
            data_scalerTransf = data_scaler.fit_transform(train_data)

            numOk = 0
            eval_data_transform = []
            for j in range (len(eval_data)):
                trans_data = data_scaler.transform(eval_data[j].reshape(1,-1))
                eval_data_transform.append(trans_data)
                prediction = clf.predict(trans_data)
                if (prediction == eval_labels[j]): numOk += 1
            score2 = clf.score(eval_data_transform, eval_labels)
            finish = time.time()
            score = np.around(numOk/len(eval_data)*100,2)
            if score > max_score:
                max_params = 'Score: '+ str(score), " %, param1 = ", str(param1), "param2 = ", str(param2)
                max_score = score
            scores.append(score)
            print ('Score: ', score, " %, param1 = ", param1, "param2 = ", param2)
            print ('Time: ', finish-start)
    print (max_params)
    X, Y = np.meshgrid(X,Y)
    Z = np.asarray(scores).reshape(X.shape[0], Y.shape[1])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('C')
    ax.set_ylabel('tol')
    ax.set_zlabel('result')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

    plt.show()

    return clf, data_scaler

def save_clf_svm(classifier, scaler):
    #Saving classifier configuration and data scaler
    joblib.dump(classifier,'svm_clf.pkl')
    joblib.dump(scaler,'svm_scaler.pkl')

def load_clf_svm():
    #Loading classifier configuration and data scaler
    classifier = joblib.load('svm_clf.pkl')
    data_scaler = joblib.load('svm_scaler.pkl')

    return classifier, data_scaler

def predict_svm(image, classifier, data_scaler, hog):
    img_hog = calculate_hog(hog, image)
    trans_data = data_scaler.transform(img_hog.reshape(1,-1))
    prediction = classifier.predict(trans_data)
    return prediction

#########################################################################

#Layers parameters
filter_size1 = 5
num_filters1 = 64
drop_conv1 = 0.9

filter_size2 = 5
num_filters2 = 128
drop_conv2 = None

filter_size3 = None
num_filters3 = 128
drop_conv3 = None

drop_fc1 = 0.5
drop_fc2 = None
fc_size = 1024

train_batch_size = 50
max_epochs = 100
l2_reg_enabled = True
l2_lambda = 0.0001
learning_rate = 0.001
hist_equalization = False
num_iterations = 7000

img_size = 32
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 28

def make_hparam_string():
    global filter_size1, filter_size2, filter_size3, num_filters1, num_filters2, num_filters3
    global drop_fc1, drop_fc2, fc_size, train_batch_size, l2_lambda, l2_reg_enabled, learning_rate
    global drop_conv1, drop_conv2, drop_conv3, hist_equalization, num_iterations
    str = 'iter_%d,fs1_%d,nf1_%d,fs2_%d,nf2_%d,' % (num_iterations, filter_size1, num_filters1, filter_size2, num_filters2)
    if (filter_size3 is not None): str += 'fs3_%d,nf3_%d,' % (filter_size3, num_filters3)
    str += "lr_%.0E,bs_%d," % (learning_rate, train_batch_size)
    if (hist_equalization): str += "histEq_%r" % (hist_equalization)
    if (l2_reg_enabled): str += "lambda_%.0E," % (l2_lambda)
    if (drop_conv1 is not None): str += 'dc1_%.2f,' % (drop_conv1)
    if (drop_conv2 is not None): str += 'dc2_%.2f,' % (drop_conv2)
    if (drop_conv3 is not None): str += 'dc3_%.2f,' % (drop_conv3)
    if (drop_fc1 is not None): str += 'dfc1_%.2f,' % (drop_fc1)
    if (drop_fc2 is not None): str += 'dfc2_%.2f' % (drop_fc2)

    return str

#Random weights
def new_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=1)*np.sqrt(2/6500), name=name+"_W")

#Biases = 0.05
def new_biases(length, name):
    return tf.Variable(tf.constant(0.005, shape=[length]), name=name+"_B")

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True, name="conv"):
    with tf.variable_scope(name):
        #Filter shape for convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        #Creating weights of filter size
        weights = new_weights(shape, name)

        #New biases, one for every filter
        biases = new_biases(num_filters, name=name)

        #padding - zeros added on the edges
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 2, 2, 1], padding='SAME')

        #Adding biases
        layer += biases

        if use_pooling:
            layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        #tutaj zastosowac batch normalizacje
        layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", layer)
        return layer, weights

def flatten_layer(layer, name="flatten"):
    with tf.variable_scope(name):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, name="fc"):
    with tf.variable_scope(name):
        weights = new_weights(shape=[num_inputs, num_outputs], name=name)
        biases = new_biases(length=num_outputs, name=name)
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        return layer, weights

def dropout_layer(input, rate, name="dropout"):
    with tf.variable_scope(name):
        layer_drop = tf.nn.dropout(input, rate)
        return layer_drop

def return_model_tf(x, is_training):
    #Convolution layer
    layer_conv1, weights_conv1 = new_conv_layer(x, num_channels, filter_size1, num_filters1, use_pooling = True, name = 'conv1')

    #Dropout on convolutional layer
    if (drop_conv1 is not None):
        layer_conv1 = tf.cond(is_training, lambda: dropout_layer(layer_conv1, drop_conv1, name = 'drop_conv1'), lambda: layer_conv1)

    #Convolution layer
    layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, use_pooling = True, name = 'conv2')

    #Dropout on convolutional layer
    if (drop_conv2 is not None):
        layer_conv2 = tf.cond(is_training, lambda: dropout_layer(layer_conv2, drop_conv2, name = 'drop_conv2'), lambda: layer_conv2)

    #Convolution layer
    if (filter_size3 is not None):
        layer_conv3, weights_conv3 = new_conv_layer(layer_conv2, num_filters2, filter_size3, num_filters3, use_pooling=True, name = 'conv3')

        #Dropout on convolution layer
        if (drop_conv3 is not None):
            layer_conv3 = tf.cond(is_training, lambda: dropout_layer(layer_conv3, drop_conv3, name = 'drop_conv3'), lambda: layer_conv3)

        #Flattened layer
        layer_flat, num_features = flatten_layer(layer_conv3, 'flat1')
    else:
        #Flattened layer
        layer_flat, num_features = flatten_layer(layer_conv2, 'flat1')

    #Fully connected layer
    layer_fc1, weights_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True, name='fc1')

    #Dropout on fully connected layer
    if (drop_fc1 is not None): 
        layer_fc1 = tf.cond(is_training, lambda: dropout_layer(layer_fc1, drop_fc1, name = 'drop_fc1'), lambda: layer_fc1)

    #Fully connected layer
    layer_fc2, weights_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, use_relu=False, name='fc2')

    #Dropout on fully connected layer
    if (drop_fc2 is not None):
        layer_fc2 = tf.cond(is_training, lambda: dropout_layer(layer_fc2, drop_fc2, name = 'drop_fc2'), lambda: layer_fc2)

    return layer_fc2, weights_fc2

def train_model(train_dataset, test_data, test_labels):
    global train_batch_size, num_iterations

    graph = tf.Graph()
    #Graph preparation
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        is_training = tf.placeholder(tf.bool, shape = [], name='is_training')
    
        with tf.variable_scope('cnn'):
            layer_fc2, weights_fc2 = return_model_tf(x, is_training)
            l2_loss = tf.nn.l2_loss(weights_fc2)

        #Output layer with softmax
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, axis = 1)  
        y_true_cls = tf.argmax(y_true, axis = 1)

    with tf.Session(graph = graph) as sess:
        with tf.name_scope('cross_entropy'):
           with tf.name_scope('total'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_true_cls, logits=y_pred)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            loss = tf.reduce_mean(cross_entropy) + l2_lambda * l2_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_predictions = tf.equal(y_pred_cls, y_true_cls)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        train_dataset = train_dataset.batch(train_batch_size)
        next_batch = train_dataset.make_one_shot_iterator().get_next()

        #Managing summaries
        merged_summary = tf.summary.merge_all()
        desc = make_hparam_string()
        train_writer = tf.summary.FileWriter(model_dir + 'train ' + desc, sess.graph)
        test_writer = tf.summary.FileWriter(model_dir + 'test '+ desc)
        train_writer.add_graph(sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        for i in range(num_iterations):
            batch = sess.run(next_batch)
            feed_dict_train = {x: batch[0], y_true: batch[1], is_training: True}
            feed_dict_test = {x: test_data, y_true: test_labels, is_training: False}

            if (i % 50 == 0):
                summary, acc = sess.run([merged_summary, accuracy], feed_dict=feed_dict_test)
                test_writer.add_summary(summary, i)
                print("Optimization Interation: ", i+1, " Testing Accuracy: ", np.around(acc,3))

                '''list = [0, 1, 3, 5, 6, 7, 8, 10, 11, 15]
                for i in list:
                    new_data = cv2.imread(str(i)+'.jpg')
                    print(str(i)+' -->', predict_tf(sess, x, y_pred, is_training, new_data))
                '''
            summary, _ = sess.run([merged_summary, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

            if (i % 100 == 0): 
                saver.save(sess, model_dir + 'mymodel ' + desc, i)
            print ('Iteration: ', i)
        
        saver.save(sess, model_dir + 'mymodel ' + desc)
        print ('Learning time:', np.around((time.time()-start_time), 2), ' s')

def load_model():
    start = time.time()
    saver = tf.train.import_meta_graph(model_dir + 'mymodel '+ make_hparam_string() + '.meta')

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    input = sess.graph.get_operation_by_name('x').outputs[0]
    prediction = sess.graph.get_operation_by_name('y_pred').outputs[0]
    is_training = sess.graph.get_operation_by_name('is_training').outputs[0]
    print ("Tensorflow model loading time: ", time.time()-start)
    return sess, graph, input, prediction, is_training

def evaluate_model(Xt, Yt, confusion_matrix=False, type='tf', clf=None, scaler=None, hog=None):
    predictions = []
    n_samples = len(Xt)
    pass_number = 0
    
    start = time.time()

    #Model evaluation for Convolutional Neural Network
    if (type == 'tf'):
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels])
            is_training = tf.constant(False)
            with tf.variable_scope('cnn'):
                layer_out, weights_out = return_model_tf(x, is_training)
                y_pred = tf.nn.softmax(layer_out)

        with tf.Session(graph = graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
            print ("Tensorflow model loading time: ", time.time()-start)
            for j in range(n_samples):
                prediction, value = predict_tf(sess, x, y_pred, is_training, Xt[j])
                if(confusion_matrix): predictions.append(prediction)
                if (prediction == Yt[j]): pass_number += 1
    
    #Model evaluation for Support Vector Machine
    if (type == 'svm'):
        for j in range(n_samples):
            img_hog = calculate_hog(hog, eval_data[j])
            transf_data = scaler.transform(img_hog.reshape(1,-1))
            prediction = clf.predict(transf_data)[0]
            if(confusion_matrix): predictions.append(prediction)
            if (prediction == Yt[j]): pass_number += 1

    print ("Performance: ",np.around(pass_number/n_samples*100,1),"%") 
    print ("Number of test samples: ", n_samples)
    if (confusion_matrix):
        classes = []
        for i in sign_desc.keys():
            classes.append(int(i))
        conf_mx = metrics.confusion_matrix(y_true=Yt, y_pred=predictions, labels = np.asarray(classes))

        row_sums = conf_mx.sum(axis=1)
        plt.matshow(conf_mx, cmap=plt.cm.Blues)
        conf_mx = conf_mx.astype('float')
        for i in range(conf_mx.shape[0]):
            if row_sums[i] > 1: conf_mx[i] = conf_mx[i] / row_sums[i]
        plt.matshow(conf_mx, cmap=plt.cm.Blues)
        plt.show()

def predict_tf(sess, input, output, is_training, new_data):
    global img_size, num_channels
    new_data = cv2.resize(new_data, (img_size, img_size))
    if (len(new_data.shape) > 2 and new_data.shape[2] == 3): new_data = cv2.cvtColor(new_data, cv2.COLOR_BGR2GRAY)
    new_data = new_data / 255
    new_data = np.expand_dims(new_data, axis=2)

    predictions = sess.run(output, feed_dict={input:[new_data], is_training:False})
    pred_class = np.argmax(predictions)
    pred_value = predictions[0][pred_class]
    return pred_class, pred_value

##############################################################################################################
'''#Uczenie modelu tf
data, labels = load_data(train_test_dir, show_hist = False, one_hot=True, type = 'tf')
print ('All data: ', len(data))
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, shuffle=True)
print ('Training data: ', len(train_data))

train_data, test_data = preprocess(train_data, test_data, hist_equalization)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).repeat()
train_database_size = train_data.shape[0]

train_model(train_dataset, test_data, test_labels)

val_data, val_labels = load_data(val_path, False, False, 'tf')
print ('Validation data: ', len(val_data))
evaluate_model(val_data, val_labels, True, 'tf')
'''

'''#Uczenie modelu svm
#data, labels = load_data(train_test_dir, show_hist=False, one_hot=False, type='svm')
#data, test_data, labels, test_labels = train_test_split(data, labels, test_size=0, shuffle=True)
hog = initialize_hog()
#clf, scaler = train_svm(hog, data, labels)
clf, scaler = load_clf_svm()

eval_data, eval_labels = load_data(val_path, False, False, 'svm')
evaluate_model(eval_data, eval_labels, True, 'svm', clf, scaler, hog)
'''