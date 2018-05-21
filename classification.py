'''Functions for SVM classification, Tensorflow classification, HOG descriptor'''
import cv2, time, os, glob
import tensorflow as tf
from tensorflow import saved_model as tfsm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import svm, datasets, metrics
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from skimage import data, exposure

sign_desc = {'0':'20', '1':'30', '2':'40', '3':'50', '4':'60', '5':'70', '6':'80', '7':'100', '8':'120', '9':'none', '10':'B-25'}
data_dir = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/training/"
log_dir = "C:/Users/Michau/Desktop/model_XX/"
val_path = 'C:/Users/Michau/Desktop/new_signs/'
#log_dir = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/model_96/"
#########################################################################

def load_data(path, type='svm'):
    X = []
    Y = []
    global num_classes
    for root, dirs, files in os.walk(os.fsencode(path)):
        directory = os.fsencode(root)
        print (directory)
        if os.path.basename(root):
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".ppm") or filename.endswith(".jpg"):
                    filePath = os.fsdecode(root)+"/"+filename
                    if (type=='tf'):
                        img = cv2.imread(filePath)
                        X.append(np.asarray(cv2.resize(img,(64,64)), dtype="float32"))
                    if (type=='svm'):
                        img = cv2.imread(filePath, 0)
                        X.append(cv2.resize(img,(64,64)))
                    Y.append(int(os.fsdecode(os.path.basename(root))))
    if(type=="tf"):
        Y = np.asarray(Y)
        #print (np.average(X[0]))
        #data_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        #X = data_scaler.fit_transform(X)
        #print (np.average(X[0]))
        X = np.asarray(X)
        #TODO: Add normalization
        X = X/255
        labels = np.zeros(num_classes, dtype = np.int64)
        for i in range (num_classes):
            labels[i]=i
        lb = LabelBinarizer().fit(labels)
        Y = lb.transform(Y)
    return X, Y

#########################################################################

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

def calculate_hog(hog, img):
    h = hog.compute(img)
    npNumber = np.zeros(h.shape[0], dtype=np.float)
    for i in range(0, h.shape[0]):
        npNumber[i]=h[i]
    return npNumber

def train_svm(hog, train_data, train_labels):
    X = []
    #Ximg,Y = load_data(data_dir)
    for i in train_data:
        X.append(calculate_hog(hog, i))

    print ("Number of training samples: ", len(X))
    clf = svm.SVC(C=1.95, tol=1.55, cache_size=100, kernel='rbf', probability=False)
    data_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    print (np.average(X[0]))
    data_scalerTransf = data_scaler.fit_transform(X)
    print (np.average(data_scalerTransf[0]))
    clf.fit(data_scalerTransf,train_labels)

    return clf, data_scaler

def train_many_svm(hog, data_dir):
    data = []
    dataImg, labels = load_data(data_dir, 'hog') #loading all data
    for i in dataImg:
        data.append(calculate_hog(hog, i))

    train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
    print (eval_labels)
    scores = []

    #cList = (0.1, 0.5, 1, 2, 4, 6, 8, 10, 100, 200)
    X = np.arange(0.05, 2, 0.1)
    #tolList = (1e-4, 1e-3, 1e-2, 0.1, 1, 10)
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
    ###Saving the classifier configuration###
    joblib.dump(classifier,'classifier.pkl')
    joblib.dump(scaler,'scaler.pkl')

def load_clf_svm():
    classifier = joblib.load('classifier.pkl')
    data_scaler = joblib.load('scaler.pkl')

    return classifier, data_scaler

def predict_svm(image, classifier, data_scaler, hog):
    img_hog = calculate_hog(hog, image)
    trans_data = data_scaler.transform(img_hog.reshape(1,-1))
    prediction = classifier.predict(trans_data)
    return prediction

def evaluate_svm(eval_data, eval_labels, classifier, data_scaler, hog, confusion_matrix=False):
    predictions = []
    pass_number = 0
    n_samples = len(eval_data)
    for j in range(n_samples):
        img_hog = calculate_hog(hog, eval_data[j])
        transf_data = data_scaler.transform(img_hog.reshape(1,-1))
        prediction = classifier.predict(transf_data)
        if(confusion_matrix): predictions.append(prediction)
        if (prediction == eval_labels[j]): pass_number += 1

    print ("Performance: ",np.around(pass_number/n_samples*100,1),"%") 
    print ("Number of test samples: ", n_samples)
    
    if (confusion_matrix):
        conf_mx = metrics.confusion_matrix(eval_labels, predictions)
        print("Confusion matrix: \n", conf_mx)
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()

#########################################################################

#Layers parameters
filter_size1 = 5
num_filters1 = 64
filter_size2 = 5
num_filters2 = 150
fc_size = 1000
num_iterations = 2000
train_batch_size = 50

img_size = 64
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 3
num_classes = 11

#Random weights
def new_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name+"_W")

#Biases = 0.05
def new_biases(length, name):
    return tf.Variable(tf.constant(0.05,shape=[length]), name=name+"_B")

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True, name="conv"):
    with tf.name_scope(name):
        #Filter shape for convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        #Creating weights of filter size
        weights = new_weights(shape, name=name)

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
    with tf.name_scope(name):
        layer_shape = layer.get_shape()
        #layer_shape = [num_images, img_height, img_width, num_channels]
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, name="fc"):
    with tf.name_scope(name):
        weights = new_weights(shape=[num_inputs, num_outputs], name=name)
        biases = new_biases(length=num_outputs, name=name)

        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        return layer

def dropout_layer(input, rate, name="dropout"):
    with tf.name_scope(name):
        layer_drop = tf.nn.dropout(input, rate)
        return layer_drop

def make_hparam_string(learning_rate, use_two_conv, use_two_fc):
    if use_two_conv:
        conv_param = "conv=2"
    else:
        conv_param = "conv=1"
    if use_two_fc:
        fc_param = "fc=2"
    else:
        fc_param = "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def train_model():
    global train_dataset 
    global train_database_size
    global test_data, test_labels
    global num_iterations, train_batch_size
   
    sess = tf.Session()
    
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    #x_norm = tf.nn.l2_normalize(x, axis = None, epsilon=1e-12, name='x_norm')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis = 1)

    layer_conv1, weights_conv1 = new_conv_layer(x, num_channels, filter_size1, num_filters1, True, 'conv1')
    layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, True, 'conv2')
    layer_flat, num_features = flatten_layer(layer_conv2, 'flat1')
    layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, True, 'fc1')
    dropout = dropout_layer(layer_fc1, 0.3, name='drop1')
    layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, False, 'fc2')
    y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
    y_pred_cls = tf.argmax(y_pred, axis=1)

    with tf.name_scope('cross_entropy'):
       with tf.name_scope('total'):
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits=y_pred)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_true_cls, logits=y_pred)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        #cost = tf.reduce_min(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_predictions = tf.equal(y_pred_cls, y_true_cls)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    train_dataset = train_dataset.batch(train_batch_size)
    iterator = train_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    num_full_batches = int(train_database_size/train_batch_size)
    batch_idx = 1

    #managing summaries
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+'test')
    train_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    start_time = time.time()

    for i in range(num_iterations):
        if (batch_idx == num_full_batches):
            train_dataset.shuffle(1000)
            batch_idx = 1

        batch = sess.run(next_batch)
        batch_idx += 1
        
        feed_dict_train = {x: batch[0], y_true: batch[1]}
        feed_dict_test = {x: test_data, y_true: test_labels}
        sess.run(optimizer, feed_dict=feed_dict_train)

        if (i % 50 == 0):
            summary, acc = sess.run([merged_summary, accuracy], feed_dict=feed_dict_test)
            test_writer.add_summary(summary, i)

            print("Optimization Interation: ", i+1, " Training Accuracy: ", np.around(acc,3), " Testing Accuracy: ", np.around(acc,3))
            new_data = cv2.resize(cv2.imread('1.jpg'), (64,64))
            print(predict_tf(sess, x, y_pred, new_data))
            new_data = cv2.resize(cv2.imread('2.jpg'), (64,64))
            print(predict_tf(sess, x, y_pred, new_data))
            new_data = cv2.resize(cv2.imread('3.jpg'), (64,64))
            print(predict_tf(sess, x, y_pred, new_data))
            new_data = cv2.resize(cv2.imread('4.jpg'), (64,64))
            print(predict_tf(sess, x, y_pred, new_data))
            new_data = cv2.resize(cv2.imread('5.jpg'), (64,64))
            print(predict_tf(sess, x, y_pred, new_data))
        if (i%10 == 0):
            #summary_train, acc_train = sess.run([merged_summary,accuracy], feed_dict=feed_dict_train)
            summary, _ = sess.run([merged_summary, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

    saver = tf.train.Saver()
    saver.save(sess, log_dir+'mymodel')
    end_time = time.time()

    print ('Learning time:', np.around((end_time-start_time), 2), ' s')
    #new_data = cv2.resize(cv2.imread('4.jpg'), (64,64))
    #print(predict_tf(sess, x, y_pred, new_data))
    #new_data = cv2.resize(cv2.imread('5.jpg'), (64,64))
    #print(predict_tf(sess, x, y_pred, new_data))

    sess.close()

def load_model():
    start = time.time()
    saver = tf.train.import_meta_graph(log_dir+'mymodel.meta')

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    graph = tf.get_default_graph()
    #print (tf.global_variables())
    #print(sess.run(graph.get_tensor_by_name('conv1/conv1_W:0')))
    input = sess.graph.get_operation_by_name('x').outputs[0]
    prediction = sess.graph.get_operation_by_name('y_pred').outputs[0]
    print ("Tensorflow model loading time: ", time.time()-start)
    return sess, graph, input, prediction

def predict_tf(sess, input, output, new_data):
    new_data = cv2.resize(new_data, (64,64))
    new_data = new_data/255;
    predictions = sess.run(output, feed_dict={input:[new_data]})
    pred_class = np.argmax(predictions)
    pred_value = predictions[0][pred_class]
    return pred_class, pred_value

##############################################################################################################

'''#Uczenie modelu tf
data, labels = load_data(data_dir, 'tf')
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).repeat()
train_database_size = train_data.shape[0]
train_model()
'''

''' #Uczenie modelu svm
data, labels = load_data(data_dir, 'svm')
train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.1, shuffle=True)
hog = initialize_hog()
clf, scaler = train_svm(hog, train_data, train_labels)
'''

''' #Ewaluacja modelu svm
data, labels = load_data(data_dir, 'svm')
train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
clf, scaler = load_clf_svm()
evaluate_svm(eval_data, eval_labels, clf, scaler, hog, True)
save_clf_svm(clf, scaler)
'''

''' #Wczytanie i klasyfikacja nowych danych tf
sess, graph, input, prediction = load_model()

new_data = cv2.imread('2.jpg')
print(predict_tf(sess, input, prediction, new_data))
new_data = cv2.imread('3.jpg')
print(predict_tf(sess, input, prediction, new_data))
new_data = cv2.imread('4.jpg')
print(predict_tf(sess, input, prediction, new_data))
new_data = cv2.imread('5.jpg')
print(predict_tf(sess, input, prediction, new_data))

sess.close()
'''

#data, labels = load_data(data_dir, 'svm')
Xs, Ys = load_data(val_path, 'svm')
clf, scaler = load_clf_svm()
hog = initialize_hog()
#train_svm(hog, data, labels)
evaluate_svm(Xs, Ys, clf, scaler, hog, True)

#Porównanie SVM i TF


Xt, Yt = load_data(val_path, 'tf')
sess, graph, input, output = load_model()
pass_number = 0
n_samples = len(Xt)

for j in range(n_samples):
    prediction, value = predict_tf(sess, input, output, Xt[j])
    if (prediction == np.argmax(Yt[j])): pass_number += 1

print ("Performance: ",np.around(pass_number/n_samples*100,1),"%") 
new_data = cv2.resize(cv2.imread('1.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))
new_data = cv2.resize(cv2.imread('2.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))
new_data = cv2.resize(cv2.imread('3.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))
print('another session')
new_data = cv2.resize(cv2.imread('1.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))
new_data = cv2.resize(cv2.imread('2.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))
new_data = cv2.resize(cv2.imread('3.jpg'), (64,64))
print(predict_tf(sess, input, output, new_data))

sess.close()
'''
Xs, Ys = load_data(val_path, 'svm')
clf, scaler = load_clf_svm()
hog = initialize_hog()
evaluate_svm(Xs, Ys, clf, scaler, hog)
'''