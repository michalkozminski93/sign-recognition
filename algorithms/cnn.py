import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from parameters import img_size_cnn, num_channels_cnn, dir_tf_model, model_dir, train_test_dir, val_path, signs_dict
from algorithms.preprocessing import load_data

def cnn_parameters():
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

def load_cnn_model_sess():
    start = time.time()
    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=[None, img_size_cnn, img_size_cnn, num_channels_cnn])
        is_training = tf.constant(False)
        with tf.variable_scope('cnn'):
            layer_out, weights_out = return_model_cnn(input, is_training)
            output = tf.nn.softmax(layer_out)

    sess = tf.Session(graph = graph)
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(dir_tf_model))
    print ("Tensorflow model loading time: ", time.time()-start)
    return sess, input, output, is_training

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

def predict_cnn(sess, input, output, is_training, image):
    '''
    Sign content prediction based on the provided ROI image
    '''
    image = cv2.resize(image, (img_size_cnn, img_size_cnn))
    if (len(image.shape) > 2 and image.shape[2] == 3): image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    image = np.expand_dims(image, axis=2)

    predictions = sess.run(output, feed_dict={input:[image], is_training:False})
    pred_class = np.argmax(predictions)
    pred_value = predictions[0][pred_class]
    return pred_class, pred_value

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

def return_model_cnn(x, is_training):
    #Convolution layer
    layer_conv1, weights_conv1 = new_conv_layer(x, num_channels_cnn, filter_size1, num_filters1, use_pooling = True, name = 'conv1')

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
    layer_fc2, weights_fc2 = new_fc_layer(layer_fc1, fc_size, num_channels_cnn, use_relu=False, name='fc2')

    #Dropout on fully connected layer
    if (drop_fc2 is not None):
        layer_fc2 = tf.cond(is_training, lambda: dropout_layer(layer_fc2, drop_fc2, name = 'drop_fc2'), lambda: layer_fc2)

    return layer_fc2, weights_fc2

def train_model_cnn(train_dataset, test_data, test_labels):
    global train_batch_size, num_iterations

    graph = tf.Graph()
    #Graph preparation
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, img_size_cnn, img_size_cnn, num_channels_cnn], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, num_channels_cnn], name='y_true')
        is_training = tf.placeholder(tf.bool, shape = [], name='is_training')
    
        with tf.variable_scope('cnn'):
            layer_fc2, weights_fc2 = return_model_cnn(x, is_training)
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

            summary, _ = sess.run([merged_summary, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

            if (i % 100 == 0): 
                saver.save(sess, model_dir + 'mymodel ' + desc, i)
            print ('Iteration: ', i)
        
        saver.save(sess, model_dir + 'mymodel ' + desc)
        print ('Learning time:', np.around((time.time()-start_time), 2), ' s')

def evaluate_model(Xt, Yt, confusion_matrix=False, type='svm', clf=None, scaler=None, hog=None):
    predictions = []
    n_samples = len(Xt)
    pass_number = 0
    
    start = time.time()

    #Model evaluation for Convolutional Neural Network
    if (type == 'cnn'):
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, img_size_cnn, img_size_cnn, num_channels_cnn])
            is_training = tf.constant(False)
            with tf.variable_scope('cnn'):
                layer_out, weights_out = return_model_cnn(x, is_training)
                y_pred = tf.nn.softmax(layer_out)

        with tf.Session(graph = graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
            print ("Tensorflow model loading time: ", time.time()-start)
            for j in range(n_samples):
                prediction, value = predict_cnn(sess, x, y_pred, is_training, Xt[j])
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

def run_train_cnn():
    data, labels = load_data(train_test_dir, show_hist = False, one_hot=True, type = 'tf')
    print ('All data: ', len(data))
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, shuffle=True)
    print ('Training data: ', len(train_data))

    train_data, test_data = tf.preprocess(train_data, test_data, hist_equalization)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).repeat()
    train_database_size = train_data.shape[0]

    train_model_cnn(train_dataset, test_data, test_labels)

    val_data, val_labels = load_data(val_path, False, False, 'tf')
    print ('Validation data: ', len(val_data))
    evaluate_model(val_data, val_labels, True, 'tf')

