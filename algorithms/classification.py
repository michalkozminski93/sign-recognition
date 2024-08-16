'''Functions for SVM classification, Tensorflow classification, HOG descriptor'''
import cv2, time
import numpy as np
from algorithms.preprocessing import find_roi_by_color, detect_round_signs, get_images_paths
from parameters import red_mask, yellow_mask, blue_mask
import algorithms.svm as svm
import algorithms.azure_cnn as azure_cnn

def detect_signs(frame, class_method: str, exe_mode = "normal"):
    #sess, input, output, is_training = load_cnn_model_sess()

    #cv2.imshow('Detection', frame)
    frame_copy = frame.copy()
    start = time.time()

    roi_list = []
    roi_list_red = find_roi_by_color(frame, red_mask, exe_mode)
    #roi_list_yellow = find_roi_by_color(frame, yellow_mask)
    roi_list_blue = find_roi_by_color(frame, blue_mask, exe_mode)

    roi_list.extend(roi_list_red)
    roi_list.extend(roi_list_blue)
    #roi_list.extend(roi_list_yellow)

    roi_list = detect_round_signs(frame, roi_list, exe_mode)
    #roi_list = detect_shapes(frame, roi_list)

    for i in roi_list:
        x,y,w,h = i.x, i.y, i.w, i.h
        roi_img = frame_copy[y:y+h,x:x+w]
        resized = cv2.resize(roi_img,(64,64))
        if class_method == "SVM":
            hog = svm.initialize_hog()
            clf, xScaler = svm.load_clf()
            result = svm.predict(resized, clf, xScaler, hog)[0]
        elif class_method == "CNN":
            pass
        else:
            result = azure_cnn.predict(roi_img.astype(np.float32))
        #class_tf, value_tf = cnn.predict_cnn(sess, input, output, is_training, resized2)
        #print ('tf :', class_tf, value_tf)
        if result: 
            i.prediction = result[0]
            i.show_sign(frame)
            #print ('svm :', result)
    execution_time = time.time()-start

    #cv2.imshow('Detection', frame)
    #print ("execution", execution_time)
    #print ("______________________________________________________")
    #cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return execution_time