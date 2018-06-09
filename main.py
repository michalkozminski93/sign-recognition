import numpy as np
import tensorflow as tf
import cv2, time, os, glob
from classification import load_clf_svm, initialize_hog, predict_svm, load_model, predict_tf, return_model_tf
from matplotlib import pyplot as plt
from random import shuffle

class ROI:
    def __init__(self, x, y, w, h, mask = 'unknown', shape = 'unknown'):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mask = mask
        
        self.roi_size = w*h      
        self.shape = shape
        self.prediction = -1

    def show_sign(self, image):
        cv2.putText(image, sign_desc[str(self.prediction)],(self.x + self.w, self.y + 20), cv2.FONT_ITALIC, 1, (0,255,0))
        #cv2.circle(image, (self.x + int(self.w/2), self.y + int(self.h/2)), int(self.w/2), (0,255,0), 2)
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0,255,0), 2)

    def enhance_roi(self,factor):
        if (self.x-factor/2*self.w)>0 and (self.y-factor/2*self.h)>0: 
            self.x-=int(factor/2*self.w)
            self.y-=int(factor/2*self.h)
            self.w =int(self.w*(1+factor))
            self.h =int(self.h*(1+factor))
  
class color_mask:
    def __init__(self, thresh1, thresh2, color):
        self.threshMin = thresh1
        self.threshMax = thresh2
        self.type = color

    def find_color_mask(self, img):
        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)

        hsv = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.threshMin, self.threshMax)

        #mask2 = np.zeros((mask.shape[0], mask.shape[1]), dtype = np.uint8)
        #mask right
        #mask2[int(mask.shape[0]*0.2):int(mask.shape[0]*0.7), int(mask.shape[1]*0.6):int(mask.shape[1]*0.95)] = 255
        #mask left
        #mask2[int(mask.shape[0]*0.2):int(mask.shape[0]*0.7), int(mask.shape[1]*0.05):int(mask.shape[1]*0.3)] = 255
        #mask = cv2.bitwise_and(mask, mask, mask = mask2)

        #point1, point2 = (int(frame.shape[1]*0.6),int(frame.shape[0]*0.1)), (int(frame.shape[1]*0.9),int(frame.shape[0]*0.6)) 
        mask = cv2.medianBlur(mask, 3)
        #mask = cv2.dilate(mask,kernel3,iterations=1)
        #mask = cv2.erode(mask,kernel3,iterations=1)
        #mask = cv2.erode(mask,kernel,iterations=1)
        return mask

def get_histogram(img):
    cv2.imshow('color', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    hist = cv2.calcHist([hsv], [1], None, [256], [0,255])
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    cv2.imshow('h', h)
    cv2.imshow('s', s)
    cv2.imshow('v', v)
    max = np.max(hist)
    index = np.argmax(hist)
    print (index, max)
    cv2.waitKey(0)

def histogram(path):
    img = cv2.imread(path, 0)
    plt.hist(img.ravel(), 256, [0,256])
    plt.show()

def exclude_contours_1(contours, enhancement_factor, image, exe_mode = "test"):
    roi_list = []
    global frame_size
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if ((w*h>4e-4*frame_size) and (w*h<2e-2*frame_size) and (w/h>0.5) and ((w/h)<1.5)):
            new_roi = ROI(x,y,w,h,'x',"unknown")
            new_roi.enhance_roi(enhancement_factor)
            roi_list.append(new_roi)
            if (exe_mode == "test"): 
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), 1)
    #cv2.imshow("Contours detected by color", image)
    if (exe_mode == 'test'): cv2.waitKey(0)
    return roi_list

def exclude_contours_2(contours, parent_roi, enhancement_factor, image, exe_mode = "test"):
    roi_list_out = []
    global frame_size
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if ((w*h>7e-4*frame_size) and (w*h<6e-2*frame_size) and (w/h>0.45) and ((w/h)<1.5)):
            new_roi = ROI(x,y,w,h,'x',"unknown")
            new_roi.enhance_roi(enhancement_factor)
            
            arcLength = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*arcLength, True)
            if (len(approx)>10 and len(approx)<18):
                print ('circle ', len(approx))
                cv2.drawContours(image, i, -1, (255,0,0), 2)
                if (exe_mode == "test"): 
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), 1)
                    cv2.imshow("Contours detected", cv2.resize(image, (64,64)))
                    cv2.waitKey(0)
                new_rois = extract_shapes([new_roi], parent_roi)
                roi_list_out.extend(new_rois)
            elif (len(approx)>3 and len(approx)<10):
                print ('rectangle', len(approx))
            elif (len(approx)<=3):
                print ('triangle', len(approx))
            else:
                print ('no fucking idea', len(approx))
    return roi_list_out

def find_roi_by_color(colorImg, color_mask_obj, exe_mode = 'test'):
    global frame_size
    roi_list = []

    mask = color_mask_obj.find_color_mask(colorImg)
    picture, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Contours detected by color :"+ color_mask_obj.type, mask)
    roi_list = exclude_contours_1(contours, 0.2, mask, exe_mode)

    if (exe_mode == "test"): print ("Number of color roi: ", len(roi_list))
    return roi_list

def detect_shapes(color_img, roi_list, exe_mode = 'test'):
    circles_list = []
    for roi in roi_list:
        x_roi,y_roi,w_roi,h_roi,field = roi.x, roi.y, roi.w, roi.h, roi.roi_size
        roi_img = color_img[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi]
        roi_img = cv2.resize(roi_img, (64,64))
        roi_img_copy = roi_img.copy()

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 0, 220)

        if (exe_mode == "test"): 
            cv2.imshow('color', cv2.resize(roi_img, (64,64)))
            cv2.imshow('gray', cv2.resize(gray, (64,64)))
            cv2.imshow('canny', cv2.resize(canny, (64,64)))
            cv2.waitKey(0)
        picture, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        roi_list_out = exclude_contours_2(contours, roi, 0.4, roi_img_copy, exe_mode)
    return roi_list_out

def detect_round_signs(color_image, roi_list, exe_mode = 'test'):
    roi_list_out = []
    for roi in roi_list:
        x_roi,y_roi,w_roi,h_roi,field = roi.x, roi.y, roi.w, roi.h, roi.roi_size
        roi_img = color_image[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi]
        roi_img_copy = roi_img.copy()
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        min_dim = min(roi_img.shape[0], roi_img.shape[1])
        max_dim = max(roi_img.shape[0], roi_img.shape[1])
        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=int(max_dim*0.25), param1=14, param2=33, minRadius=int(min_dim*0.2), maxRadius=int(min_dim*0.55))

        if (exe_mode=='test'): 
            cv2.imshow('ROI blurred', cv2.resize(gray_roi, (64,64)))
            if circles is not None: 
                print("Number of detected circles: ", circles.shape[1], '`n')
                #for circle in circles:
                #    cv2.circle(roi_img_copy, (circle[0], circle[1]), circle[2], (255,0,0), 1)
                #cv2.imshow("Detected circles", cv2.resize(roi_img_copy, (64,64)))
            
        if (w_roi/h_roi)>0.7 and (w_roi/h_roi)<1.2:
            if (exe_mode=='test'): print ('Ratio ok', 'param2: ',w_roi*h_roi/100, 'ratio: ', np.around(w_roi/h_roi,2))
            if circles is not None and len(circles[0])<=5: #if more than 5 circles also false 
                new_rois = extract_circles_from_roi(roi, circles, roi_img.copy(), color_image, exe_mode)
                roi_list_out.extend(new_rois)
            else: 
                roi.shape = "unknown"
                roi_list_out.append(roi)
        elif circles is not None and len(circles[0])<=10: #if more than 5 circles also false
            if (exe_mode=='test'): print ('Ratio nok', 'param2: ',w_roi*h_roi/100, 'ratio: ', np.around(w_roi/h_roi,2))
            new_rois = extract_circles_from_roi(roi, circles, roi_img.copy(), color_image, exe_mode)
            roi_list_out.extend(new_rois)
        else:
            if (exe_mode=='test'): print ('No regions detected', 'param2: ',w_roi*h_roi/100, 'ratio: ', np.around(w_roi/h_roi,2))
        if (exe_mode == 'test'): cv2.waitKey(0)
        cv2.destroyWindow("Detected circles")
        cv2.destroyWindow("ROI blurred")
    return roi_list_out

def extract_shapes(child_roi_list, parent_roi):
    roi_list_out = []
    for roi in child_roi_list:
        new_roi = ROI(roi.x+parent_roi.x, roi.y+parent_roi.y, roi.w, roi.h)
        roi_list_out.append(new_roi)
    return roi_list_out

def extract_circles_from_roi(roi, circles, roi_img, colorImg, exe_mode = 'test'):
    roi_list_out = []
    circles = np.uint16(np.around(circles))
    for circle in circles[0,:]:
        xCircle, yCircle = circle[:2]
        #roi extention by factor of 1.44 (1.2*1.2)
        rCircle = int(circle[2]*1.2)
               
        if (xCircle-rCircle<0): x1=0 
        else: x1=xCircle-rCircle
        if (xCircle+rCircle>roi.w): x2=roi.w
        else: x2=xCircle+rCircle
        if(yCircle-rCircle<0): y1=0
        else: y1=yCircle-rCircle
        if(yCircle+rCircle>roi.h): y2=roi.h
        else: y2=yCircle+rCircle

        xNew = int(roi.x + x1)
        yNew = int(roi.y + y1)
        wNew = int(x2-x1)
        hNew = int(y2-y1)

        new_roi = ROI(xNew, yNew, wNew, hNew, roi.mask, roi.shape)

        if (exe_mode=='test'):
            copy = roi_img.copy()
            cv2.circle(copy,(xCircle,yCircle),rCircle,(0,255,0),1)
            cv2.imshow("detected circles", cv2.resize(copy, (64,64)))
            cv2.imshow('roi after circles detection',cv2.resize(colorImg[yNew:yNew+hNew, xNew:xNew+wNew],(64,64)))
            cv2.waitKey(0)
            cv2.destroyWindow('detected circles')
    
        roi_list_out.append(new_roi)
    if (exe_mode=='test'): print ("Number of detected circles: ", len(roi_list_out))
    return roi_list_out

def x_cord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        moment = int(M['m10']/M['m00'])
        return (moment)
    else:
        return (0)

def detect_signs(frame, exe_mode = "normal"):
    global red_mask, yellow_mask, blue_mask, file_number, sess, input, output, is_training

    cv2.imshow('Detection', frame)
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
        resized2 = cv2.resize(roi_img,(64,64))
        class_svm = predict_svm(resized,clf,xScaler,hog)
        class_tf, value_tf = predict_tf(sess, input, output, is_training, resized2)
        print ('tf :', class_tf, value_tf)
        print ('svm :', class_svm)
        if (class_svm[0] != 27): 
            i.prediction = class_svm[0]
            i.show_sign(frame)
            #if (class_svm[0]!=9):
            #    cv2.imwrite('C:/Users/Michau/Desktop/new_signs/ax'+str(file_number)+'.jpg', roi_img)
            #else:
            #cv2.imwrite('C:/Users/Michau/Desktop/new_none/x'+str(file_number)+'.jpg', roi_img)
            #file_number+=1
    execution_time = time.time()-start
    cv2.destroyWindow('color')
    cv2.destroyWindow('canny')
    cv2.destroyWindow('Contours detected')
    cv2.imshow('Detection', frame)
    print ("execution", execution_time)
    print ("______________________________________________________")
    if (exe_mode == 'test'): cv2.waitKey(0)
    return execution_time
    
##############################################################################################################
#Signs description dictionary
sign_desc = {'0':'20', '1':'30', '2':'40', '3':'50', '4':'60', '5':'70', '6':'80', '7':'90', '8':'100', '9':'110', '10':'120', '11':'Zakaz wyprz.', '12':'Ustap piersz.',
            '13':'Stop', '14':'Zakaz wjazdu', '15':'Zakaz ruchu', '16':'Zakaz w lewo', '17':'Zakaz w prawo', '18':'Niebezp. lewo', '19':'Niebezp. prawo', '20':'Zwezenie',
            '21':'Nakaz w prawo', '22':'Nakaz w lewo', '23':'Nakaz prosto', '24':'Z prawej', '25':'Z lewej', '26':'Rondo', '27':'Tlo'}

file_number = 999
dir_snapshots = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/snapshots/"
dir_validation_img = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/validation_img/"
dir_tf_model = "C:/Users/Michau/Desktop/tf_model/"

###Color masks for image segmentation
red_mask = color_mask((70,9,130), (97,255,255), "red")
yellow_mask = color_mask((100,24,140), (110,255,255), "yellow")
blue_mask = color_mask((18,30,193), (22,255,255), "blue")

#Parameters for TF classification
img_size = 32
num_channels = 1

#SVM
hog = initialize_hog()
clf, xScaler = load_clf_svm()

#TENSORFLOW
start = time.time()
graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels])
    is_training = tf.constant(False)
    with tf.variable_scope('cnn'):
        layer_out, weights_out = return_model_tf(input, is_training)
        output = tf.nn.softmax(layer_out)

with tf.Session(graph = graph) as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(dir_tf_model))
    print ("Tensorflow model loading time: ", time.time()-start)

    ##############################################################################################################
    '''#Wideo
    camera = cv2.VideoCapture('afternoon1.wmv')
    ret, frame = camera.read()
    ret, frame = camera.read()

    while(ret):
        # Capture frame-by-frame
        #ret, frame = camera.read()
        ret, frame = camera.read()
        frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        frame_size = frame.shape[0]*frame.shape[1]
        #ret, frame = camera.read()
        before = time.time()
        if ret: detect_signs(frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

    camera.release()
    '''
    ##############################################################################################################
    #Snapshoty
    filenames = []
    dir = os.fsencode(dir_validation_img)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".ppm"): 
            filenames.append(os.fsdecode(dir+file))
    #shuffle(filenames)
    detection_time = []

    for file in filenames:
        print('%s' % file)
        frame = cv2.imread(file)
        print (frame.shape)
        if (frame.shape[1]>1000): frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        frame_size = frame.shape[0]*frame.shape[1]
        detection_time.append(detect_signs(frame, 'normal'))
        cv2.waitKey(0)
    
    print('Average execution time: ', np.mean(np.asarray(detection_time)))
    print('Standard deviation for execution time: ', np.std(np.asarray(detection_time)))
    ##############################################################################################################
   