import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from skimage import transform, util, exposure
import parameters

# Image loading

def get_images_paths(path: str, do_random=True) -> list:
    '''
    Getting the image filenames from the defined location (only jpg, png and ppm) with optional randomization
    '''
    filenames = []
    dir = os.fsencode(path)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".ppm"): 
            filenames.append(os.fsdecode(dir+file))
    if (do_random): random.shuffle(filenames)
    return filenames

def load_data(path, show_hist=False, one_hot=True, type='svm') -> tuple:
    '''
    Loading image files, converting them to numpy arrays, resizing to a required format for the classifier, assigning correct label
    '''
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
        plt.xticks(labels, parameters.signs_dict.values(), rotation=90)
        plt.show()
    if(type=="tf" and one_hot):
        Y = np.asarray(Y)
        X = np.asarray(X)
        lb = LabelBinarizer().fit(labels)
        Y = lb.transform(Y)
    return (X, Y)

# Image manupulations

def flip_and_save(path, i):
    print (path)
    img = cv2.imread(path)
    img = cv2.flip(img, 1)
    cv2.imwrite('ll_'+str(i)+'.png', img)

def flip_signs(to_folder_path: str):
    i=0
    filenames = []
    dir = os.fsencode(to_folder_path)
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

def x_cord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        moment = int(M['m10']/M['m00'])
        return (moment)
    else:
        return (0)

# ROI detection

class ROI:
    '''
    This class represent ROI - region of interest. This is a area in the picture with high probability that a road sign is present.
    '''
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
        cv2.putText(image, parameters.signs_dict[str(self.prediction)],(self.x + self.w, self.y + 20), cv2.FONT_ITALIC, 1, (0,255,0))
        #cv2.circle(image, (self.x + int(self.w/2), self.y + int(self.h/2)), int(self.w/2), (0,255,0), 2)
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0,255,0), 2)

    def enhance_roi(self,factor):
        if (self.x-factor/2*self.w)>0 and (self.y-factor/2*self.h)>0: 
            self.x-=int(factor/2*self.w)
            self.y-=int(factor/2*self.h)
            self.w =int(self.w*(1+factor))
            self.h =int(self.h*(1+factor))

def detect_shapes(color_img, roi_list, exe_mode = 'test'):
    #TODO: check if maybe this function could be better
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

def exclude_contours_1(boundingRects, enhancement_factor, image, exe_mode = "test"):
    '''
    A simple filter for the ROI frames based on its size
    '''
    roi_list = []
    height, width = image.shape[:2]
    frame_size = height*width
    for i in boundingRects:
        x, y, w, h = i
        if ((w*h>4e-4*frame_size) and (w*h<2e-2*frame_size) and (w/h>0.5) and ((w/h)<1.5)):
            new_roi = ROI(x,y,w,h,'x',"unknown")
            new_roi.enhance_roi(enhancement_factor)
            roi_list.append(new_roi)
            if (exe_mode == "test"): 
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), 1)
    #cv2.imshow("Contours detected by color", image)
    #if (exe_mode == 'test'): cv2.waitKey(0)
    return roi_list

def exclude_contours_2(contours, parent_roi, enhancement_factor, image, exe_mode = "test"):
    '''
    A simple filter for the ROI frames based on its size
    '''
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

def extract_shapes(child_roi_list, parent_roi):
    roi_list_out = []
    for roi in child_roi_list:
        new_roi = ROI(roi.x+parent_roi.x, roi.y+parent_roi.y, roi.w, roi.h)
        roi_list_out.append(new_roi)
    return roi_list_out

def find_roi_by_color(colorImg, color_mask_obj, exe_mode = 'test'):
    global frame_size
    roi_list = []

    # Find contours based on the colour mask
    mask = color_mask_obj.find_color_mask(colorImg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRects = [None]*len(contours)
    #centers = [None]*len(contours)
    #radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRects[i] = cv2.boundingRect(contours_poly[i])
        #centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    roi_list = exclude_contours_1(boundRects, 0.2, mask, exe_mode)

    cv2.imshow("Contours detected", mask)
    if (exe_mode == 'test'): cv2.waitKey(0)
    cv2.destroyWindow("Contours detected")
    if (exe_mode == "test"): print ("Number of color roi: ", len(roi_list))
    return roi_list

def detect_round_signs(color_image, roi_list, exe_mode = 'test'):
    roi_list_out = []
    for roi in roi_list:
        x_roi,y_roi,w_roi,h_roi,field = roi.x, roi.y, roi.w, roi.h, roi.roi_size
        roi_img = color_image[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi]
        roi_img_copy = roi_img.copy()
        cv2.imshow("ROI image", roi_img)
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        min_dim = min(roi_img.shape[0], roi_img.shape[1])
        max_dim = max(roi_img.shape[0], roi_img.shape[1])
        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=int(max_dim*0.25), param1=14, param2=33, 
                                   minRadius=int(min_dim*0.2), maxRadius=int(min_dim*0.55))

        if (exe_mode=='test'): 
            cv2.imshow('ROI blurred', cv2.resize(gray_roi, (64,64)))
            if circles is not None: 
                print("Number of detected circles: ", circles.shape[1], '`n')
            
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
        #cv2.destroyWindow("ROI image")
        #cv2.destroyWindow("ROI blurred")
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

# Analysis
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
