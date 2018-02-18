import numpy as np
import cv2, time, sklearn, os, glob
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from hog_svm import loadClassifier, initializeHOG, predictSign

###File number for saving images###
fileNumber = 1

###Parameters for image segmentation###
RED1 = (45,40,40)
RED2 = (125,255,255)
redMask = (RED1, RED2, "red")

#night mode
#RED1 = (45,30,30)
#RED2 = (125,255,255)

class ROI:
    def __init__(self, x, y, w, h, mask):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mask = mask
        
        self.roiSize = w*h      
        self.shape = "unknown"
        self.prediction = "none"

    def showSign(self, image):
        cv2.putText(image, self.prediction,(self.x + self.w, self.y + 20), cv2.FONT_ITALIC, 1.5, (255,255,255))
        cv2.circle(image, (self.x + int(self.w/2), self.y + int(self.h/2)), int(self.w/2), (0,255,0), 2)
        
def findRoiByColour(colourImg, colourMask, exeMode):
    colourThreshMin, colourThreshMax, maskType = colourMask
    mask = findColorsMask(colourImg, colourThreshMin, colourThreshMax, exeMode)
    picture, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roiList = roiDetection(colourImg, contours, 0.1, maskType, exeMode)
    return roiList

def findColorsMask(img, color1, color2, exeMode):
    kernel3 = np.ones((3,3),np.uint8)
    kernel5 = np.ones((5,5),np.uint8)
    
    hsv = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color1, color2)
    if (exeMode=="test"): cv2.imshow('mask1', mask)
    mask = cv2.dilate(mask,kernel5,iterations=1)
    mask = cv2.erode(mask,kernel3,iterations=1)
    #mask = cv2.erode(mask,kernel,iterations=1)
    if (exeMode=="test"): cv2.imshow('mask2', mask)
    return mask

def roiDetection(colourImg, contours, enhacementFactor, maskColour, exeMode):
    imgField = colourImg.shape[0]*colourImg.shape[1]
    roiList=[]
    index = 1
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        x, y, w, h = enhanceRoi(x,y,w,h,enhacementFactor)
        newRoi = ROI(x,y,w,h,maskColour)
        if ((newRoi.roiSize>6e-4*imgField) and (newRoi.roiSize<6e-2*imgField) and (w/h>0.45) and ((w/h)<1.5)):
            roiList.append(newRoi)
            if (exeMode=="test"):
                cv2.rectangle(colourImg, (x,y), (x+w,y+h), (255,255,255), 1)
                cv2.putText(colourImg,str(index),(x+w,y+20), cv2.FONT_ITALIC, 1.5, (255,255,255))
                print ('roi nr ',str(index), '  w/h', w/h)
                index+=1
    if (exeMode=="test"):
        cv2.imshow('rois', colourImg)
        cv2.waitKey(0)
        #cv2.destroyWindow('mask1')
        #cv2.destroyWindow('mask2')
        #cv2.destroyWindow('rois')
    return roiList

def enhanceRoi(x,y,w,h,factor):
    if (x-factor/2*w)>0 and (y-factor/2*h)>0: 
        x-=int(factor/2*w)
        y-=int(factor/2*h)
        w*=int(1+factor)
        h*=int(1+factor)
    return x,y,w,h

def showSigns(contours, roiList, img):
    contoursRound = []
    contoursRect = []
    image = img.copy()

    for i in range (len(contours)):
        arcLength = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.01*arcLength, True)
        x,y,w,h,field = roiList[i]
        
        if (len(approx)>8 and len(approx)<30):
            cv2.putText(image,"Circle",(x+w,y+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
            cv2.circle(image, (x+int(w/2),y+int(h/2)), int(w/2), (0,255,0), 2)
            contoursRound.append(contours[i])     
        elif (len(approx)>3 and len(approx)<8):
            cv2.putText(image,"Rectangle",(x+w,y+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            contoursRect.append(contours[i])

    return contoursRound, contoursRect, image

def detectRoundSigns(colourImage, roiList, exeMode):
    roiListOut = []
    index = 1
    for i in roiList:
        xRoi,yRoi,wRoi,hRoi,field = i.x, i.y, i.w, i.h, i.roiSize
        sizeX, sizeY = colourImage.shape[:2]
        roiImg = colourImage[yRoi:yRoi+hRoi,xRoi:xRoi+wRoi]
        grayRoi = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
        if (exeMode=='test'): 
            cv2.imshow('roi', grayRoi)
        
        #circles = cv2.HoughCircles(grayRoi,cv2.HOUGH_GRADIENT,dp=1,minDist=5,param1=40,param2=int(wRoi*hRoi/100),minRadius=int(wRoi*0.25),maxRadius=int(wRoi/2))
        circles = cv2.HoughCircles(grayRoi,cv2.HOUGH_GRADIENT,dp=1,minDist=5,param1=40,param2=10,minRadius=int(wRoi*0.25),maxRadius=int(wRoi/2))

        if (wRoi/hRoi)>0.8 and (wRoi/hRoi)<1.2:
            if (exeMode=='test'): print ('roi nr', str(index), 'option 1','param2: ',wRoi*hRoi/100, 'ratio: ', wRoi/hRoi)
            if circles is not None: i.shape = "circle"
            else: i.shape = "unknown"
            roiListOut.append(i)
        elif circles is not None and len(circles[0])<3: #if more than 3 circles also false
            if (exeMode=='test'): print ('roi nr', str(index), 'option 2','param2: ',wRoi*hRoi/100, 'ratio: ', wRoi/hRoi)
            circles = np.uint16(np.around(circles))
            for j in circles[0,:]:
                if (exeMode=='test'): 
                    roiImgCopy = roiImg.copy()
                    cv2.circle(roiImgCopy,(j[0],j[1]),j[2],(0,255,0),1)
                xCircle, yCircle = j[:2]
                rCircle = int(j[2]*1.2)
                
                if (xCircle-rCircle<0): x1=0 
                else: x1=xCircle-rCircle
                if (xCircle+rCircle>wRoi): x2=wRoi
                else: x2=xCircle+rCircle
                if(yCircle-rCircle<0): y1=0
                else: y1=yCircle-rCircle
                if(yCircle+rCircle>hRoi): y2=hRoi
                else: y2=yCircle+rCircle

                i.x = int(xRoi + x1)
                i.y = int(yRoi + y1)
                i.w = int(x2-x1)
                i.h = int(y2-y1)
                i.roiSize = i.w*i.h
                i.shape = "circle"

                if (exeMode=="test"):
                    cv2.circle(roiImg,(xCircle,yCircle),rCircle,(0,255,0),1)
                    cv2.imshow('detected circles',cv2.resize(roiImg,(64,64)))
                    cv2.waitKey(0)
                    cv2.destroyWindow('detected circles')
                roiListOut.append(i)
        else:
            if (exeMode=='test'): print ('roi nr: ', str(index), 'option 3 - REJECTED', 'param2: ',wRoi*hRoi/100, 'ratio: ', wRoi/hRoi)
            #cv2.imshow('missing circles',cv2.resize(roiImg,(64,64)))
            #cv2.waitKey(0)
            #cv2.destroyWindow('missing circles')
        #cv2.drawContours(col,contours,-1,(255,0,0),1)
        if (exeMode=='test'):
            cv2.waitKey(0)
            index+=1
    return roiListOut

def x_cord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        moment = int(M['m10']/M['m00'])
        return (moment)
    else:
        return (0)

###Parameter for performance test###
start = time.time()
camera = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi',-1, 20.0, (640,480))

directory_in_str = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/SignRecognition/snapshots/"
#directory_in_str = "C:/Users/Michau/Desktop/GermanTrafficSigns/SpeedLimits/20/"

hog = initializeHOG()
clf, xScaler = loadClassifier()

before = time.time()
while(False):
    # Capture frame-by-frame
    ret, frame = camera.read()
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    #frame = cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)))
    imgField = frame.shape[0]*frame.shape[1]    

    #Finding rois
    roiList = findRoiByColour(frame, imgField, RED1, RED2)
    roiLits = detectRoundSigns(frame, roiList)

    if roiList:
        for i in roiList:
            x,y,w,h,field = i
            prediction = predictSign(cv2.resize(frame[y:y+h,x:x+w],(64,64)),clf,xScaler,hog)
            if prediction != 'none': showSign(i, frame, prediction)
            #cv2.imwrite('new'+str(fileNumber)+'.jpg',signColor)
            fileNumber+=1
            
    cv2.imshow('Detection', frame)
    print ("execution", time.time()-before)
    out.write(frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture 
camera.release()
out.release()

filenames = []

directory = os.fsencode(directory_in_str)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        filenames.append(os.fsdecode(directory+file))

fileNumber = 0
for i in range(len(filenames)):
    print('%s' % filenames[i])

    #Preprocessing of image
    colour = cv2.imread(filenames[i]) 

    #Finding rois
    roiList = findRoiByColour(colour.copy(), redMask, 'test')
    roiList = detectRoundSigns(colour.copy(), roiList, 'test')

    if roiList:
        for i in roiList:
            x,y,w,h,field = i.x, i.y, i.w, i.h, i.roiSize
            sign = colour[y:y+h,x:x+w]
            sign = cv2.resize(colour[y:y+h,x:x+w],(64,64))
            prediction = predictSign(sign,clf,xScaler,hog)
            if prediction[0] != 'none':
                i.prediction = prediction[0]
                i.showSign(colour)
            cv2.imshow('Detection', colour)
            #cv2.imwrite('new'+str(fileNumber)+'.jpg', sign)
            fileNumber+=1
    else:
        cv2.imshow('NO-Detection', colour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

finish = time.time()
print("Execution time:",finish-start)
cv2.waitKey(0)