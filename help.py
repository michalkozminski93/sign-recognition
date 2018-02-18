import os
import cv2
import numpy

#directory_in_str = "C:/Users/Michau/Desktop/GermanTrafficSigns/SpeedLimits/"
directory_snap = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/SignRecognition/snapshots/"
directory_double = "C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/SignRecognition/double/"

if False:#Checking the cirlces finding algorithm in the ROIs
    for i in range(1):
        directory = os.fsencode(directory_double)
        index=999
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                print (directory_in_str+filename)
                imgCol = cv2.imread(directory_double+filename)
                img = cv2.cvtColor(imgCol,cv2.COLOR_BGR2GRAY)
                print ('img shape', img.shape[:])
                wRoi = img.shape[1]
                hRoi = img.shape[0]
                for k in range (1):
                    img2 = img.copy()
                    circles = cv2.HoughCircles(img2,cv2.HOUGH_GRADIENT,dp=1,minDist=5,param1=40,param2=int(wRoi*hRoi/100),minRadius=int(wRoi*0.25),maxRadius=int(wRoi/2))
                    if circles is not None:# and len(circles[0])<3: #if more than 3 circles also false
                        circles = np.uint16(np.around(circles))
                        print ('parametr2=',wRoi*hRoi/100)
                        for j in circles[0,:]:
                            cv2.circle(img2,(j[0],j[1]),j[2],(0,255,0),1)
                            xCircle = j[0]
                            yCircle = j[1]
                            rCircle = int(j[2]*1.2)
                            #1
                            if (xCircle-rCircle<0): x1=0 
                            else: x1=xCircle-rCircle
                            #2
                            if (xCircle+rCircle>wRoi): x2=wRoi
                            else: x2=xCircle+rCircle
                            #3
                            if(yCircle-rCircle<0): y1=0
                            else: y1=yCircle-rCircle
                            #4
                            if(yCircle+rCircle>hRoi): y2=hRoi
                            else: y2=yCircle+rCircle

                            new = imgCol[y1:y2,x1:x2]
                            cv2.imshow('new', new)
                            cv2.imshow('detected circles'+str(k),cv2.resize(img2,(64,64)))
                            #cv2.imwrite('det'+str(index)+'.jpg',cv2.resize(new,(64,64)))
                            print (j, rCircle/img.shape[1])
                            cv2.waitKey(0)
                            index+=1
                    else:
                        cv2.imshow('no circles',cv2.resize(img2,(64,64)))
                    cv2.destroyAllWindows()

if False:#Resizing the data from the snapshots directory
    for i in range(1):
        directory = os.fsencode(directory_snap)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                print (directory_snap+filename)
                img = cv2.imread(directory_snap+filename)
                #print (img.shape)
                if img.shape[0]>1000:
                    img = cv2.resize(img, (int(img.shape[1]*0.5),int(img.shape[0]*0.5)))
                    cv2.imshow('img', img)
                    cv2.imwrite(directory_in_str+filename, img)

