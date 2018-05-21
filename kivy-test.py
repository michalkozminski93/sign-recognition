# coding:utf-8
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.slider import Slider
from kivy.uix.button import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.uix.behaviors import ToggleButtonBehavior
import cv2
import os
import numpy as np
from random import shuffle

class ROI:
    def __init__(self, x, y, w, h, mask, shape):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mask = mask
        
        self.roiSize = w*h      
        self.shape = shape
        self.prediction = -1

    def showSign(self, image):
        cv2.putText(image, signDesc[str(self.prediction)],(self.x + self.w, self.y + 20), cv2.FONT_ITALIC, 1.5, (0,255,0))
        #cv2.circle(image, (self.x + int(self.w/2), self.y + int(self.h/2)), int(self.w/2), (0,255,0), 2)
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0,255,0), 2)

    def enhanceRoi(self,factor):
        if (self.x-factor/2*self.w)>0 and (self.y-factor/2*self.h)>0: 
            self.x-=int(factor/2*self.w)
            self.y-=int(factor/2*self.h)
            self.w =int(self.w*(1+factor))
            self.h =int(self.h*(1+factor))


#path = "snapshots/snap-00002.png"
path = "C:/Users/Michau/Desktop/GermanTrafficSigns/FullIJCNN2013/frames/00001.ppm"
#path = 'C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/snapshots/snap-00001.png'

class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

class MyLayout(GridLayout):
    hMin = ObjectProperty()
    hMax = ObjectProperty()
    sMin = ObjectProperty()
    sMax = ObjectProperty()
    vMin = ObjectProperty()
    vMax = ObjectProperty()
    img1 = ObjectProperty() 
    img2 = ObjectProperty()
    redMask = ObjectProperty()
    yellowMask = ObjectProperty()
    blueMask = ObjectProperty()
    camera = cv2.VideoCapture('day1.wmv')
    directory_in_str = "C:/Users/Michau/Desktop/validation/"
    directory_in_str = "C:/Users/Michau/Desktop/new_none/"
    i = 1
    filenames = []
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".ppm"): 
            filenames.append(os.fsdecode(directory+file))
    #shuffle(filenames)

    def nextImage(self):
        self.i+=1
        self.updateMask()

    def previousImage(self):
        self.i-=1
        self.updateMask()

    def saveValues(self):
        if (self.redMask.state == "down"):
            filename = 'mask_red.txt'
        elif (self.yellowMask.state == "down"):
            filename = 'mask_yellow.txt'
        elif (self.blueMask.state == "down"):
            filename = 'mask_blue.txt'
        else:
            filename = ''
        try :
            file = open(filename, 'w')
            text = '{:d} {:d} {:d}\n'.format(int(self.hMin.value), int(self.sMin.value), int(self.vMin.value))
            file.write(text)
            text = '{:d} {:d} {:d}'.format(int(self.hMax.value), int(self.sMax.value), int(self.vMax.value))
            file.write(text)
            file.close()
        except IOError:
            print ("Choose mask type!")

    def loadValues(self):
        if (self.redMask.state == "down"):
            filename = 'mask_red.txt'
        elif (self.yellowMask.state == "down"):
            filename = 'mask_yellow.txt'
        elif (self.blueMask.state == "down"):
            filename = 'mask_blue.txt'
        else:
            filename = ''
        try :
            file = open(filename, 'r')
            text = file.readline()
            values = text.strip().split()
            self.hMin.value = int(values[0])
            self.sMin.value = int(values[1])
            self.vMin.value = int(values[2])
            text = file.readline()
            values = text.strip().split()
            self.hMax.value = int(values[0])
            self.sMax.value = int(values[1])
            self.vMax.value = int(values[2])
            file.close()
        except IOError:
            print ("No such file.")

    def updateMask(self):
        img = cv2.imread(self.filenames[self.i])
        #img = cv2.imread("C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/"+str(self.i)+".jpg")
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print (img.shape[0], img.shape[1])
        minDim = min(img.shape[0], img.shape[1])
        maxDim = max(img.shape[0], img.shape[1])
        #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=self.hMin.value, param1=self.sMin.value, param2=self.sMax.value, minRadius=int(self.vMin.value), maxRadius=int(self.vMax.value))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=int(maxDim*0.25), param1=int(self.hMin.value), param2=int(self.hMax.value), minRadius=int(minDim*0.2), maxRadius=int(minDim*0.55))
        if circles is not None:
            for circle in circles[0]:
                cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0), 2)

        cv2.imwrite("mask.png", img)
        cv2.imwrite("frame.png", img_copy)
        self.img1.source = "frame.png"
        self.img2.source = "mask.png"
        self.img1.reload()
        self.img2.reload()

    def updateMask33(self):
        img = cv2.imread(self.filenames[self.i])
        #img = cv2.imread("C:/Users/Michau/Dropbox/Studia/MGR/PRACA MGR/SignRecognition/sign-recognition/canny.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (7,7), sigmaX = 3 ,sigmaY = 3)
        canny = cv2.Canny(gauss, self.hMin.value, self.hMax.value)
        picture, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for i in contours:
            arcLength = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*arcLength, True)
            
            if ((len(approx)) < 14 and (len(approx)) > 0): 
                cv2.drawContours(img, i, -1, (255,0,0), 2)
                print (len(approx))
                cv2.waitKey(0)

        cv2.imwrite("mask.png", img)
        cv2.imwrite("frame.png", canny)
        self.img1.source = "frame.png"
        self.img2.source = "mask.png"
        self.img1.reload()
        self.img2.reload()

    def updateMask3(self):
        
        ret, frame = self.camera.read()
        frame = cv2.imread(self.filenames[self.i])
        frame_size = frame.shape[0]*frame.shape[1]
        #path = frame
        #self.img1.source = path
        #self.img2.source = path
        #self.hMin, self.sMin, self.vMin = self.hsvMin[0].value, self.hsvMin[1].value, self.hsvMin[2].value 
        #self.hMax, self.sMax, self.vMax = self.hsvMax[0].value, self.hsvMax[1].value, self.hsvMax[2].value 
        #print(self.hMin, self.sMin, self.vMin)
        #print(self.hMax, self.sMax, self.vMax)
        
        #frame = cv2.imread(path)
        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)
    
        hsv = cv2.cvtColor(cv2.bitwise_not(frame), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.hMin.value, self.sMin.value, self.vMin.value), (self.hMax.value, self.sMax.value, self.vMax.value))
        mask = cv2.medianBlur(mask, 5)
        #mask = cv2.dilate(mask,kernel5,iterations=1)
        #mask = cv2.erode(mask,kernel3,iterations=1)
        #mask = cv2.erode(mask,kernel,iterations=1)
        picture, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #check cv2.RETR_EXTERNAL
        index = 1
        imgField = mask.shape[0]*mask.shape[1]

        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            newRoi = ROI(x,y,w,h,'x',"unknown")
            if ((w*h>4e-4*frame_size) and (w*h<1e-2*frame_size) and (w/h>0.5) and ((w/h)<1.5)):
                #cv2.rectangle(mask, (newRoi.x, newRoi.y), (newRoi.x+newRoi.w, newRoi.y+newRoi.h), (255,255,255), 1)
                newRoi.enhanceRoi(0.2)
                cv2.rectangle(mask, (newRoi.x, newRoi.y), (newRoi.x+newRoi.w, newRoi.y+newRoi.h), (255,255,255), 1)
                #cv2.putText(mask, str(index),(x+w,y+20), cv2.FONT_ITALIC, 1.5, (255,255,255))
                index += 1

        cv2.imwrite("mask.png", mask)
        cv2.imwrite("frame.png", frame)
        self.img1.source = "frame.png"
        self.img2.source = "mask.png"
        self.img1.reload()
        self.img2.reload()

class MyApp(App):
    path = ""
    def build(self):
        print (self.path)
        return MyLayout()

if __name__ == '__main__':
    flApp = MyApp()
    flApp.run()