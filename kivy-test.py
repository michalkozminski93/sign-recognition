from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

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

    def build(self):
        return LoginScreen()


if __name__ == '__main__':
    MyApp().run()