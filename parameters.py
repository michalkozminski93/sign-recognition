import cv2

class color_mask:
    '''
    Color mask defines a convolution matrix that sweeps through an image to find areas with specific colors that road signs are using
    '''
    def __init__(self, thresh1, thresh2, color):
        self.threshMin = thresh1
        self.threshMax = thresh2
        self.type = color

    def find_color_mask(self, img):
        hsv = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.threshMin, self.threshMax)
        mask = cv2.medianBlur(mask, 3)
        return mask

#TODO: Move parameters to ini file

# Folder directories
dir_snapshots = "C:/signs-recognition-files/full/images/"
dir_validation_img = "C:/signs-recognition-files/cropped/validation/"
train_test_dir = "C:/signs-recognition-files/cropped/train_test/"
val_path = "C:/signs-recognition-files/validation/"

# Road signs dictionary
signs_dict = {
              0:'20', 
              1:'30', 
              2:'40', 
              3:'50', 
              4:'60', 
              5:'70', 
              6:'80', 
              7:'90', 
              8:'100', 
              9:'110', 
              10:'120', 
              11:'Zakaz wyprz.', 
              12:'Ustap piersz.',
              13:'Stop',
              14:'Zakaz wjazdu', 
              15:'Zakaz ruchu', 
              16:'Zakaz w lewo', 
              17:'Zakaz w prawo', 
              18:'Niebezp. lewo', 
              19:'Niebezp. prawo', 
              20:'Zwezenie',
              21:'Nakaz w lewo', 
              22:'Nakaz w prawo', 
              23:'Nakaz prosto', 
              24:'Z prawej', 
              25:'Rondo', 
              26:'Z prawej'
}

# Image parameters for CNN 
img_size_cnn = 32
frame_size = 32
num_channels_cnn = 1
prediction_treshold = 0.75


# Color masks for image segmentation
red_mask = color_mask((70,9,130), (97,255,255), "red")
yellow_mask = color_mask((100,24,140), (110,255,255), "yellow")
blue_mask = color_mask((18,30,193), (22,255,255), "blue")