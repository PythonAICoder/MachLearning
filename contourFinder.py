import cv2
import numpy as np
from glob import glob
import imutils
from imutils import paths

class ContourFinder():
    def __init__(self):
        self.imgNames_test =glob('./x_test/*.jpeg')
        self.x_test_old = []
        self.x_test = []
        self.y_test = []
        self.num_shapes = 0

    def find_contours(self, min_area = 4600):
        for imgName in self.imgNames_test:
            self.y_test.append(int(imgName[9]))
            self.y_test.append(int(imgName[10])) 
            self.y_test.append(int(imgName[11]))
            rimg = cv2.imread(imgName)
            rot_img = cv2.rotate(rimg,cv2.ROTATE_90_COUNTERCLOCKWISE)  
            self.x_test_old.append(cv2.resize(rot_img, (0,0),fx=0.2,fy=0.2))
        self.x_test_old = np.asarray(self.x_test_old)
        self.y_test = np.asarray(self.y_test)
        print(self.x_test_old.shape, self.y_test)


        cv2.imshow('img window', self.x_test_old[0])
        cv2.waitKey()

        for num_pic in self.x_test_old:
            print(num_pic)
            num_pic = cv2.cvtColor(num_pic, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((5,5),dtype=np.uint8)
            num_pic = cv2.GaussianBlur(num_pic, (5, 5),0) 
            num_pic = cv2.erode(num_pic, kernel, iterations=3) 
            num_pic = cv2.dilate(num_pic, kernel, iterations=3) ###try denoising the images try using it through dilation
            num_pic = cv2.fastNlMeansDenoising(num_pic,num_pic)
            imgth = cv2.adaptiveThreshold(num_pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 199, 5)

            contours, hierarchy = cv2.findContours(imgth.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            c = []
            area = []
            for ctr in contours:
                if cv2.contourArea(ctr) > min_area:
                    area.append(cv2.contourArea(ctr))
                    print(cv2.contourArea(ctr))
                    c.append(cv2.boundingRect(ctr))
            print('max:', max(area))
            for rect in c:
                #print(rect)
                cv2.rectangle(num_pic, (rect[0]-20,rect[1]-20),(rect[0]+rect[2]+20, rect[1]+rect[3]+20), (0,255,0),3)
                roi = imgth[rect[1]-20:rect[1]+rect[3]+20,rect[0]-20:rect[0]+rect[2]+20]
                try:
                    roi = cv2.resize(roi, (28,28))
                    self.x_test.append(roi)
                except Exception as e:
                    pass
                #print(roi)
                # try:
                #     roi = cv2.resize(roi, (28,28))
                #     print(roi)
                #     self.x_test.append(roi)
                # except Exception:
                #     pass
        for x in self.x_test:
            cv2.imshow('img window', x)
            cv2.waitKey()
        self.x_test = np.asarray(self.x_test)
        return self.x_test, self.y_test

# cf = ContourFinder()
# a,b = cf.find_contours()