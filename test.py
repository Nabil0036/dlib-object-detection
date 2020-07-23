import pickle 
import dlib
import cv2

test_path = "D:\\hog_object_detect\\ku.jpg"
img = cv2.imread(test_path)
detector  = dlib.simple_object_detector("detector.svm")
boxes = detector(img)
box= boxes[0]
(x,y,xb,yb) = [box.left(),box.top(),box.right(),box.bottom()]
cv_image = cv2.imread("D:\\hog_object_detect\\ku.jpg")
cv_image = cv2.rectangle(cv_image, (x,y), (xb,yb), (255,0,0), 2)
cv2.imshow("hola_image",cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()