import pickle 
import cv2
import dlib
f = open('pos.pickle','rb')
data = pickle.load(f)
f.close()
img = dlib.load_rgb_image("D:\\hog_object_detect\\ku.jpg")
images =[]
annots =[]
for d in data:
    p_a = d[1:]
    annots.append([dlib.rectangle(left=p_a[0],top=p_a[1],right=p_a[2],bottom=p_a[3])])
    t_img = d[0]
    t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
    images.append(t_img)


options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True

options.C = 5
options.num_threads = 4
options.be_verbose = True

detector = dlib.train_simple_object_detector(images, annots, options)
detector.save("det.svm")
boxes = detector(img)
box= boxes[0]
(x,y,xb,yb) = [box.left(),box.top(),box.right(),box.bottom()]
    
cv_image = cv2.imread("D:\\hog_object_detect\\ku.jpg")

cv_image = cv2.rectangle(cv_image, (x,y), (xb,yb), (255,0,0), 2)
cv2.imshow("hola_image",cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()