import pickle 
import cv2
import dlib
from main import Train

f = open('pos.pickle','rb')
data = pickle.load(f)
f.close()

train = Train(data)
images,annotations = train._prepare_image_and_annotations()
detector = train.trainer(images,annotations)
detector.save("detector.svm")
