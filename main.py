import dlib
import cv2
import pickle

class Train:
    def __init__(self,pickle_data):
        self.pickle_data = pickle_data

    def _prepare_image_and_annotations(self):
        self.images = []
        self.annots = []
        for data in self.pickle_data:
            annot = data[1:]
            self.image = data[0]
            self.image = cv2.cvtColor(self.image , cv2.COLOR_BGR2RGB)
            self.annots.append([dlib.rectangle(left=annot[0],top=annot[1],right=annot[2],bottom=annot[3])])
            self.images.append(self.image)
        return self.images, self.annots
    
    def trainer(self,imgs,ants):
        self.images = imgs
        self.annots = ants
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.C = 5
        options.num_threads = 4
        options.be_verbose = True
        detector = dlib.train_simple_object_detector(self.images, self.annots, options)

        return detector
    