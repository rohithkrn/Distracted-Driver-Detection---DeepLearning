import cv2
import glob

def process(filename, key):

    image = cv2.imread(filename)

    #print image.shape

    #r = 100.0 / image.shape[1]
    #dim = (100, int(image.shape[0] *r))

    imageresized = cv2.resize(image,(250,250))

    cv2.imwrite( 'imageresized2_{}.jpg'.format(key) ,imageresized )
    #print 'imageresized_{}.jpg'.format(key)
    
for (i,image_file) in enumerate(glob.iglob('E:/uminn_notes/ComputerVision/Project/c2_segmented/*.jpg')):
        process(image_file, i)
