from cv2 import threshold
from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
# give path of the image
imagePath = ""
# give path of the vide input
# videoPath = "video (2).mp4"
#videoPath = 0
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadmodel(modelURL)
detector.loadModel()
# passes image as input
detector.predictImage(imagePath, threshold)
# passes video as input
# detector.predictVideo(videoPath, threshold)
