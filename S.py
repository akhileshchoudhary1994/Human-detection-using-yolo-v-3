import cv2
import numpy as np

config_file="C:/yolov3-320.cfg"
frozen_model="C:/yolov3.weights"
model=cv2.dnn.readNetFromDarknet(config_file,frozen_model)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classNames=['person',
'bicycle',
'car',
'motorbike',
'aeroplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'sofa',
'pottedplant',
'bed',
'diningtable',
'toilet',
'tvmonitor',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush']
#img=cv2.imread("C:/Users/Akhilesh/Downloads/group-people.jpg")
cap=cv2.VideoCapture(0)
confThreshold=0.5
nmsThreshold=0.3
wht=320
cap.set(3,648)
cap.set(4,488)
#model.setInputSize(320,320)
#model.setInputScale(1.0/127.5)
#model.setInputMean((127.5,127.5,127.5))
#model.setInputSwapRB(True)
def findObjects(outputs,img):
	ht,wt,ct=img.shape
	bbox=[]
	classIds=[]
	confs=[]
	for output in outputs:
		for det in output:
			scores=det[5:]
			classId=np.argmax(scores)
			confidence=scores[classId]
			if confidence > confThreshold:
				w,h=int(det[2]*wt),int(det[3]*ht)
				x,y=int((det[0]*wt) - w/2),int((det[1]*ht)-h/2)
				bbox.append([x,y,w,h])
				classIds.append(classId)
				confs.append(float(confidence))
	print(len(bbox))
	indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
	for i in indices:
				i=i[0]
				box=bbox[i]
				x,y,w,h=box[0],box[1],box[2],box[3]
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
				cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
while True:
	success,img=cap.read()
	blob=cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
	model.setInput(blob)
	layerNames=model.getLayerNames()
	print(layerNames)
	outputNames=[layerNames[i[0]-1] for i in model.getUnconnectedOutLayers()]
	print(model.getUnconnectedOutLayers())
	outputs=model.forward(outputNames)
	findObjects(outputs,img)
	
	
	
			
	cv2.imshow("Output",img)
	cv2.waitKey(1)
