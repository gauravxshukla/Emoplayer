from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from keras.models import load_model
from time import sleep
from keras.preprocessing import image
import cvlib as cv


emo_classifier = load_model(os.path.join(settings.BASE_DIR,'face_detector/Emotion_little_vgg.h5'))
gender_model = load_model(os.path.join(settings.BASE_DIR,'face_detector/gender_detection.model'))
face_classifier = cv2.CascadeClassifier(r'C:\Python\Python37\Project\haarcascade_frontalface_default.xml')
#changes
faceProto =os.path.sep.join([settings.BASE_DIR, "face_detector/opencv_face_detector.pbtxt"])
faceModel = os.path.sep.join([settings.BASE_DIR, "face_detector/opencv_face_detector_uint8.pb"])
ageProto = os.path.sep.join([settings.BASE_DIR, "face_detector/age_deploy.prototxt"])
ageModel = os.path.sep.join([settings.BASE_DIR, "face_detector/age_net.caffemodel"])
genderProto = os.path.sep.join([settings.BASE_DIR, "face_detector/gender_deploy.prototxt"])
genderModel = os.path.sep.join([settings.BASE_DIR, "face_detector/gender_net.caffemodel"])


genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
padding = 20


        
class EmoDetect(object):

    def __init__(self):
    		self.vs =cv2.VideoCapture(0)
    def __del__(self):
    		cv2.destroyAllWindows()
    def getFaceBox(self,net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

    def get_frame(self):
        class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
        classes = ['Man','Woman']
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']
        
        jarurat=[]
        # Grab a single frame of video
        ret, frame = self.vs.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        face, confidence = cv.detect_face(frame)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)


            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = emo_classifier.predict(roi)[0]
                
                #print(preds)
                label=class_labels[preds.argmax()]
                jarurat.append(label)
                #print(label)
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        #cv2.imshow('Emotion Detector',frame)
        # loop through detected faces
        frameFace, bboxes = self.getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            #continue
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            #print("Age Output : {}".format(agePreds))
            #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
            jarurat.append(gender)
            jarurat.append(age)
            label = "{},{}".format(gender, age)
            cv2.putText(frame, label, (bbox[0], bbox[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
        
       
       
        

        # display output
        #cv2.imshow("emotion and gender detection", frame)
        #break;
        
    
        print(jarurat)
        ret, jpeg = cv2.imencode('.jpg', frame)
        jarurat2=[]
        jarurat2.append(jpeg.tobytes())
        jarurat2.append(jarurat)
        #print(jarurat2)
        return jarurat2





