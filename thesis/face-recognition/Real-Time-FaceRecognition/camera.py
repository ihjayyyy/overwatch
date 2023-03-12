import os
# uncomment this line if you want to run your tensorflow model on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import face_recognition
import tensorflow as tf
import pickle
import time
import cv2
import os
from datetime import datetime, date
import numpy as np

date = date.today()
date = date.strftime("%B %d, %Y")

AttedanceSheetsFolder = "C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/Attendance-Sheets-Folder"
if not os.path.exists(AttedanceSheetsFolder):
    os.makedirs(AttedanceSheetsFolder)  # create folder if doesn't exist

AttendessPicturesFolder = "C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/AttendeesPictures"
if not os.path.exists(AttendessPicturesFolder):
    os.makedirs(AttendessPicturesFolder)  # create folder if doesn't exist

SpoofingPicturesFolder = "C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/SpoofingPictures"
if not os.path.exists(SpoofingPicturesFolder):
    os.makedirs(SpoofingPicturesFolder)  # create folder if doesn't exist

SPicturesFolder = "C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/SpoofingPictures/"+date
if not os.path.exists(SPicturesFolder):
    os.makedirs(SPicturesFolder)  # create folder if doesn't exist   

PicturesFolder = "C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/AttendeesPictures/"+date
if not os.path.exists(PicturesFolder):
    os.makedirs(PicturesFolder)  # create folder if doesn't exist
    
liveness_model = tf.keras.models.load_model("liveness.model")
label_encoder = 'label_encoder.pickle'
le = pickle.loads(open(label_encoder, "rb").read())

sheetToday = 'C:/Users/Ejay Olivar/Desktop/facial-recognition-main/Real-Time-FaceRecognition/Attendance-Sheets-Folder/'+date+'.csv'

def markAttendance(name):
    if not os.path.exists(sheetToday):
        with open(sheetToday,'w+') as f:
            f.writelines('Name,Time')
    with open(sheetToday,'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now =datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}') 


#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

class Video(object):
    def __init__(self) -> None:
        self.video=cv2.VideoCapture(1)
    def __del__(self) -> None:
        self.video.release()

    def get_frame(self) -> None:
        ret,frame = self.video.read()

        #Initialize 'currentname' to trigger only when a new person is identified.
        currentname = "unknown"

        # Detect the fce boxes
        boxes = face_recognition.face_locations(frame)
            
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

            # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding,tolerance=0.5)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                #If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
                    
            face = frame[top:bottom,left:right]
            saveface=face
        try:
            face = cv2.resize(face, (32,32))
        except:
            ret,jpg=cv2.imencode('.jpg',frame)
            return jpg.tobytes()

        face = face.astype('float') / 255.0 
        face = tf.keras.preprocessing.image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        preds = liveness_model.predict(face)[0]
        j = np.argmax(preds)
        label = le.classes_[j] # get label of predicted class
        text = name +' - '+ label
        label1 = f'{label}: {preds[j]:.4f}'
        print(label1)

        if (name != 'Unknown' and label=='real'):
			# draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				.8, (0, 255, 0), 2)
            markAttendance(name)
            cv2.imwrite(PicturesFolder+"/"+name+'.jpg',saveface) 
                        
        elif(name == "Unknown" or label == 'fake'):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 0, 255), 2)
            y = top - 15 if top - 15 > 15 else top + 15          
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				.8, (0, 0, 255), 2)
            cv2.imwrite(SPicturesFolder+'/'+name+'.jpg',frame)

        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()