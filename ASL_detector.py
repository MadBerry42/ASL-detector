# In order to run this code:
#  activate the virtual environment with venv\Script\activate
#  write python ASL_detector.py in the terminal 
import numpy as np
import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#------------------------------------------------------ Initialize functions ---------------------------------------------------
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['Hello', 'Thanks', 'I love you', 'World', 'Background'])
no_sequences = 30                                     
sequence_length = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB. Necessary for mediapipe to do a detection
    image.flags.writeable = False                   # Set it to unwritable to save memory
    results = model.process(image)                  # Process the image (perform detection) using mediapipe
    image.flags.writeable = True                    # Set it back to writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    return image, results

def draw_styled_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
    #                           mp_drawing.DrawingSpec(color= (80, 110, 10), thickness = 1, circle_radius = 1),    # Landmarks specifications
    #                           mp_drawing.DrawingSpec(color= (80, 256, 10), thickness = 1, circle_radius = 1))    # Connections specifications                        
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    # Landmarks specifications
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    # Landmarks specifications
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))    # Connections specifications
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    # Landmarks specifications
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))    # Connections specifications
    
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        pose = (pose - pose[0]).flatten() 
    else: pose = np.zeros(33 * 4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    
    if results.left_hand_landmarks:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        left_hand = (left_hand - left_hand[0]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        right_hand = (right_hand - right_hand[0]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
                                                                                  
    return np.concatenate([pose, left_hand, right_hand])

#-------------------------------------------------------- Create the empty model -------------------------------------------------------
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)                               
        sequences.append(window)                           
        labels.append(label_map[action])

X = np.array(sequences)
n_keypoints = X.shape[2]

model = Sequential()        
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, n_keypoints)))
# Syntax: model.add(LSTM(number of LSTM units, 
#                   return_sequences which is necessaty with tensorflow because it allows to stack sequences and the next layer will need 
#                   activation type
#                   input_shape(number of frames, number of keypoints)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) # In this case don't return the sequence because the next layer is dense
model.add(Dense(64, activation='relu'))                        # 64 units of densely connected neural network neurons
model.add(Dense(128, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) 
                
res = []
sequence = []                               # Collects out thirty frames in order to generate a prediction
sentence = []                               # History of detections, so they can be concatenated
predictions = []
threshold = 0.87                             # Confidence (if the highest probability > 40%, assune it's right)

# ------------------------------------------------------ Load model weights -----------------------------------------------------------
model.load_weights('action.h5')

#--------------------------------------------------------- ASL detection --------------------------------------------------------------
cap = cv2.VideoCapture(0)                    
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    while cap.isOpened():                         
        ret, frame = cap.read()                   

        image, results = mediapipe_detection(frame, holistic)
        print(results)

        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)                           
        sequence.append(keypoints)
        sequence = sequence[-30:]                                       

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]    
                                                                        
            predictions.append(np.argmax(res))
            predictions = predictions[-10:]

            if res[np.argmax(res)] > threshold and len(np.unique(predictions[-10:])) == 1:
                
                if len(sentence) > 0:
                    # Check that the word is not the same as the previous one
                    if actions[np.argmax(res)] != sentence[-1] and actions[np.argmax(res)] != "Background":             
                        sentence.append(actions[np.argmax(res)])            
                    elif actions[np.argmax(res)] == "Background":
                            sentence.append(" ")
                elif actions[np.argmax(res)] != "Background":
                    sentence.append(actions[np.argmax(res)])                
                elif actions[np.argmax(res)] == "Background":
                        sentence.append(" ")

            if len(sentence) > 2:
                sentence = sentence[-2:]                                     

        cv2.putText(image, ' '.join(sentence), (230, 450),                 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
        cv2.imshow('OpenCV Feed', image)         

        if cv2.waitKey(10) & 0xFF == ord('q'):  
            break

cap.release()
cv2.destroyAllWindows()
