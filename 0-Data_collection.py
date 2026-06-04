import numpy as np
import os
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import time

# Set up the path for data collection and define the actions you want to train the model with.
DATA_PATH = os.path.join('MP_Data')                   
actions = np.array(['Hello', 'Thanks', 'I love you', 'World', 'Background'])  
no_sequences = 30                                     
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# ------------------------------------------ Set up mediapipe holistic -----------------------------------------------------
mp_holistic = mp.solutions.holistic         
mp_drawing = mp.solutions.drawing_utils      

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image.flags.writeable = False                   
    results = model.process(image)                  
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results

# Uncomment the first two lines to include face landmarks into the dataset.
# Personally, I found that this would not improve accuracy but would only increase computational costs
def draw_styled_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
    #                           mp_drawing.DrawingSpec(color= (80, 110, 10), thickness = 1, circle_radius = 1),   
    #                           mp_drawing.DrawingSpec(color= (80, 256, 10), thickness = 1, circle_radius = 1))                           
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))     
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (165, 255, 254), thickness = 1, circle_radius = 2),    
                              mp_drawing.DrawingSpec(color= (44, 167, 236), thickness = 1, circle_radius = 2))    

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        pose = (pose - pose[0]).flatten() 
    else: pose = np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    # Array containing all the pose landmarks for a specific frame. The same instruction for left hand (note, there is no visibility attribute for
    # hands) will give error if there is no left hand in the frame. That's why we create the zero array.
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

    # return np.concatenate([pose, face, left_hand, right_hand])                                                                                  
    return np.concatenate([pose, left_hand, right_hand])     

# Look through the actions and the frames this time, not through the webcam
cap = cv2.VideoCapture(0) 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    for action in actions:
        for sequence in range (no_sequences):
            # Loop through each video
            for frame_num in range(sequence_length):
                ret, frame = cap.read()              

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Print messages to report status
                if frame_num == 0:
                    cv2.putText(image, 'START COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)  # 2 second break
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                    
                cv2.imshow('OpenCV Feed', image)

                # Getting out of the loop
                if cv2.waitKey(10) & 0xFF == ord('q'):    # wait until a key is pressed and if the key is q, get out of the loop
                    break

# Once it breaks, release the webcame and close the frame
cap.release()
cv2.destroyAllWindows()