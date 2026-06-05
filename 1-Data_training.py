from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os 
from keras.models import Sequential       
from keras.layers import LSTM, Dense, Input                                               
from keras.callbacks import TensorBoard

DATA_PATH = os.path.join('MP_Data')  
actions = os.listdir(DATA_PATH)
label_map = {label:num for num, label in enumerate(actions)}
print(f"Actions list: {actions}")

i = 0
no_sequences = np.zeros((len(actions)))
sequence_length = np.zeros((len(actions)))
for action in actions:
    no_sequences[i] = int(len(os.listdir(f"./MP_Data/{action}")))
    sequence_length[i] = int(len(os.listdir(f"./MP_Data/{action}/0")))
    i+=1
print(f"Actions have {no_sequences} sequences with {sequence_length} frames each")

i = 0
sequences, labels = [], []
for action in actions:
    for sequence in range(int(no_sequences[i])):
        window = []
        for frame_num in range(int(sequence_length[i])):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)                               # List of landmarks position for one single video
        sequences.append(window)                             # List of landmark positions for each sign in all videos
        labels.append(label_map[action])
    i+=1

# --------------------------------- Create train and test set ----------------------------------------
y = np.eye(len(np.unique(labels)))[np.array(labels).astype(int)]
X = np.array(sequences)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)
# Save variables
if os.path.exists("Data") == False:
    os.mkdir("Data")   
np.save(f'Data/X_train.npy', X_train)
np.save(f'Data/X_test.npy', X_test)
np.save(f'Data/y_train.npy', y_train)
np.save(f'Data/y_test.npy', y_test)

# --------------------------------- Create the neural network ----------------------------------------
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
n_keypoints = X.shape[2]

model = Sequential()        
model.add(Input(shape=(None, n_keypoints)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) # In this case don't return the sequence because the next layer is dense
model.add(Dense(64, activation='relu'))                        # 64 units of densely connected neural network neurons
model.add(Dense(128, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))       

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs= 10)# , callbacks=[tb_callback])
model.save('Data/action.h5')

y_predicted = model.predict(X_test)
np.save('Data/y_predicted.npy', y_predicted)
                                                            
                                                            
                                                            
                                                               




