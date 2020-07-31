import tensorflow as tf
import cv2
import os
import posenet
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

# Configuring my cudnn for tensorflow 1
# Needed for my laptop, unsure for others
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def create_csv(file):
    csv_name = 'entire_data.csv'
    with tf.Session() as sess:
        # Setting ouput stride to 16
        output_stride = 16
        model_id = 101
        model_cfg, model_outputs = posenet.load_model(model_id, sess)

        # change to for loop after testing on single file
        #file = files[0]
        action = file.split("_")[-1].split(".")[0] # Name_Category/Label.mp4
        cap = cv2.VideoCapture(os.path.join(dir_, file))
        input_image, display_image, output_scale = posenet.read_cap(cap)

        frame_count = 0
        check = True
        while check:
            try:
                input_image, display_image, output_scale = posenet.read_cap(cap)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image})

                # print(heatmaps_result.shape) # (1, 33, 58, 17)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= output_scale

                # TODO this isn't particularly fast, use GL for drawing and display someday...
                overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.3)
                # setting min_pose_score and min_part_score =0 prints the estimated value

                x = keypoint_coords[0][:,0]
                y = keypoint_coords[0][:,1]
                data = {'action': action, 
                        'frame_number': frame_count+1, 
                        'input_number': np.arange(1, posenet.NUM_KEYPOINTS+1), 
                        'x_inputs': x, 'y_inputs': y}
                temp = pd.DataFrame(data=data)

                if csv_name not in os.listdir():
                    temp.to_csv(csv_name, 
                                header=True, 
                                index=False)
                else:
                    temp.to_csv(csv_name, 
                                mode='a', 
                                header=False, 
                                index=False)
            except:
                print(f"Completed Writing {file} data")
                check = False
                
        # print(cap.get(cv2.CAP_PROP_FPS)) -> frame rate

# Reading files from said directory
dir_ = ".\data"

files = os.listdir(dir_)

for file in files:
    print(f"Reading {file}")
    create_csv(file)

# Input csv
data = pd.read_csv("entire_data.csv")

# Output csv
csv_name = "cumulative.csv"

# Dropping all 0 values
df = data.drop(data[(data["x_inputs"] == 0) & (data["y_inputs"] == 0)].index)

# Creating column name
attach = ["_x", "_y"]
column_names = []

for i in posenet.PART_NAMES:
    for x in attach:
        column_names.append(i+x)

# Need to save column names first followed by the data
pd.DataFrame(columns=column_names+['action']).to_csv(csv_name, mode='a', index=False, header = True)

# Method to ouput x and y coordinates in single row
def combine_alternate(x_, y_):
    temp = []
    for i in range(len(posenet.PART_NAMES)):
        for j in [x_, y_]:
            temp.append(j[i])
    return temp

print(f"Creating {csv_name}")
for i in tqdm(range(int(len(df)/17))):
    x_ = df.iloc[i*17 : (i+1)*17]["x_inputs"].to_list()
    y_ = df.iloc[i*17 : (i+1)*17]["y_inputs"].to_list()
    
    temp = combine_alternate(x_, y_)
    action = df.iloc[i*17]["action"]
    
    temp_dict = dict(zip(column_names, temp))
    temp_dict['action'] = action
    
    pd.DataFrame.from_dict(temp_dict, orient='index').T.to_csv(csv_name, mode='a', index=False, header = False)


# Displaying details of the Cumulative.csv
df = pd.read_csv(f"{csv_name}")
print(f" ---------- Details of {csv_name} ---------- ")
print('Number of Classes = ', len(df['action'].value_counts()))
print('Names of Classes  = ', df['action'].value_counts().index.values)
print('\n Count of each class in the dataset :\n', df['action'].value_counts())
print('\nPercentage of each class in the dataset :\n', [i/len(df)*100 for i in df['action'].value_counts()])

# Stratified Split for RFC and ANN
sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# the for loop is for when we want to see the index of train and test from one or more splits
# Maybe more splits in future
for train_index, test_index in sss.split(X, y):
    print("ANN split\nTRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

##### Random Forest
# The following block of code takes significant amount of time, hence I've disabled it
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier(max_features=None, n_jobs=-1, verbose=1)
#params = {'criterion' : ['gini', 'entropy'],
#          'n_estimators' : [10, 100, 1000, 10000]}

#model = GridSearchCV(rfc, params).fit(X_train, y_train)
#print('Best estimator = ', model.best_estimator_,'\nBest score = ',model.best_score_,'\nBest Params =',model.best_params_)


# Preprocessing Label into OneHotEncoding
ohe = OneHotEncoder()
ohe.fit_transform(df['action'].values.reshape(-1,1))

# Saving the OneHotEncoding categories as text file for use in prediction
with open('OneHotEncoding categories.txt', 'w') as f:
    f.write(str(ohe.categories_[0]))

##### Artificial Neural Network
y_train = ohe.transform(df['action'][train_index].values.reshape(-1,1)).toarray()
y_test = ohe.transform(df['action'][test_index].values.reshape(-1,1)).toarray()

model_ANN = tf.keras.Sequential([tf.keras.layers.Dense(1024, activation="relu", name="Dense_1", 
                                                    input_dim=df.T.iloc[:-2, 1].values.reshape(-1, 1).shape[0]),
                                tf.keras.layers.BatchNormalization(name="Batch_Norm"),
                                tf.keras.layers.Dense(720, activation="relu", name="Dense_2"),
                                tf.keras.layers.Dropout(0.5),
                                tf.keras.layers.Dense(480, activation="relu", name="Dense_3"),
                                tf.keras.layers.Dense(360, activation="relu", name="Dense_4"),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Dense(180, activation="relu", name="Dense_5"),
                                tf.keras.layers.Dense(len(ohe.categories_[0]), activation="softmax", name="Dense_OP")])

model_ANN.compile(loss='categorical_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

print(f"Training ANN model \n {model_ANN.summary()}")

history_ANN = model_ANN.fit(X_train, y_train, 
                            validation_data=(X_test, y_test), 
                            batch_size=64, epochs=200, verbose=1)

model_ANN.save('ANN', save_format='h5')


##### Convolutional Neural Network

# Preprocessing data for CNN training
X_CNN = []
Y_CNN = []

# Since its 30 frames per second
seconds = 2
buffer_size = seconds*30 
shift_size = 30

print(f"Preparing data for CNN model considering {seconds} seconds")
# Number of Features = 34
for i in tqdm(range(0, (df.shape[0]-buffer_size), shift_size)):
    X_frame = np.zeros((buffer_size,34))
    for j in range(34):
        X_frame[:,j] = df.iloc[:, :-1].values[:,j] [i:i+buffer_size]
    label = stats.mode(df.iloc[:, -1].values[i:i+buffer_size])[0][0]
    X_CNN.append(X_frame)
    Y_CNN.append(label)

X_CNN = np.asarray(X_CNN).reshape(-1,buffer_size,34)
Y_CNN = np.asarray(Y_CNN)

X_CNN = X_CNN.reshape(X_CNN.shape[0], X_CNN.shape[1], X_CNN.shape[2], 1)

sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(X_CNN, Y_CNN):
    print("CNN split\nTRAIN shape : ", train_index.shape, "\nTEST shape : ", test_index.shape)

model_CNN = tf.keras.Sequential([tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation="relu", name="Conv_1",
                                                        input_shape=X_CNN[0].shape),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation="relu", name="Conv_2"),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(32, activation="relu", name="Dense_1"),
                                 tf.keras.layers.Dense(32, activation="relu", name="Dense_2"),
                                 tf.keras.layers.Dropout(0.25),
                                 tf.keras.layers.Dense(16, activation="relu", name="Dense_3"),
                                 tf.keras.layers.Dense(10, activation="relu", name="Dense_4"),
                                 tf.keras.layers.Dense(len(ohe.categories_[0]), activation="softmax", name="Dense_OP")])

model_CNN.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(f"Training CNN model \n {model_CNN.summary()}")

history_CNN = model_CNN.fit(X_CNN[train_index], ohe.transform(Y_CNN[train_index].reshape(-1,1)).toarray(),
                            validation_data=(X_CNN[test_index], ohe.transform(Y_CNN[test_index].reshape(-1,1)).toarray()), 
                            batch_size=64, epochs=400, verbose=1)

model_CNN.save('CNN', save_format='h5')


##### Recurrent Neural Network

# Necessary modifications for RNN
X_RNN = X_CNN.reshape(X_CNN.shape[0],
                      X_CNN.shape[1],
                      X_CNN.shape[2])

model_RNN = tf.keras.Sequential([tf.keras.layers.SimpleRNN(64, activation="relu", name="RNN_1",
                                                           input_shape=X_RNN.shape[1:]),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(64, activation="relu", name="Dense_1"),
                                 tf.keras.layers.Dense(32, activation="relu", name="Dense_2"),
                                 tf.keras.layers.Dense(32, activation="relu", name="Dense_3"),
                                 tf.keras.layers.Dense(32, activation="relu", name="Dense_4"),
                                 tf.keras.layers.Dense(10, activation="relu", name="Dense_5"),
                                 tf.keras.layers.Dense(len(ohe.categories_[0]), activation="softmax", name="Dense_OP")])

model_RNN.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(f"Training RNN model \n {model_RNN.summary()}")

history_RNN = model_RNN.fit(X_RNN[train_index], ohe.transform(Y_CNN[train_index].reshape(-1,1)).toarray(),
                            validation_data=(X_RNN[test_index], ohe.transform(Y_CNN[test_index].reshape(-1,1)).toarray()), 
                            batch_size=64, epochs=75, verbose=1)

model_RNN.save('RNN', save_format='h5')


##### Long Short Term Memory
model_LSTM = tf.keras.Sequential([tf.keras.layers.LSTM(64, activation="relu", name="LSTM_1",
                                                      input_shape=X_RNN.shape[1:]),
                                  tf.keras.layers.Dropout(0.5),
                                  tf.keras.layers.Dense(32, activation="relu", name="Dense_1"),
                                  tf.keras.layers.Dense(32, activation="relu", name="Dense_2"),
                                  tf.keras.layers.Dense(32, activation="relu", name="Dense_3"),
                                  tf.keras.layers.Dense(len(ohe.categories_[0]), activation="softmax", name="Dense_OP")])

model_LSTM.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(f"Training LSTM model \n {model_LSTM.summary()}")

history_LSTM = model_LSTM.fit(X_RNN[train_index], ohe.transform(Y_CNN[train_index].reshape(-1,1)).toarray(),
                              validation_data=(X_RNN[test_index], ohe.transform(Y_CNN[test_index].reshape(-1,1)).toarray()), 
                              batch_size=64, epochs=500, verbose=1)

model_LSTM.save('LSTM', save_format='h5')