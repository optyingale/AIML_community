import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import cv2
import posenet
import pickle
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import time


def main():
    st.title("Fall Detection Using various Deep Learning models")

    # Selecting whether Live prediction or prediction on uploaded video
    options = ["Live Feed", "Upload a Video"]
    st.markdown(">")
    live_or_upload = st.radio("Something", options)

    video = None
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if live_or_upload == options[1]:
        video = st.file_uploader("Upload Video", type=["avi", "mp4"])

    # Getting ready with the file names
    filename_RFC = "RFC.pickle"
    filename_ANN = "ANN"
    filename_CNN = "CNN"
    # filename_RNN = "RNN"
    # filename_LSTM = "LSTM"

    # Using values from OneHotEncoding
    with open("OneHotEncoding categories.txt", 'r') as f:
        ohe_categories = f.readline()
    ohe_categories = list(ohe_categories.split("'")[1:-1])
    while " " in ohe_categories:
        ohe_categories.remove(" ")

    print(tf.__version__)

    # Setting up session and graph for tensorflow use
    global sess
    global graph

    # Setting up my PC for prediction
    # Below block can be removed if there's an issue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    graph = tf.get_default_graph()
    set_session(sess)

    model_RFC = pickle.load(open(filename_RFC, 'rb'))
    model_ANN = load_model(filename_ANN)
    model_CNN = load_model(filename_CNN)
    # model_RNN = load_model(filename_RNN)
    # model_LSTM = load_model(filename_LSTM)

    temporary = pd.DataFrame()

    # with tf.Session() as sess:
    with graph.as_default():

        set_session(sess)

        # Setting ouput stride to 16
        output_stride = 16
        model_id = 101
        model_cfg, model_outputs = posenet.load_model(model_id, sess)

        if video is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video)
        # cap = cv2.VideoCapture(0)
        input_image, display_image, output_scale = posenet.read_cap(cap)

        # file_save = "live"
        # result = cv2.VideoWriter(f'{file_save}.avi',
        #                     cv2.VideoWriter_fourcc(*'MJPG'),
        #                     10, (int(cap.get(3)), int(cap.get(4))))

        start = time.time()
        while True:
            input_image, display_image, output_scale = posenet.read_cap(cap)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image})

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

            x = keypoint_coords[0][:, 0]
            y = keypoint_coords[0][:, 1]

            data = []
            for i in range(len(posenet.PART_NAMES)):
                for j in [x, y]:
                    data.append(j[i])
            # print(temp)

            temp = pd.DataFrame(data).T

            prediction_RFC = model_RFC.predict(temp)
            prediction_ANN = ohe_categories[np.argmax(model_ANN.predict(temp))]
            prediction_CNN = "Not Ready yet"

            temporary = pd.concat([temporary, temp], ignore_index=True)
            X_CNN = []
            if len(temporary) == 4 * 30:
                X_CNN.append(temporary.values)
                X_CNN = np.asarray(X_CNN).reshape(-1, 4 * 30, 34, 1)
                prediction_CNN = ohe_categories[np.argmax(model_CNN.predict(X_CNN))]
                temporary.drop(0, inplace=True)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            color = (255, 150, 0)  # BGR
            thickness = 1

            coords_RFC = (0, 20)
            coords_ANN = (0, 50)
            coords_CNN = (0, 85)

            cv2.putText(overlay_image, 'RFC = ' + prediction_RFC[0], coords_RFC, font, fontscale, color, thickness)
            cv2.putText(overlay_image, 'ANN = ' + prediction_ANN, coords_ANN, font, fontscale, color, thickness)
            cv2.putText(overlay_image, 'CNN = ' + prediction_CNN, coords_CNN, font, fontscale, color, thickness)

            cv2.imshow('posenet', overlay_image)

            # result.write(overlay_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                # result.release()
                break

    # print(cap.get(cv2.CAP_PROP_FPS)) -> frame rate
    print(f'time taken = {(time.time() - start) / 60} mins')


if __name__ == "__main__":
    main()