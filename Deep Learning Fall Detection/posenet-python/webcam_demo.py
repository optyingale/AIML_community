import tensorflow as tf
import cv2
import time
import argparse

import posenet

# Added below 3 lines as cudnn was not getting initialized
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        # print(posenet.PART_CHANNELS) # given below
        # #24 points['left_face', 'right_face', 'right_upper_leg_front', 'right_lower_leg_back', 'right_upper_leg_back',
        # 'left_lower_leg_front', 'left_upper_leg_front', 'left_upper_leg_back', 'left_lower_leg_back', 'right_feet',
        # 'right_lower_leg_front', 'left_feet', 'torso_front', 'torso_back', 'right_upper_arm_front',
        # 'right_upper_arm_back', 'right_lower_arm_back', 'left_lower_arm_front', 'left_upper_arm_front',
        # 'left_upper_arm_back', 'left_lower_arm_back', 'right_hand', 'right_lower_arm_front', 'left_hand']
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

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

            # print(cap.get(cv2.CAP_PROP_FPS))
            # zipped = list(zip(posenet.PART_CHANNELS, keypoint_coords[0]))
            # print(zipped)
            # print(keypoint_coords[0])

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # print(cap.get(cv2.CAP_PROP_FPS)) -> frame rate
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
