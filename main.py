"""
main file for cropping mouth from video frames.
"""

from __future__ import absolute_import, division, print_function

import os
import cv2

import torch
import argparse

import numpy as np
import face_alignment
from tqdm import tqdm
from glob import glob
from skimage import io

from cropper.eye_cropper import crop_eye_image
from cropper.mouth_cropper import crop_mouth_image


def parse_args():
    parser = argparse.ArgumentParser(description='extract mouth from videos')
    parser.add_argument('--videos_path', type=str,
                        default='./test', help='the input videos folder')
    args = parser.parse_args()
    return args

class Cropper:
    def __init__(self) -> None:
        self.eye_width = 60
        self.eye_height = 48
        self.face_roi_width = 300
        self.face_roi_height = 300
        self.mouth_width = 48
        self.mouth_height = 48
        self.start_idx = 48
        self.stop_idx = 68
        self.window_margin = 12

        self.mean_face = 'D:/Glasgow/RVTALL/EyeLipCropper/cropper/20words_mean_face.npy'

        self.args = parse_args()

    def crop_files(self):

        video_list = glob(self.args.videos_path+'/*.avi')

        for idx, video_name in enumerate(video_list):
            print('Processing video {}/{}.'.format(idx+1, len(video_list)))
            self.crop_one_video(video_name=video_name)

    def crop_one_video(self, video_name):
        os.makedirs(video_name[0:-4], exist_ok=True)

        tgt_folder = video_name[0:-4]
        imgs_folder = tgt_folder + '/images'
        landmarks_folder = tgt_folder + '/landmarkers'
        boxes_folder = tgt_folder + '/boxes'
        logs_folder = tgt_folder + '/logs'
        left_eyes_folder = tgt_folder + '/left_eyes'
        right_eyes_folder = tgt_folder + '/right_eyes'
        mouths_folder = tgt_folder + '/mouths'

        os.makedirs(imgs_folder, exist_ok=True)
        reader = cv2.VideoCapture(video_name)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(os.path.join(imgs_folder,
                        '{:04d}.png'.format(frame_num)), image)
            frame_num += 1
        reader.release()

        os.makedirs(landmarks_folder, exist_ok=True)
        os.makedirs(boxes_folder, exist_ok=True)
        fan = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False)
        preds = fan.get_landmarks_from_directory(
            imgs_folder, return_bboxes=True)
        for image_file, (landmark, _, box) in preds.items():
            if not box:
                os.makedirs(logs_folder, exist_ok=True)
                with open(os.path.join(logs_folder, 'log.txt'), 'a') as logger:
                    logger.write(os.path.abspath(image_file) + '\n')
                continue
            landmark = np.array(landmark)[0]
            box = np.array(box)[0, :4]
            npy_file_name = os.path.splitext(
                os.path.basename(image_file))[0] + '.npy'
            image_landmark_path = os.path.join(landmarks_folder, npy_file_name)
            image_box_path = os.path.join(boxes_folder, npy_file_name)
            np.save(image_landmark_path, landmark)
            np.save(image_box_path, box)

            # crop eyes
        print('\033[36mCropping eye images ...\033[0m')
        os.makedirs(left_eyes_folder, exist_ok=True)
        os.makedirs(right_eyes_folder, exist_ok=True)
        for box_file in tqdm(sorted(os.listdir(boxes_folder))):
            box_path = os.path.join(boxes_folder, box_file)
            landmarks_path = os.path.join(landmarks_folder, box_file)
            box_file = os.path.splitext(box_file)[0] + '.png'
            image_path = os.path.join(imgs_folder, box_file)
            left_eye_img, right_eye_img, _, _ = crop_eye_image(np.load(landmarks_path),
                                                            np.load(box_path),
                                                            image_path,
                                                            eye_width=self.eye_width,
                                                            eye_height=self.eye_height,
                                                            face_width=self.face_roi_width,
                                                            face_height=self.face_roi_height)
            if left_eye_img is None or right_eye_img is None:
                print(f'\033[35m[WARNING] Failed to crop eye image in {box_file}, \
                please lower the argument `--face-roi-width` or `--face-roi-height`\033[0m')
            else:
                io.imsave(os.path.join(left_eyes_folder, box_file), left_eye_img)
                io.imsave(os.path.join(
                    right_eyes_folder, box_file), right_eye_img)

        # crop mouth
        print('\033[36mCropping mouth images ...\033[0m')
        os.makedirs(mouths_folder, exist_ok=True)
        crop_mouth_image(imgs_folder,
                        landmarks_folder,
                        mouths_folder,
                        np.load(self.mean_face),
                        crop_width=self.mouth_width,
                        crop_height=self.mouth_height,
                        start_idx=self.start_idx,
                        stop_idx=self.stop_idx,
                        window_margin=self.window_margin)

if __name__ == '__main__':
    cropper = Cropper()
    ### process one folder
    # cropper.crop_files()
    ### process only one video
    cropper.crop_one_video(video_name='./test/video_proc_0.avi')
