from facenet_code.detection import Detection
from facenet_code.encoder import Encoder
from scipy.linalg import svd
from imutils import paths
import numpy as np
import argparse
import cv2
import os

class DetectBlur(object):
    def __init__(self, video, threshold=0.8):
        self.video = video
        self.threshold = threshold
        print(self.threshold)
        self.video_frames = []

        self.detect = Detection()

        self.process()
    
    def process(self):
        self.create_output_folder()
        self.get_video_frames()
        self.detect_blur()

    def create_output_folder(self):
        if not os.path.isdir('output'):
            os.mkdir('output')
        video_name = self.video.split('.')[0]
        if not os.path.isdir('output/'+video_name):
            os.mkdir('output/'+video_name)
        if not os.path.isdir('output/'+video_name+'/'+'frames'):
            os.mkdir('output/'+video_name+'/'+'frames')

    def get_blur_degree(self, img, sv_num=10):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        u, s, v = np.linalg.svd(gray_img)
        top_sv = np.sum(s[0:sv_num])
        total_sv = np.sum(s)
        return top_sv/total_sv

    # def get_blur_map(self, img, win_size=10, sv_num=3):
    #     gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     new_img = np.zeros((gray_img.shape[0]+win_size*2, gray_img.shape[1]+win_size*2))
    #     for i in range(new_img.shape[0]):
    #         for j in range(new_img.shape[1]):
    #             if i<win_size:
    #                 p = win_size-i
    #             elif i>gray_img.shape[0]+win_size-1:
    #                 p = gray_img.shape[0]*2-i
    #             else:
    #                 p = i-win_size
    #             if j<win_size:
    #                 q = win_size-j
    #             elif j>gray_img.shape[1]+win_size-1:
    #                 q = gray_img.shape[1]*2-j
    #             else:
    #                 q = j-win_size
    #             new_img[i,j] = img[p,q]
    #     blur_map = np.zeros((gray_img.shape[0], gray_img.shape[1]))
    #     max_sv = 0
    #     min_sv = 1
    #     for i in range(gray_img.shape[0]):
    #         for j in range(gray_img.shape[1]):
    #             block = new_img[i:i+win_size*2, j:j+win_size*2]
    #             u, s, v = np.linalg.svd(block)
    #             top_sv = np.sum(s[0:sv_num])
    #             total_sv = np.sum(s)
    #             sv_degree = top_sv/total_sv
    #             if max_sv < sv_degree:
    #                 max_sv = sv_degree
    #             if min_sv > sv_degree:
    #                 min_sv = sv_degree
    #             blur_map[i, j] = sv_degree
    #     blur_map = (blur_map-min_sv)/(max_sv-min_sv)
    #     return blur_map

    def get_video_frames(self):
        vidcap = cv2.VideoCapture(self.video)
        success, image = vidcap.read()
        count = 0
        while success:
            self.video_frames.append(image)    
            success, image = vidcap.read()

    def print_box(self, frame, name, blur_degree, face_bb, color):
        left, top, right, bottom = face_bb
        width = right - left
        height = bottom - top

        if height > width:
            tam = int(height/4)
        else:
            tam = int(width/4)

        cv2.putText(frame, name, (right + 15, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, blur_degree, (right + 15, top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), color, 1)

        cv2.line(frame, (left, top), (left+tam, top), color, 3)
        cv2.line(frame, (left, top), (left, top+tam), color, 3)

        cv2.line(frame, (left, bottom), (left, bottom-tam), color, 3)
        cv2.line(frame, (left, bottom), (left+tam, bottom), color, 3)

        cv2.line(frame, (right, top), (right-tam, top), color, 3)
        cv2.line(frame, (right, top), (right, top+tam), color, 3)

        cv2.line(frame, (right, bottom), (right-tam, bottom), color, 3)
        cv2.line(frame, (right, bottom), (right, bottom-tam), color, 3)

    def detect_blur(self):
        output_video = None
        if output_video is None:
            video_name = self.video.split('.')[0]
            size = (self.video_frames[0].shape[1], self.video_frames[0].shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_video = cv2.VideoWriter('output/'+video_name+'/'+video_name+'.avi',fourcc, 5, size, True)
        for i, frame in enumerate(self.video_frames):
            print('[INFO] detecting blur in image '+str(i+1)+'/'+str(len(self.video_frames)))
            faces = self.detect.find_faces(frame)
            if len(faces) > 0:
                for face in faces:
                    if face.confidence > 0.9:
                        text = "Not Blurry"
                        boxes = face.bounding_box.astype(int)
                        left, top, right, bottom = boxes
                        face_image = frame[top:bottom, left:right]
                        blur_degree = self.get_blur_degree(face_image)
                        if blur_degree > self.threshold:
                            text = "Blurry"
                        self.print_box(frame, text, "{:.2f}".format(blur_degree), boxes, (255,255,255))
            if output_video is not None:
                output_video.write(frame)
                cv2.imwrite('output/'+video_name+'/'+'frames/frame_'+str(i+1)+'.jpg', frame)
                
        if output_video is not None:
            output_video.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=str, help='the video input to detect blurry faces')
    ap.add_argument('--threshold', default=0.8, type=float, help='the threshold of blur degree to classify if some face is blurry or not')
    args = vars(ap.parse_args())

    DetectBlur(video=args['video'], threshold=args['threshold'])

    

