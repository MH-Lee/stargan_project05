from __future__ import print_function
from elice_solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

import numpy as np
import face_recognition
import argparse
import cv2
import os
import pickle
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from datetime import datetime

now = datetime.now()

"""parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()"""

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,help="path to output video file")
ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=50, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
ap.add_argument('--with_draw', help='do draw?', default='True')
args = vars(ap.parse_args())
##
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None

###### GAN Argument Start#####
'''
gan_args = dict()
gan_args['c_dim'] = 5
gan_args['c2_dim'] = 8
gan_args['celeba_crop_size'] = 178
gan_args['image_size'] = 128
gan_args['g_conv_dim'] = 64
gan_args['d_conv_dim'] = 64
gan_args['g_repeat_num'] = 6
gan_args['d_repeat_num'] = 6
gan_args['lambda_cls'] = 1
gan_args['lambda_rec'] = 10
gan_args['lambda_gp'] = 10
# Training configuration.
gan_args['dataset'] = 'CelebA'
gan_args['batch_size'] = 16
gan_args['num_iters'] = 200000
gan_args['num_iters_decay'] = 100000
gan_args['g_lr'] = 0.0001
gan_args['d_lr'] = 0.0001
gan_args['n_critic'] = 5
gan_args['beta1']=0.5
gan_args['beta2']=0.999
gan_args['resume_iters']=None
gan_args['selected_attrs']=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
# Test configuration.
gan_args['test_iters'] = 200000

#Miscellaneous
gan_args['num_workers'] = 1
gan_args['mode'] = 'test'
#Directories
gan_args['celeba_image_dir'] = 'test_data/'
gan_args['attr_path'] = 'test_data/info.txt'
gan_args['log_dir'] = 'stargan/logs'
gan_args['model_save_dir'] = 'stargan_celeba_128/models'
gan_args['sample_dir'] = 'stargan/samples'
gan_args['result_dir'] = 'test_data/result'
# Step size.
gan_args['log_step'] = 10
gan_args['sample_step'] = 1000
gan_args['model_save_step'] = 10000
gan_args['lr_update_step'] = 1000
config = gan_args
'''
###### GAN Argument End#####

mode = 'k'
video = 'v'
face_function = input("face mode(b=black hair, g=gender, age=a) : ")

if mode == 'k':
    mode = "knn"
    os.system("python knn.py")
elif mode == 's':
    mode = "svm"
    os.system("python svm.py")

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')
#gender_model = load_model('./gender_model/gender_mini_XCEPTION.21-0.95.hdf5')

knn_clf = pickle.load(open('./models/fr_knn.pkl', 'rb'))
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def preprocess(img):
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)
        else:
            break
    return img

if video == 'v':
    vc = cv2.VideoCapture('./data/video3.mp4')
    length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('length :', length)

    dir_name = 'test_data'  #+ "_" + str(idx)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print(dir_name)

    f = open("test_data/"+'info.txt','a+')
    f.write("1"+ '\n')
    f.write("5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"+'\n')
    f.write("0001.jpg"+" "+"-1   -1   1   -1   -1   -1   -1   -1   -1   1   -1   -1   -1   -1   -1   -1   -1   -1   -1   1   -1   1   -1   -1   1   1   -1   1   -1   -1   -1   -1   1   -1   -1   -1   1   -1   -1   1")
    f.close()
    for idx in range(length):
        img_bgr = vc.read()[1]
        if img_bgr is None:
            break

        start = cv2.getTickCount()


    ### preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_bgr_ori = img_bgr.copy()
        print(img_bgr_ori.shape)
        img_bgr = preprocess(img_bgr)

    ### detection
        border = (img_bgr.shape[1] - img_bgr.shape[0])//2
        img_bgr = cv2.copyMakeBorder(img_bgr,
                                    border, # top
                                    border, # bottom
                                    0, # left
                                    0, # right
                                    cv2.BORDER_CONSTANT,
                                    value=(0,0,0))

        (h, w) = img_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

    ### bbox
        list_bboxes = []
        list_confidence = []
    # list_dlib_rect = []



        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.6:
                    continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (l, t, r, b) = box.astype("int") # l t r b

            original_vertical_length = b-t
            t = int(t + (original_vertical_length)*0.15) - border
            b = int(b - (original_vertical_length)*0.05) - border

            margin = ((b-t) - (r-l))//2
            l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
            r = r + margin
            refined_box = [t-40,r+40,b+40,l-40]
            list_bboxes.append(refined_box)
            list_confidence.append(confidence)

    ### facenet
        if(len(list_bboxes)>0) :
            face_encodings = face_recognition.face_encodings(img_rgb, list_bboxes)

            if mode == "knn":
                closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
                is_recognized = [closest_distances[0][i][0] <= 0.4 for i in range(len(list_bboxes))]
                list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in zip(knn_clf.predict(face_encodings), list_bboxes, is_recognized, list_confidence)]

            elif mode == "svm":
                predictions = knn_clf.predict_proba(face_encodings)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                is_recognized = [ best_class_indices == 1 for i in range(len(list_bboxes))]
                list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in zip(knn_clf.predict(face_encodings), list_bboxes, best_class_indices, list_confidence)]
    # print (list_reconized_face)

            time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
            print ('%d, elapsed time: %.3fms'%(idx,time))

    ### blurring
            """try:
                f = open("/Users/a/Workspace/tkwoo_project/Eolgani_project/"+'info.txt','r+')
            except IOError:
                f = open("/Users/a/Workspace/tkwoo_project/Eolgani_project/"+'info.txt','w+')
                f.write("1"+ '\n')
                f.write("5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"+'\n')
                f.close()"""


            img_bgr_blur = img_bgr_ori.copy()
            for name, bbox, conf in list_reconized_face:
                t,r,b,l = bbox
                if name == 'unknown':
                    #file_name = str(now.day) + str(now.minute) + str(now.second) + str(idx) + ".jpg"
                    file_name = "0001.jpg"
                    #ww = r-l
                    #hh = b-t
                    #crop_img = img_bgr_blur[t:t+hh, l:l+ww]

                    crop_img = img_bgr_blur[t:b, l:r]
                    #gan_img = crop_img.copy()
                    #b,g,r = cv2.split(crop_img)
                    #crop_img = cv2.merge([r,g,b])
                    cv2.imwrite(os.path.join(dir_name , file_name) ,crop_img)
                    height, width = crop_img.shape[:2]
                    print(height, width)
                    os.system("python elice_project.py")
                    # celeba_loader = get_loader(config['celeba_image_dir'], config['attr_path'], config['selected_attrs'],
                    #                            config['celeba_crop_size'], config['image_size'], config['batch_size'],
                    #                            'CelebA', config['mode'], config['num_workers'])
                    #
                    # solver = Solver(celeba_loader, config)
                    # solver.test()
                    if face_function == "b":
                        gan_img = cv2.imread(os.path.join('test_data/result' , '1-1-images-black.jpg') , 1)
                    elif face_function == "g":
                        gan_img = cv2.imread(os.path.join('test_data/result' , '4-1-images-gender.jpg') , 1)
                    elif face_function == "a":
                        gan_img = cv2.imread(os.path.join('test_data/result' , '5-1-images-age.jpg') , 1)
                    else:
                        gan_img = cv2.imread(os.path.join('test_data/result' , '2-1-images-blond.jpg') , 1)
                    gan_img2 = cv2.resize(gan_img,(height,width))
                    img_bgr_blur[t:b, l:r] = gan_img2

                """#cv2.CV_Assert(ssize.width> 0 and ssize.height> 0)
                    face = img_bgr_blur[t:b, l:r]
                    #print ("face : ",face)
                    small = cv2.resize(face, (None, fx=.05, fy=.05,) interpolation=cv2.INTER_NEAREST)
                    blurred_face = cv2.resize(small, (face.shape[:2]), interpolation=cv2.INTER_NEAREST)
                    #print ("blurred_face : ",face)
                    img_bgr_blur[t:b, l:r] = blurred_face"""


    ### draw rectangle bbox
            if args["with_draw"] == 'True':
                source_img = Image.fromarray(img_bgr_blur)
                draw = ImageDraw.Draw(source_img)
                for name, bbox, confidence in list_reconized_face:
                    t,r,b,l = bbox
                # print (int((r-l)/img_bgr_ori.shape[1]*100))
                    font_size = int((r-l)/img_bgr_ori.shape[1]*500)

                    draw.rectangle(((l,t),(r,b)), outline=(0,255,128))

                    draw.rectangle(((l,t-font_size-2),(r,t+2)), fill=(0,255,128))
                    draw.text((l, t - font_size), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', font_size), fill=(0,0,0,0))
                show = np.asarray(source_img)
                if idx == 0:
                    writer = cv2.VideoWriter('video_3.avi', fourcc, args["fps"], (640,360), True)
                #output = np.zeros((300,300,3),dtype="uint8")
                #output[0:300, 0:300] = show
                writer.write(show)
                """
                show = np.asarray(source_img)
                cv2.imshow('show', show)
                cv2.imshow('blur', img_bgr_blur)
                key = cv2.waitKey(30)
                if key == 27:
                    break"""
        else :
            source_img = Image.fromarray(img_bgr_ori)
            draw = ImageDraw.Draw(source_img)
            show = np.asarray(source_img)
            if idx == 0:
                writer = cv2.VideoWriter('video_3.avi', fourcc, args["fps"], (640,360), True)
                #output = np.zeros((300,300,3),dtype="uint8")
                #output[0:300, 0:300] = show
            writer.write(show)
            #cv2.imshow('show', show)
            #cv2.imshow('blur', img_bgr_blur)
            """key = cv2.waitKey(30)
            if key == 27:
                break"""

writer.release()
