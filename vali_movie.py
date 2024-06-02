import os
import math
import cv2
import numpy as np



def reconstruct_movie(vae, input_shape, filepath, input_file, output_file, fps=30, stride_x=40, stride_y=40, max_frame_count=None):
    cap = cv2.VideoCapture(filepath)
    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    frame_count = max_frame_count
    if frame_count is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, codec, fps, (img_size[1],img_size[0]))

    codec_in = cv2.VideoWriter_fourcc(*'mp4v')
    video_in = cv2.VideoWriter(input_file, codec_in, fps, (img_size[1],img_size[0]))

    model_out_height = input_shape[1]
    model_out_width = input_shape[2]


    for idx in range( int(frame_count/input_shape[0]) ):
        print(idx)

        model_in = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        model_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        count_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        for i in range(input_shape[0]):
            ret, frame = cap.read()
            model_in[i,:,:,:] = frame

        for y in range(0,img_size[0]-model_out_height,stride_y):
            for x in range(0,img_size[1]-model_out_width,stride_x):
                subimg = model_in[:, y:y+model_out_height, x:x+model_out_width, :]
                #print(subimg.shape)
                subimg = subimg.astype(np.float32)
                subimg = subimg/255
                subimg = np.reshape(subimg,(1,input_shape[0],model_out_height,model_out_width, input_shape[3]))
                img_mu, img_sigma = vae.img_predict(subimg)

                model_out[:,y:y+model_out_height, x:x+model_out_width, :] += img_mu[0]
                count_out[:,y:y+model_out_height, x:x+model_out_width, :] += 1

        model_out[count_out!=0] = model_out[count_out!=0]/count_out[count_out!=0]
        model_out *= 255
       

        for i in range(model_out.shape[0]):
            video.write(model_out[i,:,:,:].astype(np.uint8))
            video_in.write(frame)

    video.release()
    video_in.release()


def reconstruct_movie_new(vae, input_shape, filepath, output_file, fps=30, stride_x=40, stride_y=40, max_frame_count=None):
    cap = cv2.VideoCapture(filepath)
    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    frame_count = max_frame_count
    if frame_count is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, codec, fps, (img_size[1],img_size[0]))

    model_out_height = input_shape[1]
    model_out_width = input_shape[2]

    for idx in range( int(frame_count/input_shape[0]) ):
        print(idx)

        model_in = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        model_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        count_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        for i in range(input_shape[0]):
            ret, frame = cap.read()
            model_in[i,:,:,:] = frame

        model_in = model_in.astype(np.float32)
        model_in = model_in/255

        for y in range(0,img_size[0]-model_out_height,stride_y):
            for x in range(0,img_size[1]-model_out_width,stride_x):
                subimg = model_in[:, y:y+model_out_height, x:x+model_out_width, :]
                subimg = np.reshape(subimg,(1,input_shape[0],model_out_height,model_out_width, input_shape[3]))
                img_mu, img_sigma = vae.img_predict(subimg)

                model_out[:,y:y+model_out_height, x:x+model_out_width, :] += img_mu[0]
                count_out[:,y:y+model_out_height, x:x+model_out_width, :] += 1

        model_out[count_out!=0] = model_out[count_out!=0]/count_out[count_out!=0]
        model_out *= 255
       
        for i in range(model_out.shape[0]):
            video.write(model_out[i,:,:,:].astype(np.uint8))

    video.release()


def encode_mp4(data, output_file, fps=30):
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, codec, fps, (data.shape[2],data.shape[1]))

    for i in range(data.shape[0]):
        video.write(data[i,:,:,:].astype(np.uint8))

    video.release()


def heatmap_movie(vae, input_shape, filepath, output_file, fps=30, stride_x=40, stride_y=40, max_frame_count=None):
    cap = cv2.VideoCapture(filepath)
    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    frame_count = max_frame_count
    if frame_count is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, codec, fps, (img_size[1],img_size[0]))

    model_out_height = input_shape[1]
    model_out_width = input_shape[2]

    for idx in range( int(frame_count/input_shape[0]) ):
        print(idx)

        model_in = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        model_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        count_out = np.zeros((input_shape[0],img_size[0],img_size[1],3))
        for i in range(input_shape[0]):
            ret, frame = cap.read()
            model_in[i,:,:,:] = frame

        for y in range(0,img_size[0]-model_out_height,stride_y):
            for x in range(0,img_size[1]-model_out_width,stride_x):
                subimg = model_in[:, y:y+model_out_height, x:x+model_out_width, :]
                subimg = subimg.astype(np.float32)
                subimg = subimg/255
                subimg = np.reshape(subimg,(1,input_shape[0],model_out_height,model_out_width, input_shape[3]))
                img_mu, img_sigma = vae.img_predict(subimg)

                model_out[:,y:y+model_out_height, x:x+model_out_width, :] += ((subimg - img_mu)**2/img_sigma)[0] 
                count_out[:,y:y+model_out_height, x:x+model_out_width, :] += 1

        model_out[count_out!=0] = model_out[count_out!=0]/count_out[count_out!=0]
        model_out *= 255
       

        for i in range(model_out.shape[0]):
            video.write(model_out[i,:,:,:].astype(np.uint8))

    video.release()


###
VAEModulePath = '/content/VAE_movie'

import sys
sys.path.append(VAEModulePath)

import VAE_movie
vae = VAE_movie.VAE_movie(None, None)
vae.load('models_movie/encoder.h5','models_movie/decoder.h5')

validimgFolderPath='/content/data/movies/'
filename = '1normal.mp4'

input_shape = (6,64,64,3)
filepath = os.path.join(validimgFolderPath,filename)

#reconstruct_movie(vae, input_shape, filepath, input_file='testin1.mp4', output_file='testout1.mp4', max_frame_count=20*6, stride_x=40, stride_y=40)
#reconstruct_movie_new(vae, input_shape, filepath, output_file='testout_8frame.mp4', max_frame_count=20*6, stride_x=40, stride_y=40)

heatmap_movie(vae, input_shape, filepath, output_file='heatmap_1normal.mp4', fps=30, stride_x=32, stride_y=32, max_frame_count=None)

# import numpy as np
# np.save('heatmap', heatmap_movie(vae, input_shape, filepath, output_file='heatmap.mp4', max_frame_count=20*6, stride_x=40, stride_y=40))