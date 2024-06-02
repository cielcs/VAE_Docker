import os
import cv2
import numpy as np
import math
#from numpy import random

class ImageDataGenerator(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.images = []
    self.img = None
    self.subset=''
    
  def sampling_image(self, img, input_shape):
    
    img_size = img.shape
    r = np.random.random_sample(2)
    y = (img_size[0]-input_shape[0])*r[0]
    y = int(y)
    x = (img_size[1]-input_shape[1])*r[1]
    x = int(x)
    samp_img = img[y:y+input_shape[0], x:x+input_shape[1],:]
    return samp_img

  def cropImg(self, img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_v = np.array([0, 0, 0])
    upper_v = np.array([255, 255, 100])
    v_mask = cv2.inRange(img_HSV, lower_v, upper_v)

    contours, hierarchy = cv2.findContours(v_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    x = [cv2.contourArea(c) for c in contours]
    rect = contours[np.argmax(x)]
    x, y, w, h = cv2.boundingRect(rect)
    return img[y:y+h, x:x+w]

  def flow(self, imgFolderPath, input_shape, batch_size=32, subset='training'):

    files = os.listdir(imgFolderPath)
    files_list = [f for f in files if os.path.isfile(os.path.join(imgFolderPath, f))]
    np.random.shuffle(files_list)

    while True:
      for filename in files_list:
        filepath = os.path.join(imgFolderPath,filename)

        #self.img = cv2.imread('/content/pantilt2/imgs.B/image_-5_-10.png')
        self.img = cv2.imread(filepath)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = self.cropImg(self.img)
        self.img = self.img.astype(np.float32)
        self.img = self.img/255

        while len(self.images)<batch_size:
            img = self.sampling_image(self.img, input_shape)
            self.images.append(img)

            # if subset=='validation':
            #   print(img)
        inputs = np.asarray(self.images, dtype=np.float32)
        self.reset()
        yield inputs


class MovieImageDataGenerator(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.movies = []
    self.cap = None
    self.subset=''
    
  def open(self,file):
    self.cap = cv2.VideoCapture(file)
    #変えた場所https://note.nkmk.me/python-opencv-fps-measure/
    self.cap.set(cv2.CAP_PROP_FPS, 30)
    # print(type(self.cap))

    # print(self.cap.isOpened())
    # print(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    # print(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    self.img_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # ret, frame1 = self.cap.read()
    # # print(frame1)
    # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    # ret, frame100 = self.cap.read()
    # # print(frame100)
    # print(frame1-frame100)

  def close(self):
    self.cap.release()


  def trim_images(self, startFrame, position, input_shape):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    
    samp_movie = np.zeros(input_shape)
    for i in range(0,input_shape[0]):
      ret,frame = self.cap.read()
      y = position[1]
      x = position[0]
      samp_movie[i,:,:,:] = frame[y:y+input_shape[1], x:x+input_shape[2],:]
  
    samp_movie = samp_movie.astype(np.float32)
    samp_movie = samp_movie/255

    return samp_movie

  def sampling_image(self, input_shape):
    
    frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    r = np.random.random_sample(3)
    #書き換えた場所
    
    r[0] = np.random.uniform(0.2,0.8)
    r[1] = np.random.uniform(0.2,0.8)
    # elif r[1] > 0.8:
    #   r[1] = np.random.uniform(0.7,0.8)
    # elif r[1]>0.4 and r[1]<0.6:
    #   r[1] = np.random.uniform(0.2,0.8)
    ######
    randposx = (self.img_size[0]-input_shape[1])*r[0]
    randposx = int(randposx)
    randposy = (self.img_size[1]-input_shape[2])*r[1]
    randposy = int(randposy)
    randposf = (frame_count-input_shape[0])*r[2]
    # print("%d, %d"%(self.img_size[0], self.img_size[1]))
    # print("%d, %d"%(input_shape[1], input_shape[2]))
    # print("%d, %d"%(self.img_size[0]-input_shape[1], self.img_size[1]-input_shape[2]))
    # print("%d, %d"%(randposx, randposy))
    
    samp_movie = self.trim_images(randposf, (randposx, randposy), input_shape)


    return samp_movie

  def flow(self, imgFolderPath, input_shape, batch_size=32, subset='training'):

    files = os.listdir(imgFolderPath)
    files_list = [f for f in files if os.path.isfile(os.path.join(imgFolderPath, f))]
    np.random.shuffle(files_list)
    while True:
      for filename in files_list:
        filepath = os.path.join(imgFolderPath,filename)

        self.open(filepath)
        while len(self.movies)<batch_size:
            movie = self.sampling_image(input_shape)
            self.movies.append(movie)
            #print("num of movies = %d"%(len(self.movies)))

        inputs = np.asarray(self.movies, dtype=np.float32)
        #self.close()
        self.reset()
        yield inputs

####
def test1():
  gen = MovieImageDataGenerator()
  gen.open('/content/data/movies/sample1.mp4')

  #x = gen.trim_images(0,10,(640-32,480-32),(32,32,3))
  w=128
  h=128
  x = gen.sampling_image((100,w,h,3))
  print(x.shape)

  CLIP_FPS = 20
  filepath = 'testtttt.mp4'
  codec = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter(filepath, codec, CLIP_FPS, (w, h))

  for idx in range(10):
#    f = x[idx,:,:,:]
#    print(f.shape)
    video.write(x[idx,:,:,:].astype(np.uint8))

  video.release()

def test2():
  datagen = MovieImageDataGenerator()
  #imgFolderPath='/content/data/movies/train/'
  imgFolderPath='/content/data/movies/test/'
  input_shape=(30,128,128,3)
  batch_size=16
  train_generator=datagen.flow(imgFolderPath, input_shape, batch_size=batch_size)
  for i in range(0,10):
    dataset = next(train_generator)
    print(dataset.shape)

    #print(next(train_generator))
  # print(next(train_generator))
  # print(next(train_generator))

####
if __name__=='__main__':
  test2()