#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Embedding, Conv2D, TimeDistributed, Reshape, Flatten, Concatenate
from keras.models import Sequential
from keras import Model
from keras.losses import mean_absolute_error, categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
import time
import os
from os.path import isfile, join
from google.colab.patches import cv2_imshow


# In[ ]:


def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


# In[ ]:


def rand_img(size):
    return np.random.randint(0, 256, size) / 255.0


# In[ ]:


def rand_sentence(len, dict_len):
    return np.random.randint(0, dict_len, len)


# In[ ]:


def ascii_encode(message, sentence_len):
    sen = np.zeros((1, sentence_len))
    for i, a in enumerate(message.encode("ascii")):
        sen[0, i] = a
    return sen


def ascii_decode(message):
    return ''.join(chr(int(a)) for a in message[0].argmax(-1))


# In[ ]:


def data_generator(image_size, sentence_len, dict_len, batch_size=32):
    while True:
        x_img = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        x_sen = np.zeros((batch_size, sentence_len))
        y_img = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        y_sen = np.zeros((batch_size, sentence_len, dict_len))
        for i in range(batch_size):
            img = rand_img(image_size)
            sentence = rand_sentence(sentence_len, dict_len)
            sentence_onehot = to_categorical(sentence, dict_len)
            x_img[i] = img
            x_sen[i] = sentence
            y_img[i] = img
            y_sen[i] = sentence_onehot
        yield [[x_img, x_sen], [y_img, y_sen]]


# In[ ]:


def get_model(image_shape, sentence_len, dict_len):
  # the encoder part
    input_img = Input(image_shape)
    input_sen = Input((sentence_len,))
    embed_sen = Embedding(dict_len, 450)(input_sen)
    embed_sen = Flatten()(embed_sen)
    embed_sen = Reshape((image_shape[0], image_shape[1], 1))(embed_sen)
    convd_img = Conv2D(20, 1, activation="relu")(input_img)
    cat_tenrs = Concatenate(axis=-1)([embed_sen, convd_img])
    out_img = Conv2D(3, 1, activation='relu', name='image_reconstruction')(cat_tenrs)
    
    # the decoder part
    decoder_model = Sequential(name="sentence_reconstruction")
    decoder_model.add(Conv2D(1, 1, input_shape=(450, 450, 3)))
    decoder_model.add(Reshape((sentence_len, 450)))
    decoder_model.add(TimeDistributed(Dense(dict_len, activation="softmax")))
    out_sen = decoder_model(out_img)
    
    # creating models
    model = Model(inputs=[input_img, input_sen], outputs=[out_img, out_sen])
    model.compile('adam', loss=[mean_absolute_error, categorical_crossentropy], metrics={'sentence_reconstruction': categorical_accuracy})
    encoder_model = Model(inputs=[input_img, input_sen], outputs=[out_img])
    return model, encoder_model, decoder_model


# In[ ]:


image_shape = (450, 450, 3)
sentence_len = 450 #100
dict_len = 450  #200
gen = data_generator(image_shape, sentence_len, dict_len, 32)
model, encoder, decoder = get_model(image_shape, sentence_len, dict_len)


# In[ ]:


input_loc = 'test2.mp4'
output_loc = '/content/Frames/'
video_to_frames(input_loc, output_loc)


# In[ ]:


pathIn = '/content/Frames/'


# In[ ]:


routnames = []
for f in os.listdir(pathIn):
  filename = os.fsdecode(f)
  routandname =  str(str(pathIn)[0:-1] + '/' + str(filename))
  routnames.append(routandname)
  img = cv.imread(routandname)
  img = cv.resize(img,(450,450))
  cv.imwrite(routandname,img)


# In[ ]:


files = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key = lambda x: x[5:-4])
files.sort()
routnames.sort(key = lambda x: x[5:-4])
routnames.sort()


# In[ ]:


cjump = 0
num = 0
Array_to_encode_text = []
text_to_encode = "Este.es.el.texto.para.ocultar.en.el.video"
for letter in text_to_encode:
  Array_to_encode_text.append(letter)
while len(Array_to_encode_text)<len(files)/jump:
  Array_to_encode_text.append("|")
Array_to_encode = []
while len(Array_to_encode)<len(files):
  Array_to_encode.append("*")
for i in range(len(Array_to_encode)):
  Array_to_encode[cjump] = Array_to_encode_text[num]
  cjump+=jump
  num+=1
  if cjump>len(files):
    break


# In[ ]:


directoryresult = '/content/EncodedFrames/'


# In[ ]:


targets = []
resultr = []
model.load_weights("best_weights_450.h5")
for f in os.listdir(pathIn):
  filename = os.fsdecode(f)
  #for im in FramestoEncode:
   #if im == os.fsdecode(f):
      #for letter in Array_to_encode:
  routandname =  str(str(pathIn)[0:-1] + '/' + str(filename))
  targets.append(routandname)
  targets.sort(key = lambda x: x[5:-4])
  targets.sort()
  routresult = str(str(directoryresult)[0:-1]  + '/' + str(filename[:-4])+"enc.jpg")
  resultr.append(routresult)
  resultr.sort(key = lambda x: x[5:-4])
  resultr.sort()


# In[ ]:


for i in range(len(targets)):
    img = np.expand_dims(img_to_array(load_img(targets[i])) / 255.0, axis=0)
    sen = ascii_encode(Array_to_encode[i], sentence_len)
    y_img = encoder.predict([img, sen])
    imgENC = np.squeeze(y_img,axis=0)
    R,G,B = cv.split(imgENC)
    R = R*255
    G = G*255
    B = B*255
    resultimg = cv.merge((B,G,R))
    cv.imwrite(resultr[i],resultimg)
print("success")


# In[ ]:


pathInCod= '/content/EncodedFrames/'
pathOutCod = '/content/videoEncoded.avi'
fps = 25
frame_array = []
filesCod = [f for f in os.listdir(pathInCod) if isfile(join(pathInCod, f))]
#for sorting the file names properly
filesCod.sort(key = lambda x: x[5:-4])
filesCod.sort()
frame_array = []
filesCod = [f for f in os.listdir(pathInCod) if isfile(join(pathInCod, f))]
#for sorting the file names properly
filesCod.sort(key = lambda x: x[5:-4])
filesCod.sort()
for i in range(len(files)):
    filename=pathInCod + filesCod[i]
    #reading each files
    img = cv.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv.VideoWriter(pathOutCod,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()

