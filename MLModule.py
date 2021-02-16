
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, LSTM, BatchNormalization, Input, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import requests as images
import json 
import numpy as np
import re
import pandas
import time
from skimage import io
import datetime
import sys

class SuperML:

    labels = []
    labels_converter = []
    labels_count = 0

    train_folder_name = "NewDataSet1"
    train_data_count = 0
    train_path =  train_folder_name #os.getcwd() + "\\" +
    plot = "print"
    image_rows=32
    image_cols=32
    output_type = "json"

    lettersdict = {0 : 'ё', 1 : 'Ё', 2 : 'а', 3 : 'А',4 : 'б', 5 : 'Б', 6 : 'в', 7 : 'В',8 : 'г',9 : 'Г', 10 : 'д',
    11 : 'Д', 12 : 'е', 13 : 'Е', 14 : 'ж', 15 : 'Ж', 16 : 'з', 17 : 'З', 18 : 'и', 19 : 'И', 20 : 'й',21 : 'Й',
    22 : 'к', 23 : 'К', 24 : 'л', 25 : 'Л', 26 : 'м', 27 : 'М', 28 : 'н', 29 : 'Н', 30 : 'о', 31 : 'О', 32 : 'п',
    33 : 'П', 34 : 'р', 35 : 'Р', 36 : 'с', 37 : 'С', 38 : 'т', 39 : 'Т', 40 : 'у', 41 : 'У', 42 : 'ф', 43 : 'Ф',
    44 : 'х', 45 : 'Х', 46 : 'ц', 47 : 'Ц', 48 : 'ч', 49 : 'Ч', 50 : 'ш', 51 : 'Ш', 52 : 'щ', 53 : 'Щ', 54 : 'ъ',
    55 : 'Ъ', 56 : 'ы', 57 : 'Ы', 58 : 'ь', 59 : 'Ь', 60 : 'э', 61 : 'Э', 62 : 'ю', 63 : 'Ю', 64 : 'я', 65 : 'Я',
    66 : '0', 67 : '1', 68 : '2', 69 : '3', 70 : '4', 71 : '5', 72 : '6', 73 : '7', 74 : '8', 75 : '9', 76 : '.', 77 : ':',
    78 : "-", 79 : '+', 80 : '(', 81 : ')', 82 : '№'}


    def __init__(self):

        #Labels list init
        self.labels = os.listdir(self.train_path)
        #Labels count INIT
        self.labels_count = len(self.labels)

        #Train_data_count INIT
        for folder in os.listdir(self.train_path):
            self.train_data_count += len(os.listdir(self.train_path + '/' + folder))

        #Labels dictionary init
        self.labels_converter = dict(zip(range(0,self.labels_count), self.labels) )

    def emnist_model(self):
        input_shape = (32, 32, 1)    
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  

        inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  

        inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) 

        inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(128, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner) 
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) 

        inner = Conv2D(256, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv6')(inner)  
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Reshape(target_shape=((32, 256)), name='reshape')(inner)  
        inner = Dense(32, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 

        lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner) 
        lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
        reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)

        lstm1_merged = add([lstm_1, reversed_lstm_1b]) 
        lstm1_merged = BatchNormalization()(lstm1_merged)
        
        lstm_2 = LSTM(256, return_sequences=False, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(256, return_sequences=False, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
        reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

        lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  
        lstm2_merged = BatchNormalization()(lstm2_merged)

        y_pred = Dense(83, activation='softmax')(lstm2_merged) 

        model= Model(inputs=[inputs], outputs=y_pred)
            
        model.compile(loss='KLD', optimizer='Adam', metrics=['accuracy',  tf.keras.metrics.Precision(), 
        tf.keras.metrics.AUC(),  tf.keras.metrics.Recall()])
        #print(model.summary())
        return model


    def emnist_train(self, model):
        t_start = time.time()
        
        train_imgs = np.ndarray(((self.train_data_count-self.train_data_count//10)*2, self.image_rows, self.image_cols), dtype=np.uint8)
        val_imgs = np.ndarray(((self.train_data_count//10)*2, self.image_rows, self.image_cols), dtype=np.uint8)
        train_labels=[]
        val_labels=[]
        i, l, m=(0,0,0)
        for training_name in self.labels:
            current_dir = os.path.join(self.train_path, training_name)
            images = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and f.endswith(".png")]
            current_label = training_name
            for file in images:
                file_path = os.path.join(current_dir, file)
                image=cv2.imread(file_path, 0)
                image=cv2.resize(image, (self.image_rows, self.image_cols), interpolation=cv2.INTER_AREA)
                image = np.array([image])
                image = np.true_divide(image, 255)
                if i%10==9:
                    val_imgs[m] = image
                    val_labels.append(current_label)
                    m+=1
                    noise = np.random.normal(loc=0.3, scale=0.3, size=image.shape)
                    imageNoisy = np.clip(image + noise, 0, 1)
                    val_imgs[m] = imageNoisy
                    val_labels.append(current_label)
                    m+=1
                else:
                    train_imgs[l]=image
                    train_labels.append(current_label)
                    l+=1
                    noise = np.random.normal(loc=0.3, scale=0.3, size=image.shape)
                    imageNoisy = np.clip(image + noise, 0, 1)
                    train_imgs[l] = imageNoisy
                    train_labels.append(current_label)
                    l+=1
                i += 1
        
        train_imgs = train_imgs[..., np.newaxis]
        val_imgs = val_imgs[..., np.newaxis]
        train_labels = keras.utils.to_categorical(train_labels, len(self.labels))
        val_labels = keras.utils.to_categorical(val_labels, len(self.labels))
        model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
        history= model.fit(train_imgs, train_labels, validation_data=(val_imgs, val_labels),  shuffle=True, callbacks=[model_checkpoint], batch_size=512, epochs=35)
        print("Training done, dT:", time.time() - t_start)


    def emnist_predict_img(self, model, img):
        img_arr = np.expand_dims(img, axis=0)
        img_arr =  img_arr//255.0
        img_arr = img_arr.reshape((1, 32, 32, 1))
        result = np.argmax(model.predict([img_arr]),axis=1)
        return str(result)

    def letters_extract(self, image_file: str, out_size=32):
        img = io.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        url = image_file.replace(self.plot, self.output_type)
        img_erode = cv2.erode(thresh, np.ones((2,1), np.uint8), iterations=1)
        data = images.get(url).json()
        # Get contours
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        output = img.copy()
        letters = []
        number = data["receiptId"]
        date = data["operationTime"].split("T")[0].split("-")
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            #print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                letter_crop = gray[y:y + h, x:x + w]
                #print(letter_crop.shape)
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    y_pos = size_max//2 - h//2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max//2 - w//2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop
        letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
        new_date = datetime.datetime(int(date[0]), int(date[1]), int(date[2])).strftime('%d.%m.%Y')   
        return letters, number, new_date

    def img_to_str(self, model: emnist_model, image_file: str):
        letters, number, date = self.letters_extract(image_file)
        s_out = ""
        for i in range(len(letters)):
            image = letters[i][2]
            dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
            numletter=self.emnist_predict_img(model, image)
            s_out += self.lettersdict[int(numletter[1:-1])]
            if (dn > letters[i][1]/3.5):
                s_out += ' '
        data = json.dumps({"check_number": number, "check_date": date})
        return data

    
    def start_train(self):
        model = self.emnist_model()
        self.emnist_train(model)
    
    def start_test(self, image_file):
        model = keras.models.load_model('weights.h5')
        s_out = self.img_to_str(model, image_file)
        return s_out

    def info_search(self, input_str):
        FIOs_inn = re.findall(r'\)\s+\S+\s*\S+\s*\S+\s*ИНН:\s*[0-9]+', input_str)
        summ = re.findall(r'Итого:\s*\d*\s?\d+\.?\d\d', input_str)
        summ = summ[0].replace(' ', '').replace('Итого:', '')
        fio = FIOs_inn[0]
        fio = FIOs_inn[0].replace('\n', ' ')
        fio = fio.replace(')', '')
        fio = re.sub(" +", " ", fio)
        inn = re.findall(r'ИНН:\s*[0-9]+', fio)[0]
        fio = fio.replace(inn, '').lstrip()
        fio = fio.replace('.', '').rstrip()
        inn = inn.split()[1]
        return(fio, inn, summ)
'''
def main(argv):
    for url in argv:
        try:
            mlmodel = SuperML()
            text = mlmodel.start_test(url.replace(" ", ""))
            print(text)
        except Exception as e:
            print("image can’t processed", e)

if __name__ == "__main__":
    main(sys.argv[1:])
'''


def get_info_from_check(url):
    print(url)
    try:
        print("GELLLLOODFSDJFBDSKF")
        mlmodel = SuperML()
        text = mlmodel.start_test(url.replace(" ", ""))
        print(text)
        
        return text
    except Exception as e:
        print("image can’t processed", e)
        return json.dumps({"check_number": "None", "check_date": "None"})



    
