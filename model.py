

from keras.models import Sequential,model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(3, 3), input_shape = (128, 128, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters=8, kernel_size=(5, 5), activation = 'relu',strides=(3,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale = 1./255,shear_range=0.25,zoom_range = 0.25,horizontal_flip = True)
test_datagen=ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r"C:\Users\Pavan\Desktop\hackonhills\leapGestRecog\train",target_size = (128, 128),batch_size = 32,class_mode = 'categorical')
test_data=test_datagen.flow_from_directory(r'C:\Users\Pavan\Desktop\hackonhills\leapGestRecog\test',target_size = (128, 128),batch_size = 1,class_mode = 'categorical')
#model.fit_generator(training_set,steps_per_epoch=3950,epochs=10,validation_data=test_data,validation_steps=50)

'''json_file = open('models\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk") 
model=loaded_model'''
import cv2
import numpy as np
#import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    
    face_img = img
  
    face = face_cascade.detectMultiScale(face_img) 
    for (x,y,w,h) in face: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), -1) 
        
    return face_img
    
cap = cv2.VideoCapture(0) 

proba=np.zeros((20,1))
s=0
count=0
jjj=0
font=cv2.FONT_HERSHEY_SIMPLEX
while True: 
    jjj+=1
    ret, frame = cap.read(0) 
    hhh=frame
    #cv2.imshow('VideoFace Detection', frame) 
    frame = detect_face(frame)
    
    ##########################
    import cv2
    import numpy as np
    #import matplotlib.pyplot as plt
    #%matplotlib inline
    #frame=cv2.imread('/home/user/Pictures/hand4.jpeg')
    kernel = np.ones((3,3),np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)
    mask,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (0,255,0), 3)
    ##############################
    #new_height = 300
    #img.resize(128, 128, Image.ANTIALIAS)
    #frame = 
    #mask=cv2.cvtColor(mask,cv2.COLOR_HSV2RGB)
    img=mask 
    #height , width , layers =  img.shape
    new_h=128
    new_w=128
    frame = cv2.resize(frame, (new_w, new_h))
    
 #   str=model.predict_classes()
    frame=np.expand_dims(frame,0)
    prob=model.predict(frame)
    stri=model.predict_classes(frame)[0]
    if prob[0][stri]>=0.90:
        
        proba=np.append(proba,[prob[0][stri]])
        s=1
    if (float(prob[0][stri]) < 0.90) & s:
        for j in range((len(proba)-16),(len(proba)-1)):
            if(proba[j]>=0.80):
                count+=1
        if count>=12 :
            frame=hhh
            if stri==1:
                cv2.putText(hhh,"DOWN",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==2:
                cv2.putText(hhh,"PUNCH",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==3:
                cv2.putText(hhh,"BREAK",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==4:
                cv2.putText(hhh,"INDEX",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==5:
                cv2.putText(hhh,"LETTER L",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==6:
                cv2.putText(hhh,"OK",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==7:
                cv2.putText(hhh,"STOP",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==8:
                cv2.putText(hhh,"STRAIGHT",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            if stri==9:
                cv2.putText(hhh,"THUMB",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            else :
                cv2.putText(hhh,"C",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)
            proba=np.zeros((20,1))
        else:
            cv2.putText(hhh,"NOT RECOGNISED",(360,30),font,4,(255,0,0),2,cv2.LINE_AA)

            
                    
        s=0
    cv2.imshow('Video Face Detection', hhh) 

    
 
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
        
cap.release() 
cv2.destroyAllWindows()



