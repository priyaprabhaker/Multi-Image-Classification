import os
import logging 
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from skimage import exposure, color
from tensorflow.keras.preprocessing import image

def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap

import numpy as np

def predict_frame(image):
    # apply pre-processing & reshaping to (1,224,224,3)
    preprocessed_image=np.expand_dims(image/255,axis=0)
    # saved model
    model = load_model("denselayer.h5")

    
    prediction = model.predict(preprocessed_image)    

    return prediction

if __name__ == "__main__":

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
    
    # settings for rectangle in the frame
    offset = 2
    width = 224
    x = 160
    y = 120

    font = cv2.FONT_HERSHEY_DUPLEX 
    font_size = 0.9
    color = (0, 0, 0) # black color frame
    bold = 2

    
    webcam = init_cam(640, 480)
    key = None
    count_frame = 0
    classes=['bottle','Empty','Glasses']
    #if this is false, we can also predict by loading images
    webcam_mode = True

    try:
        # q key not pressed 
        while key != 'q':
            count_frame +=1

            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            # Make real-time predictions calling the function "predict_frame" every 50 frames
            if count_frame == 30: 

                if webcam_mode:
                    # reverse color channels
                    image = frame[y:y+width, x:x+width, :]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                    
                else:
                    file_name = "test.png" 
                    image = image.load_img(file_name, target_size=(224, 224))
                    image = np.array(image)
                
                prediction = predict_frame(image)
                print(image.shape)
                print(f"\nPrediction: {prediction}\n")
                count_frame = 0
                for i, class_ in enumerate(classes):
                    ypred = 100*prediction[0][i] # prediction percentage
                    text = f"{ypred: 7.2f}% {class_}"
                    position = (x+width+10, y+40*(1+i))
                    cv2.putText(frame, text, position, font, font_size, color, bold)
                    print(text)
                    
                cv2.imshow('frame', frame) # frame to be displayed

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              

    finally:
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()