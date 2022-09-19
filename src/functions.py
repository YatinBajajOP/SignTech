import streamlit as st
import cv2
import time
import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


##########################
# SIGN TO TEXT FUNCTIONS #
##########################

# Constants
contour_model = keras.models.load_model(r"./contour_sign_detection.h5")
edge_model = keras.models.load_model(r"./edge_sign_detection.h5")

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

def signToText():
    # st.write("Button clicked")
    
    cam = cv2.VideoCapture(0)
    num_frames =0
    word_dict = {'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'A','11':'B','12':'C','13':'D','14':'E','15':'F','16':'G','17':'H','18':'I','19':'J','20':'K','21':'L','22':'M','23':'N','24':'O','25':'P','26':'Q','27':'R','28':'S','29':'T','30':'U','31':'V','32':'W','33':'X','34':'Y'}
    prediction_list=[]
    predicted_word=''
    result=[]

    while True:
        ret, frame = cam.read()

        # filpping the frame to prevent inverted image of captured frame...
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        canny_white = cv2.Canny(roi, 100,125)


        if num_frames < 90:
            
            cal_accum_avg(gray_frame, accumulated_weight)
            
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        else: 
            # segmenting the hand region
            hand = segment_hand(gray_frame)
            

            # Checking if we are able to detect the hand...
            if hand is not None:
                
                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                canny_white = cv2.Canny(roi, 100, 125)
                cv2.imshow("edge", canny_white)
                canny_white = cv2.resize(canny_white, (64, 64))
                canny_white = cv2.cvtColor(canny_white, cv2.COLOR_GRAY2RGB)
                canny_white = np.reshape(canny_white, (1,canny_white.shape[0],canny_white.shape[1],3))
                
                cv2.imshow("Thesholded Hand Image", thresholded)
                            
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
                
                contour_pred = contour_model.predict(thresholded)
                edge_pred = edge_model.predict(canny_white)
                cv2.putText(frame_copy, "Show guesture for " + str(len(result)) + " letter", (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                prediction_list.append(word_dict[str(np.argmax(contour_pred))])
                prediction_list.append(word_dict[str(np.argmax(edge_pred))])
                if num_frames % 70 == 0:
                    predicted_word = max(prediction_list,key=prediction_list.count)
                    prediction_list.clear()
                    result.append(predicted_word)
                cv2.putText(frame_copy, predicted_word, (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                try:
                    cv2.putText(frame_copy, "previous predicted word: "+result[len(result)-2], (100, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                except:
                    pass


        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

        # incrementing the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.putText(frame_copy, "Hand sign recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        cv2.imshow("Sign Detection", frame_copy)


        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            cam.release()
            cv2.destroyAllWindows()
            break
    st.write("The detected letters are:")
    st.write(result)


##########################
# TEXT TO SIGN FUNCTIONS #
##########################

def removeFile(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4  ".format(input = avi_file_path, output = output_name))
    return True


def textToSign(string):
    removeFile(file_path='../static/output/video/avi/output.avi')
    removeFile(file_path='../static/output/video/mp4/output.mp4')

    if string != "":
        string = string.upper()
        string_display = ""
        nothing = cv2.imread('../static/data/ASL_Dataset/Test/Nothing/3001.jpg', cv2.IMREAD_COLOR)
        img_array = [nothing, nothing]
        
        for ch in string:
            string_display += str(ch)
            if ch == " ":
                ch = "Space"
            img = cv2.imread('../static/data/ASL_Dataset/Test/'+str(ch)+"/"+'3001.jpg', cv2.IMREAD_COLOR)
            cv2.putText(img=img, text=string_display, org=(20, 370), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                        fontScale=1, color=(255, 255,255),thickness=1)
            
            img_array.append(img)
            height, width, layers = img.shape
            size = (width,height)   
        out = cv2.VideoWriter('../static/output/video/avi/output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        if convert_avi_to_mp4(avi_file_path="../static/output/video/avi/output.avi", output_name="../static/output/video/mp4/output"):
            time.sleep(2)
            st.video("../static/output/video/mp4/output.mp4")