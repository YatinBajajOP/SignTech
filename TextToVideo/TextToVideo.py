
import cv2
import time
import os

cwd = os.getcwd()

def text_to_video(string):
    string = string.upper()
    string_display = ""
    img_array = []
    
    for ch in string:
        string_display += ch
        if ch == " ":
            ch = "Space"
        img = cv2.imread(cwd+ '//data//ASL_Dataset//Test//'+ch+"//"+'3001.jpg', cv2.IMREAD_COLOR)
        cv2.putText(img=img, text=string_display, org=(20, 370), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255,255),thickness=1)
        
        img_array.append(img)
        height, width, layers = img.shape
        size = (width,height)
#         cv2.imshow("Image", img)
#         cv2.waitKey(800)
#         cv2.destroyAllWindows()
    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1.5, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

text_to_video("My name is Yatin")




