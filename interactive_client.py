import cv2
import numpy as np
import requests
import json
from img_utils.image_sender import ImageSender

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),8,(255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),8,(255),-1)




if __name__ == "__main__":

    host_ip = "localhost"

    host="http://{}".format(host_ip)
    port="5000"
    url = "{}:{}/process".format(host, port)


    img = np.zeros((512,512), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)


    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1)
        if k != -1:

            ret=ImageSender.prepare_to_send(img)
            files={'media': ("temp.png",ret)}
            data = requests.post(url, files=files)
            responseData = json.loads(data.content)
            image = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
            if len(image.shape)==2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image, str(responseData["prediction"]), (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.destroyWindow("image")
            cv2.imshow("Result", image)
            cv2.waitKey(0)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', draw_circle)
            img = np.zeros((512, 512), np.uint8)

    cv2.destroyAllWindows()
