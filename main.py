from cmath import sqrt
from ctypes.wintypes import POINT
from queue import Empty
from types import NoneType
import numpy as np
import cv2
from mss import mss
from PIL import Image
import pandas as pd
import pyautogui
import win32api, win32con


bounding_box = {'top': 200, 'left': 2560, 'width': 1280, 'height': 720}

def calcDistance(point1, point2):
    x_1 = point1[0]
    y_1 = point1[1]
    x_2 = point2[0]
    y_2 = point2[1]

    pythagyros = abs(sqrt(pow((x_1-x_2), 2)+ pow((y_1 - y_2),2)))

    return pythagyros

def getRelativeMovement(point1, point2):
    x_1 = point1[0]
    y_1 = point1[1]
    x_2 = point2[0]
    y_2 = point2[1]

    return ( (x_2 - x_1), (y_1 - y_2))


sct = mss()


def nothing(x):
    pass


H = 103
S = 255
V = 255
Hl = 0
Sl = 191
Vl = 119



while True:
    sct_img = np.array(sct.grab(bounding_box))

    hsv = cv2.cvtColor(sct_img, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([Hl,Sl,Vl])
    upper_blue = np.array([H,S,V])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
  


    # Find the circle blobs on the binary mask:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use a list to store the center and radius of the target circles:
    detectedCircles = []
    
    for i, c in enumerate(contours):

        # Approximate the contour to a circle:
        (x, y), radius = cv2.minEnclosingCircle(c)

        # Compute the center and radius:
        center = (int(x), int(y))
        radius = int(radius)


        if radius > 20 and radius < 40:
            # Draw the circles:
            cv2.circle(sct_img, center, radius, (0, 0, 255), 2)
            cv2.rectangle(sct_img, (center[0] - 5, center[1] - 5), (center[0] + 5, center[1] + 5), (0, 128, 255), -1)

            # Store the center and radius:
            
            detectedCircles.append([center, radius])

    ###### fixpunkt Mauszeiger
    mouseCursorPos = (int(1280/2), 353)

    if detectedCircles is not Empty:
        circles = pd.DataFrame(detectedCircles, columns=['Position', 'Center'])

        #Abstand zum Mauszeiger
        distanceToMouse = np.zeros((len(detectedCircles)))
        xpos = np.zeros((len(detectedCircles)))
        ypos = np.zeros((len(detectedCircles)))

        for i in range(len(detectedCircles)):
            distanceToMouse[i] = calcDistance(mouseCursorPos, circles.loc[i]['Position'])
            pos = circles.loc[i]['Position']
            xpos[i] = pos[0]
            ypos[i] = pos[1]
        
        #add circle to DataFrame
        circles['DistanceToCursor'] = distanceToMouse
        circles['X'] = xpos
        circles['Y'] = ypos


        if circles is not Empty and len(circles) != 0:
            for i in range(len(detectedCircles)):
                x = int(circles.loc[i]['X'])
                y = int(circles.loc[i]['Y'])
                cv2.putText(sct_img, "X:" + str(x) + " Y:" + str(y), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2,cv2.LINE_AA,False)

            minDistancePoint_index = circles[['DistanceToCursor']].idxmin()
            

            minDistancePoint = tuple(circles.loc[minDistancePoint_index]['Position'].values[0])
            minDistance = circles.loc[minDistancePoint_index]['DistanceToCursor'].values[0];
            #minDistancePoint = minDistancePoint.get(1)
            
            cv2.line(sct_img, mouseCursorPos, minDistancePoint, (255,255,255),2, cv2.LINE_AA)

            toggle = 1

            

            if minDistance < 18:
                #pyautogui.leftClick()
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, minDistancePoint[0], minDistancePoint[1], 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, minDistancePoint[0], minDistancePoint[1], 0, 0)
                pass
            elif minDistance > 18:
                
                relMov = getRelativeMovement(mouseCursorPos, minDistancePoint)
                cv2.putText(sct_img, str(relMov), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2,cv2.LINE_AA,False)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, relMov[0], relMov[1] * -1, 0, 0)
            else:
                pass






    cv2.imshow('screen', sct_img)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

