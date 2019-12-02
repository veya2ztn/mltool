import cv2
from PIL import Image, ImageOps, ImageStat, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def gray_print(img):
    plt.imshow(img,'gray')
def rgb_print(img):
    img_rgb=img[:,:,[2,1,0]]
    plt.imshow(img_rgb,'gray')
#plt.rcParams['figure.figsize'] = (10.0, 8.0) 
def draw_line(image,shape,color='white',mode='xywh'):
    img = image.copy()
    dim = len(img.shape)
    if dim == 3: color= (0, 255, 255)
    elif dim ==2: color = 0.1
    if mode == 'xyxyxyxy':
        x1s,y1s,x2s,y2s,x3s,y3s,x4s,y4s=shape
        if type(x1s) == float or type(x1s) == int:
            x1, y1, x2, y2,x3, y3, x4, y4 = x1s,y1s,x2s,y2s,x3s,y3s,x4s,y4s
            pts = np.array([[x1,y1],  [x2,y2], [x3,y3], [x4,y4]], np.int32).reshape((-1, 1, 2))
            _=cv2.polylines(img,[pts], True, color,2)
        else:
            for i in range(len(x1s)):
                x1, y1, x2, y2 = int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])
                x3, y3, x4, y4 = int(x3s[i]), int(y3s[i]), int(x4s[i]), int(y4s[i])
                pts = np.array([[x1,y1],  [x2,y2], [x3,y3], [x4,y4]], np.int32).reshape((-1, 1, 2))
                _=cv2.polylines(img,[pts], True,color ,2)
        return img
    else:
        if  mode == 'xywh':
            center_x,center_y,width,height=shape
            x1s, y1s, x2s, y2s = center_x - width//2, center_y - height//2, center_x + width//2, center_y + height//2
        elif mode == 'xyxy':
            x1s, y1s, x2s, y2s = shape

        if type(x1s) == float or type(x1s) == int or isinstance(x1s,np.float64):
            x1s, y1s, x2s, y2s = int(x1s), int(y1s), int(x2s), int(y2s)
            _=cv2.rectangle(img,(x1s, y1s),(x2s, y2s),color,2)
        else:
            for i in range(len(x1s)):
                x1, y1, x2, y2 = int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])
                _=cv2.rectangle(img,(x1, y1),(x2, y2),color,2)
        return img
