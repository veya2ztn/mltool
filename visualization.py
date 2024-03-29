try:import cv2
except:pass
from PIL import Image, ImageOps, ImageStat, ImageDraw
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
except:
    pass
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

def smoothhist(data,ax=None,**kargs):
    density = gaussian_kde(data)
    xs = np.linspace(min(data),max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    x = xs
    y = density(xs)
    y = y/y.max()
    if ax is not None:
        ax.plot(x,y,**kargs)
    else:
        plt.plot(x,y,**kargs)

def smoothheatdensity(x,y):
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xgrid = np.linspace(min(x), max(x),40)
    ygrid = np.linspace(min(y), max(y),40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    # Plot the result as an image
    plt.imshow(Z.reshape(Xgrid.shape),origin='lower', aspect='auto',extent=[min(x), max(x), min(y), max(y)], cmap='Blues')

def errorbarplot(data,x_axis=None,error_mode='var',color='r',alpha=0.3,main_linewidth=0.3,label=None,**kargs):
    assert len(data.shape)==2

    mean = data.mean(0)
    var  = data.var(0)
    ku   = ((data - mean) ** 4).mean(0) / (var**2+0.01) #计算峰度

    if   error_mode == "var": err  = var
    elif error_mode == "ku":  err  = ku
    else:raise
    if x_axis is None:x_axis = np.arange(len(mean))
    plt.errorbar(x_axis, mean, yerr = err,alpha=alpha,color=color,**kargs)
    plt.plot(x_axis, mean, linewidth=main_linewidth,color=color,label=label)
