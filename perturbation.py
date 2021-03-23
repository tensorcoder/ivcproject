import numpy as np
import cv2
from image import normaliseImg

def recursive_blur(img, n):
    blurred = img
    for i in range(n):
        blurred = Gaussian_Blur(blurred)
    return blurred

def get_random_gaussian_number(mean, std, shape):
    return np.random.normal(mean, std, shape).astype(np.uint8), np.random.normal(mean, std, shape)

def gaussian_pixel_noise(img, std):
    _, number = get_random_gaussian_number(mean=0, std=std, shape=[224, 224, 3])
    img = np.array(img)
    number = np.array(number)
    modified = img+number
    if np.max(modified)>255 or np.min(modified)<0:
        modified = np.clip(modified, 0, 255)
                 
    return np.array(modified, dtype='int')

def Gaussian_Blur(img, Intensity=1):
    ImageArray = img
    kernel = np.ones((3, 3), np.float32)
    kernel[0] = [1,2,1]
    kernel[1] = [2,4,2]
    kernel[2] = kernel[0]
    kernel = (kernel * Intensity)/16
    blurred = cv2.filter2D(ImageArray, -1, kernel)
    return blurred


def image_contrast_increase(img, intensity):
    ones = np.ones((224,224,3))
    contrast_increase = ones*intensity
    increased = np.multiply(img,contrast_increase)
    if np.max(increased)>255:
        increased = np.clip(increased, 0, 255)
    
    return np.array(increased, dtype='int')


def image_contrast_decrease(img, intensity):
    ones = np.ones((224,224,3))
    contrast_decrease = ones*intensity
    decreased = np.multiply(img,contrast_decrease)
    if np.min(decreased)<0:
        decreased = np.clip(decreased, 0, 255)
    
    return np.array(decreased, dtype='int')



def image_brightness_increase(img, intensity):
    ones = np.ones((224,224,3))
    brightness_increase = ones*intensity
    increased = img+brightness_increase
    
    if np.max(increased)>255:
        increased = np.clip(increased, 0, 255)
    return np.array(increased, dtype='int')



def image_brightness_decrease(img, intensity):
    img = np.array(img)
    ones = np.ones((224,224,3))
    brightness_decrease = ones*intensity
    decreased = img-brightness_decrease
    if np.min(decreased)<0:
        decreased = np.clip(decreased, 0, 255)
    return np.array(decreased, dtype='int')


def hsv_hue_noise_increase(img, std):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)

    _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    randomnoise = np.array(randomnoise)
    h = np.array(h)

    huenoise = h + randomnoise

    if np.max(huenoise)>179:
        for index, row in enumerate(huenoise):
            for undex, value in enumerate(row):
                if value>179:
                    huenoise[index][undex]=value-179

    huenoise = huenoise.astype(np.uint8)
    merged = cv2.merge([huenoise, s, v])
    converted_back = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
    
    return converted_back

def hsv_sat_noise_increase(img, std):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h,s,v = cv2.split(hsv)

    _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    randomnoise = np.array(randomnoise)
    s = np.array(s)

    satnoise = s + randomnoise
    
    if np.max(satnoise)>255 or np.min(satnoise)<0:
        satnoise = np.clip(satnoise, 0, 255)

    uintsatnoise = satnoise.astype(np.uint8)
    merged = cv2.merge([h, uintsatnoise, v])
    converted_back = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
    
    return converted_back


def occlusion(img, edge_length):
    square = np.zeros((edge_length, edge_length, 3))
    random_x = np.random.randint(0, img.shape[0]-edge_length)
    random_y = np.random.randint(0, img.shape[1]-edge_length)
    img2 = img.copy()
    img2[random_x:(random_x+square.shape[0]), random_y:(random_y+square.shape[1])] = square
    
    return img2