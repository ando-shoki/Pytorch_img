from PIL import Image
from torchvision import transforms

def gamma(img, gamma=1.8):
    r_lut =  [int(255 *(float(i)/255)**(1.0/gamma)) for i in range(256)]
    g_lut =  [int(255 *(float(i)/255)**(1.0/gamma)) for i in range(256)]
    b_lut =  [int(255 *(float(i)/255)**(1.0/gamma)) for i in range(256)]
    lut =  r_lut + g_lut + b_lut 
    
    return img.point(lut)