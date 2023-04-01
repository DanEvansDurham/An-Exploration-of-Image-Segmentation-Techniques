from PIL import Image
from numpy import asarray
import csv

def img_data(file_name, scale):
    """ Opens the image file, converts the image to greyscale and converts 
        it into a csv which is saved. Also reduces it by the 
        scaling factor 'scale'. 
        
        Inputs:
            file_name:  the name of the file to be segmented
            scale:      the scaling factor to reduce the image size by
    """
    
    image = Image.open('{}.jpg'.format(file_name)).convert('L')
    data_old = asarray(image)
    shape = data_old.shape
    new_shape = tuple([round(shape[1]*scale),round(shape[0]*scale)])
    print("Old image shape: ", shape)
    print("New image shape: ", new_shape)
    image = image.resize(new_shape,Image.ANTIALIAS)
    data = asarray(image)
    
    f = open('{}_data.csv'.format(file_name), 'w', newline = '')

    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    
    f.close()
    
