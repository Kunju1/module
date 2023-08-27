#Reading image
import cv2 as cv
import numpy as np
def read_img(img_path):
    img=cv.imread(img_path)
    cv.imshow('IMAGE', img)
    cv.waitKey(0)

#Reading video   
def read_video(video_path):
    capture = cv.VideoCapture(video_path)
    while True:
        isTrue, frame = capture.read()
        cv.imshow('Video', frame)
        if cv.waitKey(20)&0xFF==ord('d'):
            break
    capture.release()
    cv.destroyAllWindows

#Resize 
def rescale_image(image_path, scale=0.75):
    frame = cv.imread(image_path)
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    resized_image = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow('Resized Image', resized_image)
    cv.waitKey(0)

#Rescale
def rescale_video(video_path, scale=0.75):
    capture = cv.VideoCapture(video_path)
    while True:
        isTrue, frame=capture.read()
        
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        resized_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        cv.imshow('Video', resized_frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()

#Paint
blank = np.zeros((500,500,3), dtype='uint8')
def paint_image(color):
    #blank = np.zeros((500,500,3), dtype='uint8')
    blank[200:300,300:400] = 0,255,0
    cv.imshow ('Color Filled', blank)
    cv.waitKey(0)

#Draw Rectangle
def draw_rectangle(color, outline):
    #blank = np.zeros((500,500,3), dtype='uint8')
    cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), color, thickness=outline)
    cv.imshow('Rectangle', blank)
    cv.waitKey(0)

#Draw Circle
def draw_circle(color, outline, radius):
    #blank = np.zeros((500,500,3), dtype='uint8')
    cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), radius, color, thickness=outline)
    cv.imshow('Circle', blank)
    cv.waitKey(0)

#Draw Line
def draw_line(color, outline):
    #blank = np.zeros((500,500,3), dtype='uint8')
    cv.line(blank, (0,0), (300,400), color, outline)
    cv.imshow('Line', blank)
    cv.waitKey(0)

#Put Text
def write_text(text, color, outline):
    #blank = np.zeros((500,500,3), dtype='uint8')
    cv.putText(blank, text, (0,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, outline)
    cv.imshow('Text', blank)
    cv.waitKey(0)

#Image translation
def trans_image(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols,rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Image Reflection 
def reflect_image(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0],
    [0, -1, rows],
    [0, 0, 1]])
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    cv.imshow('img', reflected_img)
    cv.imwrite('reflection_out.jpg', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Image Rotation
def rotate_image(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
    img_rotation = cv.warpAffine(img,
    cv.getRotationMatrix2D((cols/2, rows/2), 30, 0.6),(cols, rows))
    cv.imshow('img', img_rotation)
    cv.imwrite('rotation_out.jpg', img_rotation)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Image Scaling
def scale_image(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    img_shrinked = cv.resize(img, (250, 200),
    interpolation=cv.INTER_AREA)
    cv.imshow('img', img_shrinked)
    img_enlarged = cv.resize(img_shrinked, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Image Cropping
def crop_image(img_path):
    img = cv.imread(img_path, 0)
    cropped_img = img[100:300, 
    100:300]
    cv.imwrite('cropped_out.jpg', 
    cropped_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Image Shearing in X-Axis
def shearx_img(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], 
    [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, 
    (int(cols*1.5), int(rows*1.5)))
    cv.imshow('img', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
#Image Shearing in Y-Axis
def sheary_img(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0.5, 1, 0], 
    [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M,
    (int(cols*1.5), int(rows*1.5)))
    cv.imshow('sheared_y-axis_out.jpg'
    , sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Color Channels
def color_img(img_path):
    image = cv.imread(img_path)
    B, G, R = cv.split(image)
# Corresponding channels are separated

    cv.imshow("original", image)
    cv.waitKey(0)

    cv.imshow("blue", B)
    cv.waitKey(0)

    cv.imshow("Green", G)
    cv.waitKey(0)

    cv.imshow("red", R)
    cv.waitKey(0)

    cv.destroyAllWindows()

#Kernal Blur
def kerblur_image(img_path):
    image = cv.imread(img_path)
    # Creating the kernel with numpy
    kernel2 = np.ones((5, 5), np.float32)/25
    # Applying the filter
    img = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
    # showing the image
    cv.imshow('Original', image)
    cv.imshow('Kernel Blur', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
#Average Blur
def avblur_image(img_path):
    image = cv.imread(img_path)
    # Applying the filter
    averageBlur = cv.blur(image, (5, 5))
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Average blur', averageBlur)
    cv.waitKey(0)
    cv.destroyAllWindows()
#Gaussian Blur
def gausblur_image(img_path):
    image = cv.imread(img_path)
    # Applying the filter
    gaussian = cv.GaussianBlur(image, (3, 3), 0)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Gaussian blur', gaussian)
    cv.waitKey(0)
    cv.destroyAllWindows()
#Median Blur
def medblur_image(img_path):
    image = cv.imread(img_path)
    # Applying the filter
    medianBlur =cv.medianBlur(image, 9)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Median blur',
    medianBlur)
    cv.waitKey(0)
    cv.destroyAllWindows()
#Bilateral Blur
def bilblur_image(img_path):
    image = cv.imread(img_path)
    # Applying the filter
    bilateral = cv.bilateralFilter(image,
    9, 75, 75)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Bilateral blur', bilateral)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
#Masking
def mask_image(img_path):
    img = cv.imread(img_path)
    cv.imshow('Original image', img)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    cv.imshow('Blank Image', blank)
    circle = cv.circle(blank,
    (img.shape[1]//2,img.shape[0]//2),200,255, -1)
    cv.imshow('Mask',circle)
    masked = cv.bitwise_and(img,img,mask=circle)
    cv.imshow('Masked Image', masked)
    cv.waitKey(0)













 

