#get all the imports

import numpy as np
from keras import backend as K
from keras.models import model_from_yaml
import cv2

K.set_image_dim_ordering('th') #sets depth, input_depth, rows, columns for the convolutional neural network

#start by loading the neural network - machine learning model
yaml_file = open('models/letter.yaml', 'r')
letter_model_yaml = yaml_file.read()
yaml_file.close()
letter_model = model_from_yaml(letter_model_yaml) #load the architecture
# load weights into new models
letter_model.load_weights("models/letter.h5")

letter_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compile the model with the correct optimizer - adam

word = ""

def getLetter(w, h, points):
    global letter_model
    # now we want to take all the points, but it on a blank image, and run it through our neural network

    sample = np.zeros([w, h, 3], dtype=np.uint8)  # gets a blank image

    sample.fill(255)  # turns it white

    if len(points) == 0: #nothing drawn
        return ""

    # draws the points on the image
    for pt in points:
        sample = cv2.circle(sample, (h - pt[0], pt[1]), 15, (0, 0, 0), -1)

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)  # converts to greyscale

    ret, thresh = cv2.threshold(gray, 127, 255,
                                cv2.THRESH_BINARY_INV)  # applies a threshold - basically, turns all colors past a certain number white
    # and the rest are black

    # some of the points may not be connected - this ensures they don't get cropped out
    kernel = np.ones((15, 15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # finds a coutour based around the points
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)  # gets the x, y, width, and height of the bounding box

    portion = sample[y:y + h, x:x + w]  # crops the sample image to get the letter

    flipped = cv2.bitwise_not(portion)  # makes the background black

    flipped = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)  # goes from 3 channels to 1 channel (greyscale)

    # when inputting an image into the neural network, it must be flipped and rotated 90 degrees counterclockwise
    # this is because the training data was formatted this way
    flipped = cv2.flip(flipped, 1)
    flipped = cv2.rotate(flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    letter = cv2.resize(flipped, (28, 28))  # resize the letter to 28 * 28
    letter = letter.reshape((1, 1, 28, 28)).astype("float32") / 255  # reformat and normalize

    #predict
    result = letter_model.predict(letter)  # get the prediction

    result = result[0]  # get the output vector

    max = np.where(result == np.amax(result))  # finds the index of the max value in the output vector - this will get us the letter
    return chr(64 + max[0][0])  # gets the index, adds 64 to it since A is 1 and A is 65 in ASCII


def startVideo():
    global word
    global cap

    points = [] #the points on the screen that represent the tip of the marker
    adding = False #if you are currently adding points - think as if the marker is on the page
    erasing = False #eraser
    #width, height, and channels of the video frame
    w = 0
    h = 0
    c = 0

    while True: #keep a livestream going
        ret, frame = cap.read() #get the current frame

        result = frame.copy() #get a copy of the grame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert RGB to HSV (another color format)

        #we want to find all the orange on the screen; this part takes a lower bound and upper bound for orange colors,
        #and then finds all colors in between
        lower = np.array([0, 128, 234]) #dark orange in HSV
        upper = np.array([34, 183, 255]) #light orange in HSV
        mask = cv2.inRange(image, lower, upper) #gets a mask - basically, all the orange will be white, and everything else will be black
        result = cv2.bitwise_and(result, result, mask=mask) #bitwise operation, now makes all the orange stay orange, and all the black black

        #finds countours - a curve that joins all colors of the shape shape
        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2] #gets countours

        #takes the countours and draws bounding boxes around the orange parts on the screen
        if len(cnts) > 0:
            area = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(area) #x, y, width, height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #draw a green rectangle
            if adding: #if the marker is on the page, we want to add it to points
                points.append([int((2*x + w)/2), int((2*y+h)/2)])
            elif erasing: #if you are erasing
                e_x = int((2*x + w)/2) #gets the x point of the eraser
                e_y = int((2*y+h)/2) #gets the y point
                d = 25 #distance
                points = [x for x in points if not (abs(e_x - x[0]) < d and abs(e_y - x[1]) < d)] #list comprehension: if a point is within 25 pixels
                #of the eraser's point, it won't be kept in points


        w, h, c = frame.shape #updates w, h, and c

        for pt in points: #draws every point as a circle on the page
            frame = cv2.circle(frame, (pt[0], pt[1]), 15, (0, 255, 0), -1)

        #add text to the screen

        frame = cv2.flip(frame, 1) #flip the frame first, because the camera is mirrored

        frame = cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,127,255) , 2, cv2.LINE_AA)

        cv2.imshow('AIR NOTABILITY WITH ML', frame) #shows the frame

        #key press
        k = cv2.waitKey(1)

        if k == ord('a'): #a - lift up the marker
            adding = not adding
            if adding:
                erasing = False
        if k == ord('b'): #backspace
            if len(word) > 0: word = word[0:len(word)-1]
        if k == ord('s'): #space
            word += " "
        elif k == ord('e'): #a - lift up the marker
            erasing = not erasing
            if erasing:
                adding = False
        elif k == ord('c'): #clear the page
            points = []
        elif k == ord('d'): #all done
            break
        elif k == ord('q'):
            exit(-1)


    word += getLetter(w, h, points) #get the letter and add onto the word
    print(word)
    startVideo()

cap = cv2.VideoCapture(0) #start a live stream
startVideo()
