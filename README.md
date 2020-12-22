# Air-Notability-with-ML

A Deep Learning/Image Processing project created for Conant Hack Club.

## Prerequisites ##

* Python 3.6
```
numpy
Keras 2.1.5
cv2 (OpenCV)
tensorflow 1.8
```

## Download ##

Download a zipped version of the project and locate the directory. Then, using an IDE or command line, run the test.py file. 
A window should now pop up with the program running on it.

## How to use: ##

1. Take an orange highlighter, and run test.py
2. Draw a letter in front of you
3. Press E to erase, Press
4. Press D when you are done
5. The letter you drew will appear on the screen
6. Rinse and repeat
7. If you want to change the highlighter color, find the HSV of the chosen highlighter and update the mask values in test.py

## How it works: ##
1. First, the computer tracks your orange highlighter based on its colors and draws its path. It does this by using contours and masks.
2. Next, the computer crops out the path of the highlighter using contours, resizes and reformats the colors, and
feeds it into a convolutional neural network
3. The CNN spits out a letter, A-Z
4. Prior to all of this, the CNN was trained using the EMNIST dataset
