# Air-Notability-with-ML

## when you can draw in the air and the computer understands ##

* Python 3.6
* numpy
* Keras 2.1.5
* cv2
* tensorflow 1.8

## How to use: ##

1. Take an orange highlighter, and run test.py
2. Draw a letter in front of you
3. Press E to erase, Press
4. Press D when you are done
5. The letter you drew will appear on the screen
6. Rinse and repeat
7. If you want to change the highlighter color, find the HSV of the chosen highlighter and update the mask values in test.py

## How it works:##
1. First, the computer tracks your orange highlighter based on its colors and draws its path
2. Next, the computer crops out the path of the highlighter, does some image processing, and
feeds it into a convolutional neural network
3. The CNN spits out a letter, A-Z
4. The CNN is trained using the EMNIST dataset
