# Air-Notability-with-ML
when you can draw in the air and the computer understands

Python 3.6
numpy
Keras 2.1.5
cv2
tensorflow 1.8

How to use:

--Take an orange highlighter, and run test.py
--Draw a letter in front of you
--Press E to erase, Press
--Press D when you are done
--The letter you drew will appear on the screen
--Rinse and repeat

How it works:
--First, the computer tracks your orange highlighter based on its colors and draws its path
--Next, the computer crops out the path of the highlighter, does some image processing, and
feeds it into a convolutional neural network
--The CNN spits out a letter, A-Z
--The CNN is trained using the EMNIST dataset
