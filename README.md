Hello everyone and welcome to my this side of the internet :)
This repository is an improved version based on the tutorial by Nicholas Rossetti (Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model -
https://www.youtube.com/watch?v=doDUihpj6ro&t=1016s). 

In addition to the code provided by the tutorial, I made the recognition algorithm more stable by using the wrist landmark as a reference point for landmark position, 
thus making the position estimation accuracy less dependant on the distance from the camera.
I found that the program was also a bit too confident and sometimes produced multiple outputs for the same gesture, so I added a few lines to increase confidence before 
providing an answer: the answer has to be consistent for a few seconds before it can be considered valid.

In this repository you will find multiple .py files which are expected to be run in order. The ASL_notebook.ipynb file contains all the sequences from beginning to the end. However, I recommend to go by single files for readibility and to save time.

Future developments for improving user experience will include plotting the evaluation results having a real table for the confusion matrix and a graph depicting accuracy over time.

In the long term future future I also plan to improve the accuracy of the model by testing different types of machine learning approaches to find the most accurate combination and dive deeper into
the syntax of the american sign language in order to provide a more reliable tool to whoever may benefit from it!

