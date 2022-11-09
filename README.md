# Ikigai

# DEMO
 
This demo covers two main areas: face recognition and body recognition.

Face: Bounding Box detection, database-based tracking, acquisition of facial landmarks, facial expression recognition, and storage of respective information for each ID.

Body: Bounding Box detection, tracking using database, acquisition of body landmarks, acquisition of body information using biomechanics, storage of respective information for each ID
 
# Requirement
 
* Python 3.8
* This environment can be built using .yml file.
 
Environments under [Anaconda for Windows](https://www.anaconda.com/distribution/) is tested.
 
# Usage

Run "Python3 0 - DatabaseProcessing.py": That is used to create database for both face and body
Run "Python3 1 - VideoProcessing - Box - ID.py": That is used to obtain and store BBox and ID information
Run "Python3 2 - VideoProcessing - MediaPipe.py": That is used to obtain and store landmark on both face and body
Run "Python3 3 - VideoProcessing - Biomechanics.py": That is used to obtain and store biomechanics information from body
Run "Python3 4 - VideoProcessing - Emotion.py": That is used to obtain emotion information from face

# Note
 
I don't test environments under Linux and Mac.
