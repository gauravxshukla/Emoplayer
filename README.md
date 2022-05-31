# Emoplayer (Emotion-Based Media Recommendation System)

In today’s world, digital media is a significant element of human life. People take its help to stay motivated and
explore media according to their moods. It takes a lot of effort to find appropriate music that suits the particular
emotional state from loads of options available. Media players in today’s world are not giving priority to the
emotional state and effective recommendation of a person. Human emotion plays a vital role in media selection
in recent times. Emotion expresses the individual’s behavior and state of mind and digital media has the power
to change one’s mental state from negative to positive. The objective of this paper is to extract features from
the human face and detect emotion, age, and gender, and suggest media according to the features detected. The
emotional state, age, and gender can be interpreted from facial expressions through the webcam. We have used
the CNN classifier to build a neural network model. This model is trained and subjected to detect mood, age, and
gender from facial expressions using OpenCV. A system that generates a media playlist based on the detected
emotion, age, and gender gives better results.


# Proposed Solution - 

Generally, humans tend to show their emotions through facial expressions. The proposed system
helps us to provide an interaction between the user and the media system. It aims to provide userpreferred
media which best suits one’s current emotional state. This work is based on the idea of
automating much of the interaction between the media player and its user. The system is a web
application that runs with real-time webcam support. Emotion, age, and gender are recognized
and depending upon the mood of the user the system invokes the media playlist containing media
suggestion that matches user’s age and gender. The system consists of 5 emotions on which media
recommendation is possible, 2 genders, and 7 age categories which are categorized by analyzing
the change of liking of the people from different age groups. The proposed system also gives the
user functionality to personalize his different emotion playlist by adding or deleting songs/videos.
The emotion detection model is trained with data set containing more than 30000 images for each
emotion and, the age and gender dataset contains around 25000 images which add high accuracy
to the model.

# Architectural Design Of Emo Player
![image](https://user-images.githubusercontent.com/82375003/171281100-c4be1898-2f22-4657-a2a0-3bd95ffb2749.png)


# Results - 

![image](https://user-images.githubusercontent.com/82375003/171281216-f76519d5-4f5b-48bf-a862-75f12d62df6f.png)  ![image](https://user-images.githubusercontent.com/82375003/171281243-ba16a5d1-fcaf-4448-95f3-a2b967be806b.png)

# Model Accuracy - 

![1](https://user-images.githubusercontent.com/82375003/171281563-53d75899-b512-4b1b-9792-465a9622df09.PNG)


Research Paper - https://ijngc.perpetualinnovation.net/index.php/ijngc/article/view/415

Tech Stack Used - OpenCV, HaarCascade, CNN, Django, Bootstrap, HTML, CSS, JS.
