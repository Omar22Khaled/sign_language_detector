#import pickle
import os
from gtts import gTTS
from playsound import playsound 

#from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'C', 2: 'W', 3:'L' , 4:'H', 5:'I'}

word_list = [""]
text = " "

last_detected = datetime.now()
predicted_character = ""
while True:    

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
   

    #text = word_list[0]+"".join([word_list[i] for i in range(1,len(word_list)) if word_list[i]!= word_list[i-1]])
    text = "".join(word_list)
    cv2.putText(frame, "Detected Output is: "+text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
            cv2.LINE_AA)    
        
    

    
    
    
    
    
    
    
    
    
    if cv2.waitKey(1) & 0xFF==ord('a') :     # appending a letter to the word list 
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        word_list.append(predicted_character)
        
    if cv2.waitKey(1) & 0xFF==ord('s'):     # appending a space to the word list to separate the words
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, "Dilimeter", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        word_list.append(" ")
        

#     #else:
#     #    cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) & 0xFF==ord('d'):    # display the whole sentence
#         print(word_list)
#         text ="".join(word_list)
#         cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
        
#         g = gTTS(text=text , lang= 'en')
#         g.save('audio.mp3')
#         playsound('audio.mp3')

#         # removing unused files.
#         os.remove('audio.mp3')
#         word_list = []
    
     
    if cv2.waitKey(1) & 0xFF==ord('w'):     
        text = word_list[0]+"".join([word_list[i] for i in range(1,len(word_list)) if word_list[i]!= word_list[i-1]])+"L"
        cv2.putText(frame, f"Detected Output is: {text}", (20, 100 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3,
                cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF==ord('q'):   #quitting the webcam
        cap.release()
        break
        
    
    
    cv2.imshow('frame', frame)
g = gTTS(text=text , lang= 'en')
g.save('audio.mp3')
playsound('audio.mp3')

# removing unused files.

word_list = []    
    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

os.remove('audio.mp3')


#text = "".join(word_list)
#cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
            #cv2.LINE_AA)

    
    
    
    
    
# g = gTTS(text=text , lang= 'en')
# g.save('audio.mp3')
# playsound('audio.mp3')

# # removing unused files.
# os.remove('audio.mp3')
# word_list = []