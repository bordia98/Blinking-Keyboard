import numpy as np  
import cv2  
import dlib  
from scipy.spatial import distance as dist
from gtts import gTTS 
import os
language = 'en'
import pyttsx3
engine = pyttsx3.init()

characterdict = {'0':'a','00':'d','000':'j','0000':'n','1':'s','01':'f','001':'k','0001':'y','10':'g','010':'l','0010':'t'}
characterdict['0011']='v'
characterdict['011']='m'
characterdict['11']='h'
characterdict['0100']='b'
characterdict['100']='u'
characterdict['0101']='r'
characterdict['101']='i'
characterdict['0110']='e'
characterdict['110']='o'
characterdict['0111']='c'
characterdict['111']='p'
characterdict['1000']='x'
characterdict['1001']='w'
characterdict['1010']='q'
characterdict['1011']='z'
characterdict['1100']=','
characterdict['1101']='.'
characterdict['1110']='?'
characterdict['1111']=" "
print("Enter a choice whether you want blink keyboard or wink keyboard \n 1.) Blink Keyboard \n 2.) Wink keyboard")
n = int(input())
if n==2:
    while True:
        print("You have choosen wink keyboard\n")
        print("Way of using wink keyboard\n")
        print("1.) You will be shown the keyboard structure in front of you\n")
        print("2.)   will move the pointer to left side\n")
        print("3.) Right wink will move the pointer to right side\n")
        print("4.) Blink detected when you here beep sound once will fix your character that you want to choose it\n")
        print("5.) When you hear the beep sound twice while blinking you will be back to the starting position \n")
        print("6.) On the starting node if you blink that means backspace\n")
        print("If you understand the rules press 'y' else 'press 'n' \n")
        check = input()
        if check =='y':
            break
    text = ""
    PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
    stop_flag = 0
    FULL_POINTS = list(range(0, 68))  
    FACE_POINTS = list(range(17, 68))  
    JAWLINE_POINTS = list(range(0, 17))  
    RIGHT_EYEBROW_POINTS = list(range(17, 22))  
    LEFT_EYEBROW_POINTS = list(range(22, 27))  
    NOSE_POINTS = list(range(27, 36))  
    RIGHT_EYE_POINTS = list(range(36, 42))  
    LEFT_EYE_POINTS = list(range(42, 48))  
    MOUTH_OUTLINE_POINTS = list(range(48, 61))  
    MOUTH_INNER_POINTS = list(range(61, 68))  

    EYE_AR_THRESH = 0.23  
    EYE_AR_CONSEC_FRAMES = 5

    counter_left = 0  
    total_left = 0  

    counter_right = 0  
    total_right = 0

    counter_blink = 0
    total_blink = 0

    flag_left,flag_right,flag_blink  = 0,0,0                

    def eye_aspect_ratio(eye):  
        A = dist.euclidean(eye[1], eye[5])  
        B = dist.euclidean(eye[2], eye[4])  
        C = dist.euclidean(eye[0], eye[3])  
        ear = (A + B) / (2.0 * C)  
        return ear  

    detector = dlib.get_frontal_face_detector()  

    predictor = dlib.shape_predictor(PREDICTOR_PATH)  

    video_capture = cv2.VideoCapture(0)
    image = "base"
    text = ""
    while True:  
        ret, frame = video_capture.read()  
        if ret:  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            rects = detector(gray, 0)  
            for rect in rects:  
                x = rect.left()  
                y = rect.top()  
                x1 = rect.right()  
                y1 = rect.bottom()  

                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  

                left_eye = landmarks[LEFT_EYE_POINTS]  
                right_eye = landmarks[RIGHT_EYE_POINTS]  

                left_eye_hull = cv2.convexHull(left_eye)  
                right_eye_hull = cv2.convexHull(right_eye)  
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)  
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)  

                ear_left = eye_aspect_ratio(left_eye)  
                ear_right = eye_aspect_ratio(right_eye)  

                if ear_left >= EYE_AR_THRESH and ear_right >= EYE_AR_THRESH:
                    counter_blink = 0
                    counter_left = 0
                    counter_right = 0

                # print("****************************************")
                # print("Counter Blink : " , counter_blink)
                # print("Counter LEFT : ", counter_left)
                # print("Counter Right : ", counter_right)
                # print("****************************************")
        
                if counter_blink >= 10:
                    if counter_blink == 10:
                        flag_blink = 1
                        duration = 0.05  # seconds
                        freq = 440  # Hz
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    if counter_blink == 20:
                        stop_flag = 1
                        flag_blink = 0
                        duration = 0.05  # seconds
                        freq = 440  # Hz
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                else:
                    if flag_blink == 1:
                        total_blink += 1
                        # print("Blink Occured")
                        counter_blink = 0
                        flag_blink = 0
                    if stop_flag == 1:
                        image = "base"
                        counter_blink = 0
                        flag_blink = 0

                if ear_left < EYE_AR_THRESH:
                    if ear_right < EYE_AR_THRESH :
                        counter_blink += 1
                        counter_left = 0
                    else:
                        counter_blink = 0
                        counter_left += 1  
                        counter_right = 0
                        if counter_left == EYE_AR_CONSEC_FRAMES:
                            flag_left = 1
                            duration = 0.05  # seconds
                            freq = 440  # Hz
                            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) 
                else:
                    if flag_left ==1:
                        total_left += 1
                        # print("Left eye winked")  
                        counter_left = 0
                        counter_blink = 0
                        flag_left = 0  
                        counter_right = 0
                    else:
                        if counter_left >= EYE_AR_CONSEC_FRAMES:  
                            flag_left = 1
                        
                if ear_right < EYE_AR_THRESH:
                    if ear_left < EYE_AR_THRESH: 
                        counter_right = 0 
                        pass
                    else:
                        counter_blink = 0
                        counter_right += 1                         
                        counter_left = 0  
                        if counter_right == EYE_AR_CONSEC_FRAMES:
                            flag_right = 1
                            duration = 0.05  # seconds
                            freq = 440  # Hz
                            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) 
                else:
                    if flag_right == 1:
                        total_right += 1 
                        # print("Right eye winked")  
                        counter_right = 0
                        flag_right = 0
                        counter_blink = 0
                        counter_left = 0
                    else:
                        if counter_right >= EYE_AR_CONSEC_FRAMES:  
                            flag_right = 1
                        
                # if ear_left >= EYE_AR_THRESH :
                #     counter_left = 0
                #     counter_blink = 0

                # if ear_right >= EYE_AR_THRESH:
                #     counter_right = 0
                #     counter_blink = 0


            cv2.putText(frame, "Wink Left : {}".format(total_left), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
            cv2.putText(frame, "Wink Right: {}".format(total_right), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
            cv2.putText(frame, "Blink Occured: {}".format(total_blink), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
            
            if total_left ==  1:
                if image == "base":
                    image = ""
                image+='0'
                total_left = 0
                total_right = 0
                total_blink = 0
                flag_left = 0
                flag_right = 0
                flag_blink = 0
                stop_flag = 0
            
            if total_right == 1:
                if image =="base":
                    image = ""
                image+='1'
                total_right = 0
                total_left = 0
                total_blink = 0
                flag_left = 0
                flag_right = 0
                flag_blink = 0
                stop_flag = 0
            
            if total_blink == 1:
                # print("image is   "+image+".jpg")
                if image!='base':
                    text += characterdict[image]
                else:
                    if len(text)!=0:
                        text = text[:len(text)-1]
                # do the required action
                image = "base"
                total_blink = 0
                total_left = 0
                total_right = 0
                flag_left = 0
                flag_right = 0
                flag_blink = 0
                stop_flag = 0

            if len(image)>4:
                image=image[:4]
            cv2.namedWindow("KeyBoard", cv2.WINDOW_NORMAL)
            cv2.moveWindow("KeyBoard",850,20)      
            ia = cv2.imread(image+".jpg")
            ims = cv2.resizeWindow("KeyBoard",550, 400)                    # Resize image
            cv2.imshow("KeyBoard" , ia)
            cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)   
            cv2.moveWindow("Faces",0,20)         
            ims = cv2.resizeWindow("Faces",800, 700)                    # Resize image 
            cv2.imshow("Faces", frame) 
            cv2.namedWindow("Typed_Text", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Typed_Text",850,500)      
            draw = cv2.imread("draw.jpg")
            cv2.resizeWindow("Typed_Text",550,270)
            cv2.putText(draw, "Typed Text: {}".format(text), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)  
            cv2.imshow("Typed_Text" , draw)
        ch = 0xFF & cv2.waitKey(1)  
        if ch == ord('q'):  
            break  
    cv2.destroyAllWindows() 
elif n==1:
    while True:
        print("You have choosen Blink keyboard")
        print("Way of using Blink keyboard\n")
        print("1.) You will be shown the keyboard structure in front of you\n")
        print("2.) Shorter blink: When you hear a beep sound first time, will move the pointer to left side\n")
        print("3.) Longer blink: When you hear a beep  sound second time, will move the pointer to right side\n")
        print("4.) Longest Blink: When you hear a beep sound third time, will fix your character that you want to choose it\n")
        print("5.) Back to start: When you hear the beep sound 4th time with writing character\n")
        print("6.) On the starting node if you blink that means backspace\n")
        print("If you understand the rules press 'y' else 'press 'n' \n")
        check = input()
        if check =='y':
            break
        

    text = ""
    PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

    FULL_POINTS = list(range(0, 68))  
    FACE_POINTS = list(range(17, 68))  
    JAWLINE_POINTS = list(range(0, 17))  
    RIGHT_EYEBROW_POINTS = list(range(17, 22))  
    LEFT_EYEBROW_POINTS = list(range(22, 27))  
    NOSE_POINTS = list(range(27, 36))  
    RIGHT_EYE_POINTS = list(range(36, 42))  
    LEFT_EYE_POINTS = list(range(42, 48))  
    MOUTH_OUTLINE_POINTS = list(range(48, 61))  
    MOUTH_INNER_POINTS = list(range(61, 68))  

    EYE_AR_THRESH = 0.25  
    EYE_AR_CONSEC_FRAMES = 5

    counter_blink = 0
    total_blink = 0
    '''
    There are three types of blink
    one blink --- Left blink
    two blink --- Right blink
    three blink --- Select the letter
    four blink --- Revert to start 
    '''
    flag_blink_one,flag_blink_two,flag_blink_three,stopflag  = 0,0,0,0              
    count_left,count_right,count_stop = 0,0,0
    def eye_aspect_ratio(eye):  
        A = dist.euclidean(eye[1], eye[5])  
        B = dist.euclidean(eye[2], eye[4])  
        C = dist.euclidean(eye[0], eye[3])  
        ear = (A + B) / (2.0 * C)  
        return ear  

    detector = dlib.get_frontal_face_detector()  

    predictor = dlib.shape_predictor(PREDICTOR_PATH)  

    video_capture = cv2.VideoCapture(-1)
    image = "base"
    text = ""
    while True:  
        ret, frame = video_capture.read()  
        if ret:  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            rects = detector(gray, 0)  
            for rect in rects:  
                x = rect.left()  
                y = rect.top()  
                x1 = rect.right()  
                y1 = rect.bottom()  

                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  

                left_eye = landmarks[LEFT_EYE_POINTS]  
                right_eye = landmarks[RIGHT_EYE_POINTS]  

                left_eye_hull = cv2.convexHull(left_eye)  
                right_eye_hull = cv2.convexHull(right_eye)  
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)  
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)  

                ear_left = eye_aspect_ratio(left_eye)  
                ear_right = eye_aspect_ratio(right_eye)  

                # print("****************************************")
                # print("Counter Blink : " , counter_blink)
                # print("****************************************")
        
                if counter_blink >= 10:
                    if counter_blink == 10:
                        flag_blink_one,flag_blink_two,flag_blink_three = 1,0,0
                        stopflag = 0
                        duration = 0.05  # seconds
                        freq = 440  # Hz
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    if counter_blink == 20:
                        flag_blink_two,flag_blink_one,flag_blink_three = 1,0,0
                        stopflag = 0
                        duration = 0.05  # seconds
                        freq = 440  # Hz
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    if counter_blink == 30:
                        flag_blink_three,flag_blink_one,flag_blink_two = 1,0,0
                        stopflag = 0
                        duration = 0.05  # seconds
                        freq = 440  # Hz 
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    if counter_blink==50:
                        stopflag = 1
                        flag_blink_three,flag_blink_one,flag_blink_two = 0,0,0
                        duration = 0.05  # seconds
                        freq = 440  # Hz 
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                else:
                    if flag_blink_three == 1:
                        total_blink += 1
                        # print("Stop Blink Occured")
                        counter_blink = 0
                        count_stop = 1
                        flag_blink_one,flag_blink_two,flag_blink_three  = 0,0,0
                        count_left = 0
                        count_right = 0             
                    elif flag_blink_one == 1:
                        total_blink += 1
                        # print("Left side blink occured")
                        counter_blink = 0
                        flag_blink_one,flag_blink_two,flag_blink_three  = 0,0,0
                        count_left = 1
                        count_right = 0
                        count_stop = 0
                    elif flag_blink_two == 1:
                        total_blink += 1
                        # print("Right side blink occured")
                        counter_blink = 0
                        flag_blink_one,flag_blink_two,flag_blink_three  = 0,0,0
                        count_left = 0
                        count_right = 1
                        count_stop = 0
                    elif stopflag == 1:
                        count_left,count_right,count_stop=0,0,0
                        stopflag = 0
                        image = 'base'
                if ear_left < EYE_AR_THRESH and ear_right < EYE_AR_THRESH:
                    counter_blink += 1
                else:
                    counter_blink = 0

            cv2.putText(frame, "Blink Occured: {}".format(total_blink), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
            if count_left ==  1:
                if image == "base":
                    image = ""
                image+='0'
                count_left = 0
            
            if count_right == 1:
                if image =="base":
                    image = ""
                image+='1'
                count_right = 0
            
            if count_stop == 1:
                if image == "base":
                    if len(text)!=0:
                        text = text[:len(text)-1]
                    # myobj = gTTS(text="backspace", lang=language, slow=False) 
                    # myobj.save("text.mp3") 
                    engine.say("Backspace")
                    engine.runAndWait()
                else:
                    text += characterdict[image]
                    # myobj = gTTS(text=characterdict[image], lang=language, slow=False) 
                    # myobj.save("text.mp3") 
                    engine.say(characterdict[image])
                    engine.runAndWait()
                # print("image is   "+image+".jpg")
                # do the required action
                # os.system("mpg321 text.mp3") 
                image = "base"
                count_stop,count_left,count_right = 0,0,0
            if len(image)>4:
                image=image[:4]
        
            cv2.namedWindow("KeyBoard", cv2.WINDOW_NORMAL)
            cv2.moveWindow("KeyBoard",850,20)      
            ia = cv2.imread(image+".jpg")
            ims = cv2.resizeWindow("KeyBoard",550, 400)                    # Resize image
            cv2.imshow("KeyBoard" , ia)
            cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)   
            cv2.moveWindow("Faces",0,20)         
            ims = cv2.resizeWindow("Faces",800, 700)                    # Resize image 
            cv2.imshow("Faces", frame) 
            cv2.namedWindow("Typed_Text", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Typed_Text",850,500)      
            draw = cv2.imread("draw.jpg")
            cv2.resizeWindow("Typed_Text",550,270)
            cv2.putText(draw, "Typed Text: {}".format(text), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)  
            cv2.imshow("Typed_Text" , draw)
        ch = 0xFF & cv2.waitKey(1)  
        if ch == ord('q'):  
            break  
    cv2.destroyAllWindows() 
else:
    print("You entered wrong choice ")
    exit(0)    
        
