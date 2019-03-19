import numpy as np  
import cv2  
import dlib  
from scipy.spatial import distance as dist


PREDICTOR_PATH = "/home/bordia98/eyeblink/Code/shape_predictor_68_face_landmarks.dat"

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

            print("****************************************")
            print("Counter Blink : " , counter_blink)
            print("Counter LEFT : ", counter_left)
            print("Counter Right : ", counter_right)
            print("****************************************")
    
            if counter_blink >= 10:
                flag_blink = 1
            else:
                if flag_blink == 1:
                    total_blink += 1
                    print("Blink Occured")
                    counter_blink = 0
                    flag_blink = 0

            if ear_left < EYE_AR_THRESH:
                if ear_right < EYE_AR_THRESH:
                    counter_blink += 1
                else:
                    counter_blink = 0
                    counter_left += 1  
            else:  
                if counter_left >= EYE_AR_CONSEC_FRAMES:  
                    flag_left = 1
                else:
                    if flag_left ==1:
                        total_left += 1  
                        print("Left eye winked")  
                    counter_left = 0
                    flag_left = 0  

            if ear_right < EYE_AR_THRESH:
                if ear_left < EYE_AR_THRESH:  
                    pass
                else:
                    counter_blink = 0
                    counter_right += 1  
            else:  
                if counter_right >= EYE_AR_CONSEC_FRAMES:  
                    flag_right = 1
                else:
                    if flag_right == 1:
                        total_right += 1  
                        print("Right eye winked")  
                    counter_right = 0  
                    flag_right = 0

            if ear_left >= EYE_AR_THRESH :
                counter_left = 0
                counter_blink = 0

            if ear_right >= EYE_AR_THRESH:
                counter_right = 0
                counter_blink = 0


        cv2.putText(frame, "Wink Left : {}".format(total_left), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
        cv2.putText(frame, "Wink Right: {}".format(total_right), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
        cv2.putText(frame, "Blink Occured: {}".format(total_blink), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
        if total_left ==  1:
            if image == "base":
                image = ""
            image+='0'
            total_left = 0
        
        if total_right == 1:
            if image =="base":
                image = ""
            image+='1'
            total_right = 0
        
        if total_blink == 1:
            print("image is   "+image+".jpg")
            # do the required action
            image = "base"
            total_blink = 0
            total_left = 0
            total_right = 0
        cv2.namedWindow("KeyBoard", cv2.WINDOW_NORMAL)      
        ia = cv2.imread(image+".jpg")
        ims = cv2.resize(ia, (700, 400))                    # Resize image
        cv2.imshow("KeyBoard " , ims)
        cv2.imshow("Faces found", frame)  
        
    ch = 0xFF & cv2.waitKey(1)  
    if ch == ord('q'):  
        break  
cv2.destroyAllWindows() 
