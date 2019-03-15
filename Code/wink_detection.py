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
EYE_AR_CONSEC_FRAMES = 3  

COUNTER_LEFT = 0  
TOTAL_LEFT = 0  

COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0

def eye_aspect_ratio(eye):  
   A = dist.euclidean(eye[1], eye[5])  
   B = dist.euclidean(eye[2], eye[4])  
   C = dist.euclidean(eye[0], eye[3])  
   ear = (A + B) / (2.0 * C)  
   return ear  

detector = dlib.get_frontal_face_detector()  

predictor = dlib.shape_predictor(PREDICTOR_PATH)  

video_capture = cv2.VideoCapture(-1)

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

            cv2.putText(frame, "E.A.R. Left : {:.2f}".format(ear_left), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
            cv2.putText(frame, "E.A.R. Right: {:.2f}".format(ear_right), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  

            if ear_left < EYE_AR_THRESH:  
                COUNTER_LEFT += 1  
            else:  
                if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:  
                    TOTAL_LEFT += 1  
                    print("Left eye winked")  
                COUNTER_LEFT = 0  

            if ear_right < EYE_AR_THRESH:  
                COUNTER_RIGHT += 1  
            else:  
                if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:  
                    TOTAL_RIGHT += 1  
                    print("Right eye winked")  
                COUNTER_RIGHT = 0  

        cv2.putText(frame, "Wink Left : {}".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
        cv2.putText(frame, "Wink Right: {}".format(TOTAL_RIGHT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  

        cv2.imshow("Faces found", frame)  

    ch = 0xFF & cv2.waitKey(1)  
    cv2.waitKey(10)
    if ch == ord('q'):  
        break  

cv2.destroyAllWindows() 
