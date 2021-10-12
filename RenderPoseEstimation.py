import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/bamba5.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('ballin.mp4',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    result.write(img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)


cap.release()
result.release()

cv2.destroyAllWindows()

print("the video was successfully saved")
