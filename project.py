import cv2
import time
import datetime
from queue import Queue

motion = False
first_frame = None
video = cv2.VideoCapture(0)  # video capturing starts

frame_width = int(video.get(3))
frame_height = int(video.get(4))

# to record the video for 25 seconds after motion is detected
capture_duration = 25
count = 0

# starts recording 5 sec before the motion is detected
maxQSize = 150  # 5*30 frames per second

qFrames = Queue(maxsize=maxQSize)  # queue to store the frames as cached data to write it in the video later
recording_open = False

while True:
    check, frame = video.read()  # reading of a frame at a time
    if not recording_open:

        if qFrames.full():  # if queue is full
            qFrames.get()   # pop one frame at a time
        qFrames.put(frame)  # put one extra frame after pop function

    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=0)

    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:  # to check if the motion is detected
            continue
        else:
            if not motion:
                output = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
                out = cv2.VideoWriter(output + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
                start_time = time.time()  # timestamp variable to starting time of the video

            motion = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Color Frame", frame)

    if motion:
        if not recording_open:
            recording_open = True

        # after motion is detected first writing the queue frames in the video (5 sec)
            while not qFrames.empty():
                out.write(qFrames.get())
            start_time = time.time()

        # writing the remaining video in the video file (25 sec)
        if int(time.time() - start_time) < capture_duration:
            out.write(frame)
            count = count + 1
        else:
            motion = False
            count = 0
            out.release()
            recording_open = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
