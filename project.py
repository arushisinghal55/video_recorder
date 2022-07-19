import cv2
import time
import datetime

motion = False
first_frame = None
video = cv2.VideoCapture(0)

frame_width = int(video.get(3))
frame_height = int(video.get(4))

capture_duration = 25
count = 0

while True:
    check, frame = video.read()

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
        if cv2.contourArea(contour) < 10000:
            continue
        else:
            if not motion:
                output = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
                out = cv2.VideoWriter(output + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
                start_time = time.time()
            motion = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Color Frame", frame)

    if motion:
        if int(time.time() - start_time) < capture_duration:
            out.write(frame)
            count = count + 1
        else:
            motion = False
            count = 0
            out.release()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
