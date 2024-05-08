import cv2
import torch
import numpy as np

from ultralytics import YOLO
from collections import defaultdict
import time


torch.cuda.empty_cache()

model = YOLO('s_860i_m50-0,67(0,75).pt')

video_path = "D:/for YOLO/Root111/domodedovo_test.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
cur_time = time.time() * 1000
buf_time = None
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    buf_time = cur_time
    cur_time = time.time() * 1000

    if success:
        # Run YOLOv8 inference on the frame
        time_res_1 = time.time() * 1000
        # frame = cv2.resize(frame, (640, 640))
        results = model.track(frame, conf=0.6, imgsz=640, persist=True, device='cuda')
        time_res_2 = time.time() * 1000

        time_box_1 = time.time() * 1000
        boxes = results[0].boxes.xywh.cpu()
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = results[0].boxes.id
        time_box_2 = time.time() * 1000

        # Visualize the results on the frame
        time_vis_1 = time.time() * 1000
        annotated_frame = results[0].plot(font_size=2, line_width=1)
        # all_tracks = [np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) for track in track_history.values()]
        # cv2.polylines(annotated_frame, all_tracks, isClosed=False, color=(230, 230, 230), thickness=1)
        time_vis_2 = time.time() * 1000
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    print(f"1 cycle: {cur_time - buf_time}\nResults: {time_res_2 - time_res_1}\n Boxes: {time_box_2 - time_box_1}\n Visualise: {time_vis_2 - time_vis_1}")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()