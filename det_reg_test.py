import asyncio
import json
import time
import cv2
import numpy as np

from ultralytics import YOLO
from asyncstdlib import enumerate as as_enumerate

model = YOLO("s_860i_m50-0,67(0,75).pt")  # Инициализация модели

cap = cv2.VideoCapture("D:/for YOLO/Root111/domodedovo_test.mp4")  # Инициализация видео

# Информация о видео
w_f, h_f, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))
print("Параметры видео: ", w_f, "x", h_f, "fps:", fps)

# Определение регионов
region1 = np.array([(118, 345), (196, 312), (342, 367), (262, 416), (114, 468), (113, 385)])
region2 = np.array([(386, 352), (514, 412), (628, 339), (624, 287), (576, 278), (463, 305)])
regions = [region1, region2]


classes_to_count = [0, 1, 2]  # Выбор классов
tick = 60  # Частота записи в файл

# Для логирования
count = [[], []]
count_1min = [[], []]
last_save_time = time.time()

def write_data(count: list, tick: int):
    """Записывает результаты работы в JSON"""
    avg_fps = 0
    for i in range(len(regions)):
        avg_fps += len(count[i])
    avg_fps = avg_fps / (tick * len(regions)) 
    data = {
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count_1min": {
            f"Region_{i+1}": len(count[i]) for i in range(len(regions))
        },
        "sum_1min": {
            f"Region_{i+1}": sum(count[i]) for i in range(len(regions))
        },
        "avg_1min": {
            f"Region_{i+1}": sum(count[i]) / len(count[i]) for i in range(len(regions))
        },
        "avg_fps": avg_fps
    }
    with open("output.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
# async def polytest(boxes, class_ids, track_ids):
#     """Асинхронное обновление логов.
#     Работает медленне обычного цикла!"""
#     count = [[], []]
#     async for i, reg in as_enumerate(regions):  
#         for box, cls_id, tr_id in zip(boxes, class_ids, track_ids):
#             x, y, w, h = box
#             c1, c2 = x + w / 2, y + h / 2
#             class_id, track_id = cls_id, tr_id
#             if cv2.pointPolygonTest(reg, (c1, c2), False) >= 0 and class_id in classes_to_count:
#                 count[i].append([class_id, track_id])
#         count_1min[i].append(len(count[i]))

# Основной цикл  
while cap.isOpened():
    start_time = time.time()
    success, im0 = cap.read()
    im0 = cv2.resize(im0, (640, 640))
    # В параметрах можно менять conf и imgsz
    results = model.track(source=im0,
                          conf=0.6,
                          imgsz=640,
                          persist=True,
                          device="cuda")
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Логирование результатов
    for i, reg in enumerate(regions):
        count[i] = []
        for box, cls_id, tr_id in zip(boxes, class_ids, track_ids):
            x, y, w, h = box
            c1, c2 = x + w / 2, y + h / 2
            if cv2.pointPolygonTest(reg, (c1, c2), False) >= 0 and cls_id in classes_to_count:
                count[i].append([cls_id, tr_id])
        # print(f"Count[{i}]", count[i])
        # print("Count", count)
        count_1min[i].append(len(count[i]))
    #     print(f"Count_1min[{i}]", count_1min[i])
    # print("Count_1min", count_1min)

    # Запись результатов в файл с заданной частотой
    if start_time * 1000 - last_save_time * 1000 > tick * 1000:
        write_data(count_1min, tick)
        last_save_time = time.time()
        count_1min = [[], []]

cap.release()
cv2.destroyAllWindows()

 