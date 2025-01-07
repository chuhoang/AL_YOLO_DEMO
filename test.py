import cv2

# read video and save in test_folder
video = cv2.VideoCapture('/home/mq/data_disk2T/Thang/ps/error_1911.mp4')
count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    if count % 16 == 0:
        cv2.imwrite(f'test_folder/{count}.jpg', frame)
    count += 1