import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_human_frames(frames, min_hits=10):
    hits = 0
    for frame in frames:
        boxes, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05
        )
        if len(boxes) > 0:
            hits += 1
        if hits >= min_hits:
            return True
    return False
