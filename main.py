import threading
import os

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
matched_image = None
face_match_lock = threading.Lock()  # Add a lock for synchronization

# Specify the path to the folder containing images
images_folder = "Images"

# Get the list of image files in the folder
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

def check_face(frame):
    global face_match, matched_image
    try:
        for image_file in image_files:
            reference_img = cv2.imread(os.path.join(images_folder, image_file))
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                with face_match_lock:  # Acquire the lock before modifying face_match and matched_image
                    face_match = True
                    matched_image = reference_img.copy()
                break  # Break the loop if a match is found
            else:
                with face_match_lock:  # Acquire the lock before modifying face_match and matched_image
                    face_match = False
                    matched_image = None
    except ValueError:
        with face_match_lock:  # Acquire the lock before modifying face_match and matched_image
            face_match = False
            matched_image = None


cst = False
while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        with face_match_lock:  # Acquire the lock before reading face_match and matched_image
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if matched_image is not None:
                    cv2.imshow('Matched Image', matched_image)
                    cv2.moveWindow('Matched Image', 700, 50)
                    cst = True         
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                if cst == True:
                    print("ok")
                    cv2.destroyWindow('Matched Image')  # Close the window when there is no match

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
