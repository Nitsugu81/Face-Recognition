import threading
import cv2
from deepface import DeepFace
import os

cap = cv2.VideoCapture(0, cv2.CAP_V4L)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
image_folder = "Images"

# Charger toutes les images du dossier dans une liste
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
reference_images = [cv2.imread(os.path.join(image_folder, img)) for img in image_files]

face_match = False
matching_image = None

def check_face(frame):
    global face_match, matching_image
    try:
        for i, reference_img in enumerate(reference_images):
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                face_match = True
                matching_image = image_files[i]
                break
            else:
                face_match = False
                matching_image = None
    except ValueError:
        face_match = False
        matching_image = None

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 10 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if face_match:
            cv2.putText(frame, f"MATCH! ({matching_image})", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('video', frame)
    else:
        print("Error: Could not read frame from the camera.")
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
