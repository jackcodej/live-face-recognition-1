import threading
import cv2
from deepface import DeepFace

# define a camera object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# set the camera's height and width
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
# init face match as false
face_match = False

# load reference image
reference_image = cv2.imread("./reference.jpg")

# Checks if reference image and current image has the same face on it
def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_image.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False



while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                # args must be passed as a tuple so we place the comma after so python interprets it as such
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video",frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()