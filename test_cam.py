import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = app.get(frame)

    for face in faces:
        x1,y1,x2,y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()