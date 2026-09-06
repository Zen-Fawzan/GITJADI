# this code is ordinary and gonna do simple things related to face detection (angry , sad , smile) 
# with machine learning languages 
import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        # DeepFace may return a list
        if isinstance(result, list):
            result = result[0]

        emotion = result["dominant_emotion"]

        if emotion == "happy":
            status = "HAPPY :)"
        else:
            status = "NOT HAPPY :("

        # Draw result
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    except Exception as e:
        cv2.putText(
            frame,
            "Detecting...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    cv2.imshow("Happy Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
