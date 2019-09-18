import cv2

cap = cv2.VideoCapture(0)

prevtime = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    str = cap.get(cv2.CAP_PROP_FPS)

    cv2.putText(frame, 'FPS : ' + repr(str), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    # end 프레임
    dst = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow('video', dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()