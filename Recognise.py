import cv2
import xlwrite

import time

start = time.time()
period = 8
face_cas = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer/trainer.yml');
flag = 0;
id = 0;
filename = 'filename';
dict = {
    'item1': 1
}
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2);
        id, conf = recognizer.predict(roi_gray)
        if (conf < 50):
            if (id == 2853):
                id = 'Lavish'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2853, id, 'yes');
                    dict[str(id)] = str(id);

            elif (id == 2540):
                id = 'Anirudh'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2540, id, 'yes');
                    dict[str(id)] = str(id);

            elif (id == 2482):
                id = 'Aryan'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2482, id, 'yes');
                    dict[str(id)] = str(id)
            elif (id == 1139):
                id = 'Lucky'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 1139, id, 'yes');
                    dict[str(id)] = str(id)
            elif (id == 1974):
                id = 'Rohit'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 1974, id, 'yes');
                    dict[str(id)] = str(id)

            elif (id == 2478):
                id = 'Ashish'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2478, id, 'yes');
                    dict[str(id)] = str(id)
            # elif (id == 3):
            #     id = 'Chandana'
            #     if ((str(id)) not in dict):
            #         filename = xlwrite.output('attendance', 'class1', 3, id, 'yes');
            #         dict[str(id)] = str(id)

            # elif (id == 3):
            #     id = 'Chandana'
            #     if ((str(id)) not in dict):
            #         filename = xlwrite.output('attendance', 'class1', 3, id, 'yes');
            #         dict[str(id)] = str(id)
            

        else:
            id = 'Unknown, can not recognize'
            flag = flag + 1
            break

        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
    cv2.imshow('frame', img);

    if cv2.waitKey(1)==13:
        break;

cap.release();
cv2.destroyAllWindows();

