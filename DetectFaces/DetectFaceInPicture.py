import cv2
import os

input_dir = 'images'
output_dir = 'output'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, 'Face', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        output_file_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_detect{os.path.splitext(filename)[1]}')

        cv2.imwrite(output_file_path, img)

print('Gesichtserkennung abgeschlossen. Alle bearbeiteten Bilder wurden im Ordner "output" gespeichert.')
