import cv2
from cv2 import face
import os
import numpy as np
import tkinter as tk
import tkinter.font as font

name_entry = None
id_entry = None

def collect_data():
    global name_entry, id_entry
    name = name_entry.get()
    ids = id_entry.get()

    count = 1

    cap = cv2.VideoCapture(0)

    filename = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(filename)

    while True:
        _, frm = cap.read()

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            # Resize the cropped face region to a fixed size (e.g., 100x100)
            resized_roi = cv2.resize(roi, (100, 100))

            cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", resized_roi)
            count += 1
            cv2.putText(frm, f"{count}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.imshow("new", resized_roi)

        cv2.imshow("identify", frm)

        if cv2.waitKey(1) == 27 or count > 300:
            cv2.destroyAllWindows()
            cap.release()
            train()
            break

def train():
    print("Training part initiated!")

    recog = face.LBPHFaceRecognizer_create()

    dataset = 'persons'
    paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

    faces = []
    ids = []
    labels = []
    for path in paths:
        file_name = os.path.basename(path)
        label = file_name.split('-')[0]
        id_str = file_name.split('-')[2].split('.')[0]
        if id_str:
            face_id = int(id_str)
            labels.append(label)
            ids.append(face_id)
            faces.append(cv2.imread(path, 0))

    recog.train(faces, np.array(ids))

    recog.save('model.yml')


def identify():
    cap = cv2.VideoCapture(0)

    filename = "haarcascade_frontalface_default.xml"

    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    labelslist = {}
    for path in paths:
        labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

    print(labelslist)

    model_file = 'model.yml'
    if not os.path.isfile(model_file):
        print(f"Model file '{model_file}' not found. Please train the model first.")
        return

    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read(model_file)

    cascade = cv2.CascadeClassifier(filename)

    while True:
        _, frm = cap.read()

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            try:
                label = recog.predict(roi)

                if label[1] < 100:
                    cv2.putText(frm, f"{labelslist.get(str(label[0]), 'Unknown')} + {int(label[1])}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frm, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            except cv2.error as e:
                print(f"Error occurred during face recognition: {e}")

        cv2.imshow("identify", frm)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break


def maincall():
    global name_entry, id_entry

    root = tk.Tk()
    root.geometry("480x100")
    root.title("Identify")

    label = tk.Label(root, text="Select below buttons")
    label.grid(row=0, columnspan=2)
    label_font = font.Font(size=35, weight='bold', family='Helvetica')
    label['font'] = label_font

    btn_font = font.Font(size=25)

    name_label = tk.Label(root, text="Name:")
    name_label.grid(row=1, column=0, padx=(5, 5), pady=(10, 10))
    name_entry = tk.Entry(root)
    name_entry.grid(row=1, column=1, padx=(5, 5), pady=(10, 10))

    id_label = tk.Label(root, text="ID:")
    id_label.grid(row=2, column=0, padx=(5, 5), pady=(10, 10))
    id_entry = tk.Entry(root)
    id_entry.grid(row=2, column=1, padx=(5, 5), pady=(10, 10))

    button1 = tk.Button(root, text="Add Member", command=collect_data, height=2, width=20)
    button1.grid(row=3, column=0, pady=(10, 10), padx=(5, 5))
    button1['font'] = btn_font

    button2 = tk.Button(root, text="Start with known", command=identify, height=2, width=20)
    button2.grid(row=3, column=1, pady=(10, 10), padx=(5, 5))
    button2['font'] = btn_font

    root.mainloop()

if __name__ == "__main__":
    maincall()
