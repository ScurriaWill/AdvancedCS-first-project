from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from keras.models import load_model


def clear_widget():
    global can
    can.delete("all")


def activate_event(event):
    global lastX, lastY
    can.bind('<B1-Motion>', draw_lines)
    lastX, lastY = event.x, event.y


def draw_lines(event):
    global lastX, lastY
    x, y = event.x, event.y
    can.create_line((lastX, lastY, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastX, lastY = x, y


def recognize_digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = can

    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    x1 = x + width
    y1 = y + height
    # points = cv2.boxPoints((x, y, x1, y1))
    print("DEBUG: " + str(x)+" " + str(y)+" " + str(x1)+" " + str(y1))
    # print(str(points))
    bbox = widget.bbox()
    ImageGrab.grab(bbox=bbox).save(filename)

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y - top:y + h + bottom, x - left: x + w + right]
        try:
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except Exception as e:
            continue
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, font_scale, color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey(0)


model = load_model('saved_mnist_model')

root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognizer")
width = 640
height = 480

lastX, lastY = None, None
image_number = 0

can = Canvas(root, width=width, height=height, bg='white')
can.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

can.bind('<Button-1>', activate_event)

btn_save = Button(text="Recognize Digit", command=recognize_digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
