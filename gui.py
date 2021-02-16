import tkinter as tk
from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from keras.models import load_model

model = load_model('saved_mnist_model')
size = 300


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=size, height=size, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        # self.canvas.pack(expand=YES, fill=BOTH)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        filename = "image_1.pgm"
        rect = (self.winfo_rootx() + 8, self.winfo_rooty() + 55, self.winfo_rootx() + (size*2) + 20, self.winfo_rooty() + (size*2) + 60)
        im = ImageGrab.grab(rect).save(filename)

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

        cv2.imshow('Output', image)
        cv2.imwrite(filename, image)
        '''pic = PhotoImage(master=self.canvas, file=filename)
        self.canvas.one = PhotoImage(master=self.canvas, file=filename)
        self.canvas.create_image(rect[0], rect[1], pic, anchor=NW)'''

        cv2.imshow('image', image)
        cv2.waitKey(0)


        # digit, acc = predict_digit(im)
        # self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
