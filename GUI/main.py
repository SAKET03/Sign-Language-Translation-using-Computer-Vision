import tkinter
from tkinter import ttk
from tkinter import scrolledtext
from tkinter.constants import BOTTOM, LEFT, TOP, X
from PIL import ImageTk, Image
from math import *

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pyttsx3 as tts
engine = tts.init()

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def create_model():
    num_classes = 29

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(200, 200, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


resizing_model = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(200, 200, interpolation='bilinear', crop_to_aspect_ratio=True)
])

prediction_model = create_model()
prediction_model.load_weights('training/cp.ckpt')

probability_model = tf.keras.Sequential([prediction_model,
                                         tf.keras.layers.Softmax()])

def start_capture():
    toggle_video_capture['text'] = 'Stop Capturing'
    toggle_video_capture['command'] = stop_capture

    global cap
    cap = cv2.VideoCapture(0)

    global timer_event_id
    timer_event_id = root.after_idle(step)


def step():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    roi = frame[20:220, 20:220]
    char.set(class_names[np.argmax(probability_model.predict(np.expand_dims(roi, 0))[0])])

    cv2.rectangle(frame, (20, 20), (220, 220), (0, 0, 255))

    global image
    image = ImageTk.PhotoImage(Image.fromarray(np.array(frame, dtype=np.uint8), 'RGB'))

    global input_video_frame
    input_video_frame['image'] = image

    global timer_event_id
    timer_event_id = root.after_idle(step)


def stop_capture():
    root.after_cancel(timer_event_id)

    cap.release()

    global image
    global frame_width
    global frame_height
    image = ImageTk.PhotoImage(Image.new('RGB', (frame_width, frame_height)))
    global input_video_frame
    input_video_frame['image'] = image

    toggle_video_capture['text'] = 'Start Capturing'
    toggle_video_capture['command'] = start_capture

    char.set('')

def update_word():
    ch = char.get()

    if ch == '':
        pass
    elif ch == 'nothing':
        pass
    elif ch == 'space':
        word_output.insert('end', ' ')
    elif ch == 'del':
        word_output.delete('end')
    else:
        word_output.insert('end', ch)

def text_to_speech():
    engine.say(word_output.get('1.0', 'end'))
    engine.runAndWait()

root = tkinter.Tk()
root.title('Sign Language Translator')

title = tkinter.Label(root,
                      text='Sign Language Translator',
                      font=('Times New Roman', 25))
title.pack()

input_frame = tkinter.ttk.Frame(root)
input_frame.pack()

initial = cv2.VideoCapture(0)
frame_width = int(initial.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(initial.get(cv2.CAP_PROP_FRAME_HEIGHT))
initial.release()

input_video_frame = tkinter.Label(input_frame)
image = ImageTk.PhotoImage(Image.new('RGB', (frame_width, frame_height)))
input_video_frame['image'] = image
input_video_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

toggle_video_capture = tkinter.Button(input_frame, font=("Times New Roman", 12))
toggle_video_capture['text'] = 'Start Capturing'
toggle_video_capture['command'] = start_capture
toggle_video_capture.grid(row=1, column=0, padx=10, pady=10, sticky='E')

text_to_speech_button = tkinter.Button(input_frame, text='Text to speech', font=("Times New Roman", 12))
text_to_speech_button['command'] = lambda: text_to_speech()
text_to_speech_button.grid(row=1, column=1, pady=10, padx=10, sticky='N')

clear_button = tkinter.Button(input_frame, text='Clear Words', font=("Times New Roman", 12))
clear_button['command'] = lambda: word_output.delete('1.0', 'end')
clear_button.grid(row=1, column=2, sticky='W', pady=10, padx=10)

output_frame = tkinter.ttk.Frame(root, borderwidth=1, relief="solid")
output_frame.pack()

char_label = tkinter.Label(output_frame, text="Character:", font=("Times New Roman", 15))
char_label.grid(row=0, column=0, sticky='w', pady=10, padx=10)

char = tkinter.StringVar()
char.set('')
char_output = tkinter.Label(output_frame, textvariable=char, font=("Times New Roman", 15))
char_output.grid(row=0, column=1, columnspan=3, sticky='W')

word_label = tkinter.Label(output_frame, text="Word:", font=("Times New Roman", 15))
word_label.grid(row=1, column=0, sticky='w', pady=10, padx=10)

word_output = scrolledtext.ScrolledText(output_frame, wrap='word', font=("Times New Roman", 12))
word_output['height'] = 2
word_output.grid(row=2, column=0, sticky='N', pady=10, padx=10, columnspan=3)

width = ceil(root.winfo_screenwidth()/2) 
height = root.winfo_screenheight()
root.geometry(f"{width}x{height}+0+0")
root.resizable(0,0)

root.bind('<KeyPress-space>', lambda e: update_word())
root.mainloop()
