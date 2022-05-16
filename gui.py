import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from constants import *


def classify():
    original = Image.open(image_data)
    original = original.resize((256, 256))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    label = model.predict(image_batch)
    text = "defective" if label == 0 else "non-defective"
    tk.Label(frame, text=text, font=("Ubuntu Regular", 20)).pack()


def choose():
    global image, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()
    image_data = filedialog.askopenfilename(
        initialdir=PATH_TO_IMAGES, 
        title="choose an image",
        filetypes=(("all files", "*.*"), ("png files", "*.png"))
    )
    basewidth = 600
    image = Image.open(image_data)
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    file_name = image_data.split("/")
    tk.Label(frame, text= str(file_name[-1]).upper()).pack()
    tk.Label(frame, image=image).pack()
    

def create_gui():
    global frame, model, my_dict
    root = tk.Tk()
    root.title("mini project")
    root.resizable(False, False)
    tk.Label(root, text="defective lemon classifier", padx=25, pady=6, font=("", 12)).pack()
    canvas = tk.Canvas(root, height=750, width=750, bg="grey")
    canvas.pack()
    frame = tk.Frame(root, bg="white")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    choose_image = tk.Button(
        root, text="choose image", padx=35, pady=10,
        fg="white", bg="grey", command=choose
    )
    choose_image.pack(side=tk.LEFT)
    classif_image = tk.Button(
        root, text="classify image", padx=35, pady=10,
        fg="white", bg="grey", command=classify
    )
    classif_image.pack(side=tk.RIGHT)
    model = load_model(PATH_TO_CKP)
    root.mainloop()


if __name__ == "__main__":
    create_gui()
