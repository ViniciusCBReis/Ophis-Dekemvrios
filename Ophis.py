import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def open_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    display_image(image)
    predict_and_display(image, file_path)

def display_image(image):
    img = ImageTk.PhotoImage(image.resize((300, 300)))
    panel.configure(image=img)
    panel.image = img

def predict_and_display(image, image_path):
    results = model.predict(image_path)
    result_text.set(f"Previsões:\n")
    for result in results:
        result_text.set(result_text.get() + f"Objeto: {result['class']}, Confiança: {result['confidence']:.2f}\n")

# Interface com Tkinter
root = tk.Tk()
root.title("Previsão de Objetos")
root.geometry("500x500")

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Selecionar Imagem", command=open_image)
btn.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, wraplength=300)
result_label.pack()

# Carregar o modelo
model = load_model("Modelos/treinamento_aps4/weights/best.pt")

root.mainloop()

