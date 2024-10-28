import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

model_paths = [
    "Modelos/treinamento_aps4/weights/best.pt",
    "Modelos/treinamento_equip/weights/best.pt",
    "Modelos/treinamento_ferr1/weights/best.pt",
    "Modelos/treinamento_ferr12/weights/best.pt",
    "Modelos/treinamento_concrto/weights/best.pt",
]
models = [YOLO(path) for path in model_paths]

os.makedirs("Previsoes", exist_ok=True)

def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    for tab in notebook.tabs():
        notebook.forget(tab)

    for idx, model in enumerate(models):
        if model_active[idx].get():
            results = model.predict(file_path, save=True, project="D:/Projetos/OphisDekemvrios/Previsoes", name=f"model_{idx}_exp", exist_ok=True)

            print(f"Diretório salvo para o Modelo {idx + 1}: {results[0].save_dir}")
            print(f"Arquivo original: {os.path.basename(file_path)}")

            predicted_image_path = os.path.join(str(results[0].save_dir), os.path.basename(file_path))

            if os.path.exists(predicted_image_path):
                display_image(predicted_image_path, title=f"Resultado do Identificador {idx + 1}")
            else:
                print(f"Erro: A imagem predita para o Modelo {idx + 1} não foi encontrada no caminho {predicted_image_path}.")

def display_image(image_path, title="Resultado"):

    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)

    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)


    img_label = ttk.Label(frame, image=img)
    img_label.image = img
    img_label.pack(pady=10)

root = ttk.Window(themename="darkly")
root.title("Teste de Previsão com Múltiplos Modelos")
root.geometry("500x600")

model_active = [tk.BooleanVar(value=True) for _ in model_paths]

checkbox_frame = ttk.Frame(root)
checkbox_frame.pack(pady=10)
ttk.Label(checkbox_frame, text="Selecione os modelos para identificação:", bootstyle="info").pack(anchor="w")
for idx, path in enumerate(model_paths):
    ttk.Checkbutton(
        checkbox_frame,
        text=f"Identificador {idx + 1}",
        variable=model_active[idx],
        bootstyle="round-toggle"
    ).pack(anchor="w")

btn = ttk.Button(root, text="Selecionar Imagem", command=open_image, bootstyle="primary")
btn.pack(pady=20)

notebook = ttk.Notebook(root, bootstyle="dark")
notebook.pack(expand=True, fill="both")

root.mainloop()
