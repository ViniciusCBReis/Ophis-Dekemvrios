import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

model_paths = [
    "Modelos/treinamento_aps4/weights/best.pt",

]
models = [YOLO(path) for path in model_paths]

# Cria pasta de previsões, se não existir
os.makedirs("Previsoes", exist_ok=True)

def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Limpa as abas antigas para evitar duplicação
    for tab in notebook.tabs():
        notebook.forget(tab)

    # Realiza a previsão apenas para os modelos selecionados
    for idx, model in enumerate(models):
        if model_active[idx].get():  # Verifica se o modelo está ativado
            results = model.predict(file_path, save=True, project="Previsoes", name=f"model_{idx}_exp", exist_ok=True)

            # Caminho da imagem predita
            predicted_image_path = os.path.join("Previsoes", f"model_{idx}_exp", os.path.basename(file_path))

            # Verifica se a imagem foi salva e exibe em uma nova aba
            if os.path.exists(predicted_image_path):
                display_image(predicted_image_path, title=f"Resultado do Modelo {idx + 1}")
            else:
                print(f"Erro: A imagem predita para o Modelo {idx + 1} não foi encontrada.")

def display_image(image_path, title="Resultado"):
    """Exibe a imagem em uma nova aba do notebook."""
    # Cria uma nova aba
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)

    # Carrega e redimensiona a imagem
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Ajusta para caber na interface
    img = ImageTk.PhotoImage(img)

    # Adiciona a imagem à aba
    img_label = ttk.Label(frame, image=img)
    img_label.image = img  # Evita que a imagem seja excluída pelo garbage collector
    img_label.pack(pady=10)

# Configuração da interface gráfica com tema escuro
root = ttk.Window(themename="darkly")  # Utilizando o tema escuro do ttkbootstrap
root.title("Teste de Previsão com Múltiplos Modelos")
root.geometry("500x600")

# Variáveis para armazenar o estado dos switches (Checkboxes)
model_active = [tk.BooleanVar(value=True) for _ in model_paths]

# Checkbox para selecionar quais modelos usar
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

# Botão para abrir imagem e realizar previsões com os modelos selecionados
btn = ttk.Button(root, text="Selecionar Imagem", command=open_image, bootstyle="primary")
btn.pack(pady=20)

# Notebook (abas) para exibir as previsões de cada modelo separadamente
notebook = ttk.Notebook(root, bootstyle="dark")
notebook.pack(expand=True, fill="both")

root.mainloop()
