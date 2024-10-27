import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO

def load_model(model_path):
    """Carrega o modelo para fazer previsões."""
    return YOLO(model_path)

def open_image():
    """Abre uma caixa de diálogo para selecionar uma imagem."""
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    display_image(image)
    predict_and_display(image, file_path)

def display_image(image):
    """Exibe a imagem na interface."""
    img = ImageTk.PhotoImage(image.resize((300, 300)))
    panel.configure(image=img)
    panel.image = img

def predict_and_display(image, image_path):
    """Faz previsões e exibe o resultado."""
    results = model.predict(image_path)
    result_text.set(f"Previsões:\n")
    for result in results:
        result_text.set(result_text.get() + f"Objeto: {result['class']}, Confiança: {result['confidence']:.2f}\n")

# Interface com Tkinter
root = tk.Tk()
root.title("Previsão de Objetos")
root.geometry("400x500")

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Selecionar Imagem", command=open_image)
btn.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, wraplength=300)
result_label.pack()

# Carregar o modelo
model = load_model("caminho/para/o/modelo.pt")  # Altere para o caminho do modelo salvo

root.mainloop()
