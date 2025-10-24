import tkinter as tk
from PIL import Image, ImageTk
import os

def show_label_interface(image_path, detected_text):
    def save_and_close():
        user_input = entry.get().strip()
        if not user_input:
            user_input = detected_text
        result['user_input'] = user_input
        root.destroy()

    root = tk.Tk()
    root.title("License Plate Labeling")

    # Load and display the image
    img = Image.open(image_path)
    img = img.resize((400, 120))
    img_tk = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=img_tk)
    panel.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

    # Detected text
    tk.Label(root, text="Detected Text:").grid(row=0, column=1, sticky='w')
    tk.Label(root, text=detected_text, fg='blue', font=('Arial', 14, 'bold')).grid(row=1, column=1, sticky='w')

    # User input
    tk.Label(root, text="Correct License Number:").grid(row=2, column=1, sticky='w')
    entry = tk.Entry(root, width=25, font=('Arial', 14))
    entry.insert(0, detected_text)
    entry.grid(row=3, column=1, padx=5, pady=5)

    # Save button
    save_btn = tk.Button(root, text="Save", command=save_and_close, width=10, bg='green', fg='white')
    save_btn.grid(row=4, column=1, pady=10)

    result = {'user_input': None}
    root.mainloop()
    return result['user_input']

# Example usage:
# user_label = show_label_interface('path/to/image.jpg', 'ABC1234')
# print(user_label)
