import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def generate_spd_matrix(size, preset='default'):
    A = np.random.rand(size, size)
    if preset == 'smooth':
        A = np.ones((size, size))
    elif preset == 'edge':
        A = np.eye(size) * 5 + np.random.rand(size, size) * 0.1
    spd = np.dot(A, A.T) + size * np.eye(size)
    return spd

def cholesky_to_gaussian_kernel(spd_matrix):
    L = np.linalg.cholesky(spd_matrix)
    kernel = np.dot(L, L.T)
    kernel /= kernel.sum()
    return kernel

def apply_blur(image, kernel):
    return cv2.filter2D(image, -1, kernel)

class CholeskyBlurApp:
    def __init__(self, root):  # fixed __init__
        self.root = root
        self.root.title("Image Blur using Cholesky Decomposition")
        self.root.geometry("950x750")
        self.root.configure(bg="#e0f7fa")

        self.image_path = None
        self.original_image = None
        self.blurred_image = None
        self.start_x = self.start_y = self.end_x = self.end_y = None

        button_style = {
            "width": 25,
            "height": 2,
            "bg": "#00796B",
            "fg": "white",
            "font": ('Arial', 10, 'bold')
        }

        self.load_button = tk.Button(root, text="📁 Load Image", command=self.load_image, **button_style)
        self.load_button.pack(pady=5)

        self.kernel_frame = tk.Frame(root, bg="#e0f7fa")
        self.kernel_frame.pack(pady=5)

        tk.Label(self.kernel_frame, text="Kernel Size:", bg="#e0f7fa", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.kernel_size = tk.IntVar(value=5)
        tk.Entry(self.kernel_frame, textvariable=self.kernel_size, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(self.kernel_frame, text="Preset:", bg="#e0f7fa", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.preset = tk.StringVar(value='default')
        tk.OptionMenu(self.kernel_frame, self.preset, 'default', 'smooth', 'edge').pack(side=tk.LEFT, padx=5)

        self.blur_button = tk.Button(root, text="✨ Apply Blur", command=self.blur_image, **button_style)
        self.blur_button.pack(pady=5)

        self.save_button = tk.Button(root, text="💾 Save Blurred Image", command=self.save_image, **button_style)
        self.save_button.pack(pady=5)

        self.canvas = tk.Canvas(root, width=600, height=600, bg="#ffffff", bd=2, relief="groove")
        self.canvas.pack(pady=10)
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            img = cv2.imread(self.image_path)
            self.original_image = img
            self.blurred_image = None
            self.display_image(img)

    def blur_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        try:
            ksize = int(self.kernel_size.get())
            if ksize < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Kernel size must be an integer ≥ 2.")
            return

        preset = self.preset.get()
        spd_matrix = generate_spd_matrix(ksize, preset)
        kernel = cholesky_to_gaussian_kernel(spd_matrix)

        img_copy = self.original_image.copy()

        if all(v is not None for v in [self.start_x, self.start_y, self.end_x, self.end_y]):
            x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
            x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)

            roi = img_copy[y1:y2, x1:x2]
            blurred_roi = apply_blur(roi, kernel)
            img_copy[y1:y2, x1:x2] = blurred_roi
            self.start_x = self.start_y = self.end_x = self.end_y = None
        else:
            img_copy = apply_blur(img_copy, kernel)

        self.blurred_image = img_copy
        self.display_image(self.blurred_image)

    def display_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_dim = 600
        scale = min(max_dim / h, max_dim / w) if w > 0 and h > 0 else 1
        new_size = (int(w * scale), int(h * scale))
        img_resized = cv2.resize(img_rgb, new_size)
        img_pil = Image.fromarray(img_resized)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.delete("all")  # clear previous image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Save dimensions for coordinate mapping
        self.display_w, self.display_h = new_size
        self.img_w, self.img_h = w, h
        self.displayed_image = image

    def save_image(self):
        if self.blurred_image is None:
            messagebox.showwarning("Warning", "No blurred image to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                filetypes=[("JPEG files", ".jpg"), ("PNG files", ".png")])
        if filepath:
            cv2.imwrite(filepath, self.blurred_image)
            messagebox.showinfo("Saved", f"Image saved to {filepath}")

    def start_selection(self, event):
        if self.original_image is None:
            return
        scale_x = self.img_w / self.display_w
        scale_y = self.img_h / self.display_h
        self.start_x = int(event.x * scale_x)
        self.start_y = int(event.y * scale_y)

    def end_selection(self, event):
        if self.original_image is None:
            return
        scale_x = self.img_w / self.display_w
        scale_y = self.img_h / self.display_h
        self.end_x = int(event.x * scale_x)
        self.end_y = int(event.y * scale_y)
        messagebox.showinfo("Region Selected", f"Selected region: ({self.start_x},{self.start_y}) → ({self.end_x},{self.end_y})")

# Run the app
if __name__ == "__main__":  # fixed __name__ check
    root = tk.Tk()
    app = CholeskyBlurApp(root)
    root.mainloop()
