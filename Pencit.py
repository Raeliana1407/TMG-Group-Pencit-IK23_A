import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageProcessorApp:
    # --- Palet Warna ---
    BG_COLOR = "#2c3e50"
    FRAME_COLOR = "#34495e"
    TEXT_COLOR = "#ecf0f1"
    BUTTON_COLOR = "#3498db"
    BUTTON_ACTIVE_COLOR = "#2980b9"

    def __init__(self, root):
        self.root = root
        self.root.title("TMG Group-Citra Pro ITH")
        self.root.geometry("1280x720")
        self.root.configure(bg=self.BG_COLOR)

        # --- Inisialisasi Variabel ---
        self.original_image = None
        self.processed_image = None
        self.second_image_for_logic = None
        self.selected_process = tk.StringVar()
        self.status_text = tk.StringVar(value="Belum ada proses.")

        # --- Layout Utama ---
        control_frame = tk.Frame(self.root, width=320, bg=self.FRAME_COLOR)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        control_frame.pack_propagate(False)

        image_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        # --- PERBAIKAN TATA LETAK DIMULAI DI SINI ---
        # Menggunakan 'grid' untuk menempatkan gambar bersebelahan (kiri-kanan)
        
        # Konfigurasi grid agar kedua kolom berbagi ruang secara merata
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)

        # Panel untuk gambar asli (kiri)
        self.panel_original = tk.Label(image_frame, text="Original Image", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12))
        self.panel_original.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        # Panel untuk gambar hasil (kanan)
        self.panel_processed = tk.Label(image_frame, text="Processed Result", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12))
        self.panel_processed.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        # --- PERBAIKAN TATA LETAK SELESAI ---
        
        status_bar = tk.Label(self.root, textvariable=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=self.FRAME_COLOR, fg=self.TEXT_COLOR)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_sidebar_controls(control_frame)

    def create_sidebar_controls(self, parent_frame):
        tk.Label(parent_frame, text="Image Processor", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 16, 'bold')).pack(pady=10)

        self.create_styled_button(parent_frame, "Upload Image", self.load_image).pack(fill=tk.X, pady=10, padx=10)
        
        tk.Label(parent_frame, text="Select Process:", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 10)).pack(anchor="w", padx=10)
        
        processes = [
            "Grayscale", "Biner (Threshold)", "Atur Kecerahan", 
            "Deteksi Tepi (Canny)",
            "Logika AND", "Logika OR", "Logika XOR", "Logika NOT (Invert)",
            "Morfologi Dilasi (Persegi)", "Morfologi Dilasi (Salib)",
            "Histogram Grayscale", "Histogram RGB"
        ]
        self.process_combobox = ttk.Combobox(parent_frame, textvariable=self.selected_process, values=processes, state='disabled')
        self.process_combobox.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.process_combobox.bind("<<ComboboxSelected>>", self.on_process_selected)

        self.options_frame = tk.Frame(parent_frame, bg=self.FRAME_COLOR)
        self.options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.threshold_label = tk.Label(self.options_frame, text="Threshold Value:", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR)
        self.threshold_scale = Scale(self.options_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, troughcolor='#bdc3c7', highlightthickness=0)
        self.threshold_scale.set(128)

        self.brightness_label = tk.Label(self.options_frame, text="Brightness Value:", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR)
        self.brightness_scale = Scale(self.options_frame, from_=-100, to=100, orient=tk.HORIZONTAL, bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, troughcolor='#bdc3c7', highlightthickness=0)
        
        self.canny_t1_label = tk.Label(self.options_frame, text="Canny Threshold 1 (Low):", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR)
        self.canny_t1_scale = Scale(self.options_frame, from_=0, to=250, orient=tk.HORIZONTAL, bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, troughcolor='#bdc3c7', highlightthickness=0)
        self.canny_t1_scale.set(100)
        
        self.canny_t2_label = tk.Label(self.options_frame, text="Canny Threshold 2 (High):", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR)
        self.canny_t2_scale = Scale(self.options_frame, from_=0, to=500, orient=tk.HORIZONTAL, bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, troughcolor='#bdc3c7', highlightthickness=0)
        self.canny_t2_scale.set(200)

        self.logic_info_label = tk.Label(self.options_frame, text="Operasi ini membutuhkan gambar kedua. Jika belum ada, Anda akan diminta untuk meng-upload-nya.", wraplength=280, bg=self.FRAME_COLOR, fg="#f1c40f")
        
        self.create_styled_button(parent_frame, "Process Image", self.execute_process).pack(fill=tk.X, pady=10, padx=10)

    def on_process_selected(self, event=None):
        for widget in self.options_frame.winfo_children():
            widget.pack_forget()

        selection = self.selected_process.get()
        
        if selection == "Biner (Threshold)":
            self.threshold_label.pack()
            self.threshold_scale.pack(fill=tk.X)
        elif selection == "Atur Kecerahan":
            self.brightness_label.pack()
            self.brightness_scale.pack(fill=tk.X)
        elif selection == "Deteksi Tepi (Canny)":
            self.canny_t1_label.pack()
            self.canny_t1_scale.pack(fill=tk.X)
            self.canny_t2_label.pack()
            self.canny_t2_scale.pack(fill=tk.X)
        elif selection in ["Logika AND", "Logika OR", "Logika XOR"]:
             self.logic_info_label.pack()

    def execute_process(self):
        if not self.check_image_loaded(): return
        selection = self.selected_process.get()
        if not selection:
            messagebox.showwarning("Peringatan", "Silakan pilih proses terlebih dahulu.")
            return
            
        if selection == "Grayscale": self.apply_grayscale()
        elif selection == "Biner (Threshold)": self.apply_binary(self.threshold_scale.get())
        elif selection == "Atur Kecerahan": self.apply_brightness(self.brightness_scale.get())
        elif selection == "Deteksi Tepi (Canny)": self.apply_canny_edge_detection()
        elif selection == "Logika AND": self.apply_logical_and()
        elif selection == "Logika OR": self.apply_logical_or()
        elif selection == "Logika XOR": self.apply_logical_xor()
        elif selection == "Logika NOT (Invert)": self.apply_logical_not()
        elif selection == "Morfologi Dilasi (Persegi)": self.apply_morphology('rect')
        elif selection == "Morfologi Dilasi (Salib)": self.apply_morphology('cross')
        elif selection == "Histogram Grayscale": self.show_grayscale_histogram()
        elif selection == "Histogram RGB": self.show_rgb_histogram()
        
        if "Histogram" not in selection:
             self.status_text.set(f"Processed: {selection}")

    def create_styled_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR,
                         activebackground=self.BUTTON_ACTIVE_COLOR, activeforeground=self.TEXT_COLOR,
                         font=('Arial', 12, 'bold'), relief=tk.FLAT, pady=8)

    def check_image_loaded(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Silakan upload gambar terlebih dahulu!")
            return False
        return True

    def display_image(self, cv_image, panel):
        is_gray = len(cv_image.shape) == 2
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) if not is_gray else cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Penyesuaian ukuran gambar agar sesuai dengan panelnya
        panel_w, panel_h = panel.winfo_width(), panel.winfo_height()
        if panel_w > 1 and panel_h > 1: # Pastikan panel sudah dirender
            pil_image.thumbnail((panel_w, panel_h-4), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(pil_image)
        panel.config(image=tk_image)
        panel.image = tk_image

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.original_image = cv2.imread(path)
            # Tunda display agar panel sempat di-render ukurannya
            self.root.after(100, lambda: self.display_image(self.original_image, self.panel_original))
            self.panel_processed.config(image='', text="Hasil Proses")
            self.panel_processed.image = None
            self.second_image_for_logic = None
            self.status_text.set("Gambar berhasil di-upload.")
            self.process_combobox.config(state='readonly')
            
    def load_second_image(self):
        if not self.check_image_loaded(): return
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.second_image_for_logic = cv2.imread(path)
            messagebox.showinfo("Info", "Gambar kedua berhasil dimuat untuk operasi logika.")
    
    # --- FUNGSI-FUNGSI PEMROSESAN ---
    def apply_grayscale(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_binary(self, threshold_value):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, int(threshold_value), 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_brightness(self, value):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        final_hsv = cv2.merge((h, s, v))
        self.processed_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_canny_edge_detection(self):
        t1 = self.canny_t1_scale.get()
        t2 = self.canny_t2_scale.get()
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, t1, t2)
        self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def _prepare_logic_images(self):
        if self.second_image_for_logic is None:
            self.load_second_image()
            if self.second_image_for_logic is None:
                 return None, None
        h, w, _ = self.original_image.shape
        img2_resized = cv2.resize(self.second_image_for_logic, (w, h))
        gray1 = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        return gray1, gray2

    def apply_logical_and(self):
        img1, img2 = self._prepare_logic_images()
        if img1 is not None:
            result = cv2.bitwise_and(img1, img2)
            self.processed_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.panel_processed)

    def apply_logical_or(self):
        img1, img2 = self._prepare_logic_images()
        if img1 is not None:
            result = cv2.bitwise_or(img1, img2)
            self.processed_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.panel_processed)

    def apply_logical_xor(self):
        img1, img2 = self._prepare_logic_images()
        if img1 is not None:
            result = cv2.bitwise_xor(img1, img2)
            self.processed_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.panel_processed)

    def apply_logical_not(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_not(gray)
        self.processed_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def show_grayscale_histogram(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure("Histogram Grayscale")
        plt.title("Histogram Grayscale")
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Jumlah Piksel")
        plt.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]), color='gray')
        plt.xlim([0, 256])
        plt.show()

    def show_rgb_histogram(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure("Histogram RGB")
        plt.title("Histogram RGB")
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Jumlah Piksel")
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.xlim([0, 256])
        plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
        plt.show()

    def apply_morphology(self, se_shape):
        _, binary = cv2.threshold(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT if se_shape == 'rect' else cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        self.processed_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam') 
    except tk.TclError:
        print("Tema 'clam' tidak tersedia, menggunakan tema default.")
    
    app = ImageProcessorApp(root)
    root.mainloop()