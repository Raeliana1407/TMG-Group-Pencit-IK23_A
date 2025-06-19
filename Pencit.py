import tkinter as tk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageProcessorApp:
    # --- Palet Warna untuk Tampilan Baru ---
    BG_COLOR = "#2c3e50"         # Biru gelap
    FRAME_COLOR = "#34495e"      # Biru keabuan
    TEXT_COLOR = "#ecf0f1"       # Putih pudar
    BUTTON_COLOR = "#3498db"     # Biru cerah (Aksen)
    BUTTON_ACTIVE_COLOR = "#2980b9" # Biru lebih gelap saat ditekan

    def __init__(self, root):
        self.root = root
        self.root.title("CitraPro ITH - TMG Group")
        self.root.geometry("1280x720")
        self.root.configure(bg=self.BG_COLOR)

        # Inisialisasi variabel untuk menyimpan gambar dalam format OpenCV
        self.original_image = None
        self.processed_image = None
        self.second_image_for_logic = None

        # --- Membuat Layout Utama ---
        # Frame untuk kontrol (tombol-tombol)
        control_frame = tk.Frame(self.root, width=320, bg=self.FRAME_COLOR)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        control_frame.pack_propagate(False)

        # Frame untuk menampilkan gambar
        image_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        # Panel untuk gambar asli dan hasil
        self.panel_original = tk.Label(image_frame, text="Gambar Asli", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12))
        # Mengatur posisi panel gambar asli di ATAS
        self.panel_original.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=(0, 5))

        self.panel_processed = tk.Label(image_frame, text="Hasil Proses", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12))
        # Mengatur posisi panel hasil proses di BAWAH
        self.panel_processed.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=(5, 0))

        # --- Menambahkan Widget ke Control Frame ---
        self.create_widgets(control_frame)

    def create_styled_button(self, parent, text, command):
        """Fungsi bantuan untuk membuat tombol yang seragam."""
        return tk.Button(parent, text=text, command=command, 
                         bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR, 
                         activebackground=self.BUTTON_ACTIVE_COLOR, activeforeground=self.TEXT_COLOR,
                         font=('Arial', 10, 'bold'), relief=tk.FLAT, pady=5)

    def create_widgets(self, parent_frame):
        """Membuat dan menata semua tombol dan kontrol dengan gaya baru."""
        
        # --- File Operations ---
        self.create_styled_button(parent_frame, "Buka Gambar", self.load_image).pack(fill=tk.X, pady=5, padx=10)
        
        # --- Fitur Wajib (Disederhanakan) ---
        tk.Label(parent_frame, text="--- Fitur Utama ---", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12, 'bold')).pack(pady=10)
        self.create_styled_button(parent_frame, "1. Konversi Grayscale", self.apply_grayscale).pack(fill=tk.X, pady=2, padx=10)
        self.create_styled_button(parent_frame, "2. Konversi Citra Biner", self.apply_binary).pack(fill=tk.X, pady=2, padx=10)
        
        # Operasi Aritmatika (Brightness)
        tk.Label(parent_frame, text="3. Atur Kecerahan", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 10)).pack(pady=(10, 0))
        self.brightness_scale = Scale(parent_frame, from_=-100, to=100, orient=tk.HORIZONTAL, label="Nilai",
                                      bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, troughcolor='#bdc3c7', highlightthickness=0)
        self.brightness_scale.pack(fill=tk.X, padx=10)
        self.create_styled_button(parent_frame, "Terapkan Kecerahan", self.apply_brightness).pack(fill=tk.X, pady=2, padx=10)
        
        # Operasi Logika (Hanya AND)
        tk.Label(parent_frame, text="4. Operasi Logika", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 10)).pack(pady=(10, 0))
        self.create_styled_button(parent_frame, "Buka Gambar Kedua (untuk AND)", self.load_second_image).pack(fill=tk.X, pady=2, padx=10)
        self.create_styled_button(parent_frame, "Logika AND", self.apply_logical_and).pack(fill=tk.X, pady=2, padx=10)

        # --- Fitur Opsional (Disederhanakan) ---
        tk.Label(parent_frame, text="--- Fitur Tambahan ---", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        self.create_styled_button(parent_frame, "5. Tampilkan Histogram", self.show_histogram).pack(fill=tk.X, pady=2, padx=10)
        
        # Konvolusi (Hanya Deteksi Tepi)
        tk.Label(parent_frame, text="6. Konvolusi", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 10)).pack(pady=(10, 0))
        self.create_styled_button(parent_frame, "Deteksi Tepi (Sobel)", self.apply_edge_detection).pack(fill=tk.X, pady=2, padx=10)

        # Morfologi
        tk.Label(parent_frame, text="7. Morfologi: Dilasi", bg=self.FRAME_COLOR, fg=self.TEXT_COLOR, font=('Arial', 10)).pack(pady=(10, 0))
        self.create_styled_button(parent_frame, "SE Persegi 3x3", lambda: self.apply_morphology('rect')).pack(fill=tk.X, pady=2, padx=10)
        self.create_styled_button(parent_frame, "SE Salib 3x3", lambda: self.apply_morphology('cross')).pack(fill=tk.X, pady=2, padx=10)
    
    # --- Fungsi Bantuan ---
    def check_image_loaded(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Silakan buka gambar terlebih dahulu!")
            return False
        return True

    def display_image(self, cv_image, panel):
        is_gray = len(cv_image.shape) == 2
        if is_gray:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        # --- MODIFIKASI DIMULAI DI SINI ---
        
        # Jika panel target adalah panel 'processed' dan sudah ada gambar di panel 'original'
        if panel == self.panel_processed and self.panel_original.image is not None:
            # Ambil dimensi dari gambar yang sudah ditampilkan di panel original
            ref_width = self.panel_original.image.width()
            ref_height = self.panel_original.image.height()
            
            # Ubah ukuran gambar yang akan diproses agar sama persis dengan gambar original
            pil_image = pil_image.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
        else:
            # Jika ini adalah gambar original (pertama kali dimuat), gunakan logika lama
            # untuk menyesuaikannya dengan ukuran panelnya.
            panel_w = panel.winfo_width()
            if panel_w <= 1: panel_w = 600 # Nilai default jika panel belum dirender
            panel_h = panel.winfo_height()
            if panel_h <= 1: panel_h = 600

            pil_image.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        
        # --- MODIFIKASI SELESAI ---

        tk_image = ImageTk.PhotoImage(pil_image)
        panel.config(image=tk_image)
        panel.image = tk_image
    # --- Fungsi I/O Gambar ---
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.original_image = cv2.imread(path)
            self.root.after(100, lambda: self.display_image(self.original_image, self.panel_original))
            self.panel_processed.config(image='', text="Hasil Proses")
            self.panel_processed.image = None
            self.second_image_for_logic = None # Reset gambar kedua

    def load_second_image(self):
        if not self.check_image_loaded(): return
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.second_image_for_logic = cv2.imread(path)
            messagebox.showinfo("Info", "Gambar kedua berhasil dimuat untuk operasi AND.")

    # --- Implementasi Fitur ---

    def apply_grayscale(self):
        if not self.check_image_loaded(): return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_binary(self):
        if not self.check_image_loaded(): return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_brightness(self):
        if not self.check_image_loaded(): return
        value = self.brightness_scale.get()
        # Menggunakan np.clip untuk mencegah overflow/underflow (nilai > 255 atau < 0)
        # Ini memberikan hasil yang lebih natural daripada cv2.add/subtract
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        final_hsv = cv2.merge((h, s, v))
        self.processed_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_logical_and(self):
        """Hanya menerapkan operasi Logika AND."""
        if not self.check_image_loaded(): return
        if self.second_image_for_logic is None:
            messagebox.showerror("Error", "Silakan muat gambar kedua untuk operasi AND.")
            return

        h, w, _ = self.original_image.shape
        img2_resized = cv2.resize(self.second_image_for_logic, (w, h))

        self.processed_image = cv2.bitwise_and(self.original_image, img2_resized)
        self.display_image(self.processed_image, self.panel_processed)

    def show_histogram(self):
        if not self.check_image_loaded(): return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure("Histogram")
        plt.title("Histogram Grayscale")
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Jumlah Piksel")
        plt.plot(hist, color='cyan')
        plt.xlim([0, 256])
        plt.show()

    def apply_edge_detection(self):
        """Hanya menerapkan deteksi tepi Sobel."""
        if not self.check_image_loaded(): return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Gunakan Sobel untuk sumbu X dan Y, lalu gabungkan untuk hasil yang lebih baik
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel_8u = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        self.processed_image = cv2.cvtColor(sobel_8u, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

    def apply_morphology(self, se_shape):
        if not self.check_image_loaded(): return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        if se_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        elif se_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        dilated = cv2.dilate(binary, kernel, iterations=1)
        self.processed_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.panel_processed)

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()