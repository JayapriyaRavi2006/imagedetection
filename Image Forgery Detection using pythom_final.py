import tkinter as tk
from tkinter import filedialog, messagebox
import hashlib
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from fpdf import FPDF
import datetime
import os
import webbrowser

def calculate_md5(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def calculate_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def compare_histograms(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)

def feature_match(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def generate_report(img1_path, img2_path, md5_result, ssim_score, hist_score, face1, face2, forgery_detected, forgery_image, authentic_image, angle_changed):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add current date and time at the top-right corner
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.set_font("Arial", size=10)
    pdf.set_xy(150, 10)
    pdf.cell(0, 10, txt=current_time, ln=False, align='R')

    # Set the position back to the left for the title
    pdf.set_xy(10, 20)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Forgery Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"MD5 Match: {md5_result}", ln=True)
    pdf.cell(200, 10, txt=f"SSIM Score: {ssim_score:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"Histogram Correlation: {hist_score:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"Faces in Image 1: {face1}", ln=True)
    pdf.cell(200, 10, txt=f"Faces in Image 2: {face2}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Forgery Detected: {'Yes' if forgery_detected else 'No'}", ln=True)

    if forgery_detected:
        pdf.cell(200, 10, txt=f"Forgery Image: {forgery_image}", ln=True)
        pdf.cell(200, 10, txt=f"Authentic Image: {authentic_image}", ln=True)
    else:
        pdf.cell(200, 10, txt="Note: The same image appears to be rotated or captured from a\ndifferent angle or different persons.", ln=True)

    # Use multi_cell for wrapping long text
    if angle_changed:
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt="Note: The same image appears to be rotated or captured from a different angle or different persons.", align='L')

    pdf.ln(10)
    pdf.cell(200, 10, txt="Compared Images:", ln=True)
    y_pos = pdf.get_y() + 5
    pdf.image(img1_path, x=10, y=y_pos, w=90)
    pdf.image(img2_path, x=110, y=y_pos, w=90)

    filename = f"Forgery_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    webbrowser.open_new(filename)
    return filename

class ForgeryDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Forgery Detection")

        tk.Label(master, text="Select two images to compare").pack()
        tk.Button(master, text="Select Image 1", command=self.load_image1).pack()
        tk.Button(master, text="Select Image 2", command=self.load_image2).pack()
        tk.Button(master, text="Compare Images", command=self.compare_images).pack()

        self.img1_path = None
        self.img2_path = None

    def load_image1(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            img = cv2.resize(img, (512, 512))  # Resize to 512x512
            resized_path = "resized_img1.jpg"
            cv2.imwrite(resized_path, img)
            self.img1_path = resized_path
            messagebox.showinfo("Selected", f"Image 1 selected and resized.")

    def load_image2(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            img = cv2.resize(img, (512, 512))  # Resize to 512x512
            resized_path = "resized_img2.jpg"
            cv2.imwrite(resized_path, img)
            self.img2_path = resized_path
            messagebox.showinfo("Selected", f"Image 2 selected and resized.")

    def compare_images(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)

        md5_result = calculate_md5(self.img1_path) == calculate_md5(self.img2_path)
        ssim_score = calculate_ssim(img1, img2)
        hist_score = compare_histograms(img1, img2)
        face_count1 = detect_faces(img1)
        face_count2 = detect_faces(img2)
        feature_matches = feature_match(img1, img2)

        # Setting the angle_changed flag
        angle_changed = False
        if ssim_score < 0.95 and feature_matches > 30:
            angle_changed = True  # Angle change detected based on SSIM score and feature matches

        forgery_detected = not md5_result or ssim_score < 0.94 or hist_score < 0.9 or face_count1 != face_count2 or feature_matches < 30

        forgery_image = "Image 2 (Possibly Forged)"
        authentic_image = "Image 1 (Likely Authentic)"
        if not forgery_detected:
            forgery_image = "None"
            authentic_image = "Both"

        report = generate_report(self.img1_path, self.img2_path, md5_result, ssim_score,
                                 hist_score, face_count1, face_count2, forgery_detected,
                                 forgery_image, authentic_image, angle_changed)

        result_text = f"Forgery Detected: {'Yes' if forgery_detected else 'No'}\nReport saved as: {report}"
        messagebox.showinfo("Result", result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectorApp(root)
    root.mainloop()
