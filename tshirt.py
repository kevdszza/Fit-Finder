import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
import numpy as np

# T-shirt outline generator with details and curves
def generate_tshirt_outline(chest, waist, length, sleeve_length, shoulder_length, shoulder_to_waist, tshirt_color='blue'):
    shoulder_width = chest * 0.25
    body_width_top = chest * 0.8  # Upper part of the body, near chest
    body_width_bottom = waist * 0.9  # Waist area, slightly tapered
    body_length = length
    hem_curve_depth = body_length * 0.05  # Small curve for the hem at the bottom

    # Top half (chest area and shoulders)
    body_x_top = np.array([0, -shoulder_width/2, -body_width_top/2, -body_width_bottom/2, body_width_bottom/2, body_width_top/2, shoulder_width/2, 0])
    body_y_top = np.array([0, 0, -shoulder_to_waist, -body_length, -body_length, -shoulder_to_waist, 0, 0])

    # Curve for the bottom hem of the T-shirt
    hem_x = np.linspace(-body_width_bottom / 2, body_width_bottom / 2, 100)
    hem_y = np.sqrt(hem_curve_depth ** 2 - (hem_x ** 2 / (body_width_bottom / 2) ** 2)) - body_length  # Parabolic curve

    # Sleeves (rounder sleeves)
    sleeve_left_x = np.array([-shoulder_width / 2, -(shoulder_width / 2 + sleeve_length), -(shoulder_width / 2 + sleeve_length * 0.8), -shoulder_width / 2])
    sleeve_left_y = np.array([0, 0, -sleeve_length * 0.8, -shoulder_to_waist / 2])

    sleeve_right_x = np.array([shoulder_width / 2, shoulder_width / 2 + sleeve_length, shoulder_width / 2 + sleeve_length * 0.8, shoulder_width / 2])
    sleeve_right_y = np.array([0, 0, -sleeve_length * 0.8, -shoulder_to_waist / 2])

    # Plot the T-shirt with color and curves
    plt.figure(figsize=(6, 8))
    
    # Body and sleeves filled with color
    plt.fill(body_x_top, body_y_top, tshirt_color, alpha=0.7)
    plt.fill_between(hem_x, hem_y, -body_length, color=tshirt_color, alpha=0.7)
    plt.fill(sleeve_left_x, sleeve_left_y, tshirt_color, alpha=0.7)
    plt.fill(sleeve_right_x, sleeve_right_y, tshirt_color, alpha=0.7)

    # Optional: Add black outlines
    plt.plot(body_x_top, body_y_top, color='black', linewidth=2)
    plt.plot(hem_x, hem_y, color='black', linewidth=2)
    plt.plot(sleeve_left_x, sleeve_left_y, color='black', linewidth=2)
    plt.plot(sleeve_right_x, sleeve_right_y, color='black', linewidth=2)

    plt.xlim(-1.5 * shoulder_width, 1.5 * shoulder_width)
    plt.ylim(-1.5 * body_length, 0.5 * body_length)
    plt.title("Detailed T-shirt Design")
    plt.xlabel("Width (inches)")
    plt.ylabel("Length (inches)")
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(False)
    plt.axis('equal')
    # Removed the inversion of the y-axis
    plt.show()

# GUI button to generate T-shirt outline manually
def on_generate_manual():
    try:
        chest = float(chest_entry.get())
        waist = float(waist_entry.get())
        length = float(length_entry.get())
        sleeve_length = float(sleeve_entry.get())
        shoulder_length = float(shoulder_length_entry.get())
        shoulder_to_waist = float(shoulder_to_waist_entry.get())
        tshirt_color = color_entry.get()  # Get the color entered by the user
        generate_tshirt_outline(chest, waist, length, sleeve_length, shoulder_length, shoulder_to_waist, tshirt_color)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

# Create the GUI window
root = tk.Tk()
root.title("T-shirt Outline Generator")

# Frame for styling
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure style
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12), padding=10)

# Labels and Entry for measurements
tk.Label(frame, text="Chest Measurement (inches):").grid(row=0, column=0, sticky=tk.W)
chest_entry = ttk.Entry(frame, width=20)
chest_entry.grid(row=0, column=1)

tk.Label(frame, text="Waist Measurement (inches):").grid(row=1, column=0, sticky=tk.W)
waist_entry = ttk.Entry(frame, width=20)
waist_entry.grid(row=1, column=1)

tk.Label(frame, text="Length Measurement (inches):").grid(row=2, column=0, sticky=tk.W)
length_entry = ttk.Entry(frame, width=20)
length_entry.grid(row=2, column=1)

tk.Label(frame, text="Sleeve Length (inches):").grid(row=3, column=0, sticky=tk.W)
sleeve_entry = ttk.Entry(frame, width=20)
sleeve_entry.grid(row=3, column=1)

tk.Label(frame, text="Shoulder Length (inches):").grid(row=4, column=0, sticky=tk.W)
shoulder_length_entry = ttk.Entry(frame, width=20)
shoulder_length_entry.grid(row=4, column=1)

tk.Label(frame, text="Shoulder to Waist Length (inches):").grid(row=5, column=0, sticky=tk.W)
shoulder_to_waist_entry = ttk.Entry(frame, width=20)
shoulder_to_waist_entry.grid(row=5, column=1)

tk.Label(frame, text="T-shirt Color:").grid(row=6, column=0, sticky=tk.W)  # New color entry
color_entry = ttk.Entry(frame, width=20)
color_entry.grid(row=6, column=1)

# Button for manual generation
manual_button = ttk.Button(frame, text="Generate T-shirt Outline", command=on_generate_manual)
manual_button.grid(row=7, column=0, columnspan=2)

# Start the GUI loop
root.mainloop()
