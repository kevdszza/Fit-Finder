import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import sys

def generate_tshirt_outline(chest, waist, length, sleeve_length, shoulder_length, shoulder_to_waist, tshirt_color='blue'):
    shoulder_width = chest * 0.25
    body_width_top = chest * 0.8  # Upper part of the body, near chest
    body_width_bottom = waist * 0.9  # Waist area, slightly tapered
    body_length = length
    hem_curve_depth = body_length * 0.05  # Small curve for the hem at the bottom

    # Top half (chest area and shoulders)
    body_x_top = np.array([0, -shoulder_width / 2, -body_width_top / 2, -body_width_bottom / 2,
                           body_width_bottom / 2, body_width_top / 2, shoulder_width / 2, 0])
    body_y_top = np.array([0, 0, -shoulder_to_waist, -body_length, -body_length,
                           -shoulder_to_waist, 0, 0])

    # Curve for the bottom hem of the T-shirt
    hem_x = np.linspace(-body_width_bottom / 2, body_width_bottom / 2, 100)
    hem_y = np.sqrt(hem_curve_depth ** 2 - (hem_x ** 2 / (body_width_bottom / 2) ** 2)) - body_length  # Parabolic curve

    # Sleeves (rounder sleeves)
    sleeve_left_x = np.array([-shoulder_width / 2, -(shoulder_width / 2 + sleeve_length),
                               -(shoulder_width / 2 + sleeve_length * 0.8), -shoulder_width / 2])
    sleeve_left_y = np.array([0, 0, -sleeve_length * 0.8, -shoulder_to_waist / 2])

    sleeve_right_x = np.array([shoulder_width / 2, shoulder_width / 2 + sleeve_length,
                                shoulder_width / 2 + sleeve_length * 0.8, shoulder_width / 2])
    sleeve_right_y = np.array([0, 0, -sleeve_length * 0.8, -shoulder_to_waist / 2])

    # Plot the T-shirt with color and curves
    plt.figure(figsize=(6, 8))
    
    # Body and sleeves filled with color
    plt.fill(body_x_top, body_y_top, tshirt_color, alpha=0.7)
    plt.fill_between(hem_x, hem_y, -body_length, color=tshirt_color, alpha=0.7)
    plt.fill(sleeve_left_x, sleeve_left_y, tshirt_color, alpha=0.7)
    plt.fill(sleeve_right_x, sleeve_right_y, tshirt_color, alpha=0.7)

   
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
    plt.axis('equal')  # Maintain aspect ratio
    plt.show()

if __name__ == "__main__":
   
    try:
        chest = float(sys.argv[1].split('=')[1])
        waist = float(sys.argv[2].split('=')[1])
        length = float(sys.argv[3].split('=')[1])  # Updated to accept length
        sleeve_length = float(sys.argv[4].split('=')[1])
        shoulder_length = float(sys.argv[5].split('=')[1])
        shoulder_to_waist = float(sys.argv[6].split('=')[1])
        
        # Accept color as an additional argument
        tshirt_color = sys.argv[7].split('=')[1] if len(sys.argv) > 7 else 'blue'

        generate_tshirt_outline(chest, waist, length, sleeve_length, shoulder_length, shoulder_to_waist, tshirt_color)
    except (IndexError, ValueError):
        messagebox.showerror("Input Error", "Please provide valid measurements as command line arguments.")
        sys.exit()


    root = tk.Tk()
    root.withdraw()  
    root.mainloop()
