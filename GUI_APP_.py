# gui_app.py
import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
import io 

# --- 1. Load the Pre-trained Model ---
try:
    # Load the model that we saved from the training script
    model = tf.keras.models.load_model('digit_recognizer_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'digit_recognizer_model.h5' is in the same directory.")
    exit()

# --- 2. Define the GUI Application Class ---
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        # --- GUI Elements ---
        # Canvas for drawing
        self.canvas = Canvas(root, width=280, height=280, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=2)

        # Buttons
        self.predict_button = Button(root, text="Predict", command=self.predict_digit, width=15)
        self.predict_button.grid(row=1, column=0, pady=10, padx=5)

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas, width=15)
        self.clear_button.grid(row=1, column=1, pady=10, padx=5)

        # Label to display the prediction
        self.prediction_label = Label(root, text="Draw a digit (0-9)", font=("Helvetica", 16))
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Drawing Setup ---
        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos) # Reset on mouse release
        self.last_x, self.last_y = None, None

        # In-memory image to draw on
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        """Draws on the canvas when the left mouse button is held down."""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw on the Tkinter canvas for the user to see
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=20, fill='black',
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on the in-memory PIL image for processing
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=20)

        self.last_x = x
        self.last_y = y

    def reset_last_pos(self, event):
        """Resets the last mouse position when the button is released."""
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Clears the canvas and the in-memory image."""
        self.canvas.delete("all")
        # Re-initialize the drawing image
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit (0-9)")

    def predict_digit(self):
        """Processes the drawing and uses the model to predict the digit."""

        # --- Image Processing Pipeline ---

        # 1. Invert colors (model was trained on white digits on black background)
        img = ImageOps.invert(self.image)

        # 2. Resize the image to 28x28 pixels, same as MNIST images
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 3. Convert the image to a NumPy array
        img_array = np.array(img)

        # 4. Reshape the array for the model: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        # 5. Normalize the pixel values to be between 0 and 1
        img_array = img_array.astype('float32') / 255.0

        # --- Make Prediction ---
        try:
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            self.prediction_label.config(text=f"Predicted: {predicted_digit} ({confidence:.2f}%)")

        except Exception as e:
            self.prediction_label.config(text=f"Error: {e}")


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
