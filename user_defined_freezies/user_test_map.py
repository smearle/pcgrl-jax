import os
import tkinter as tk
import numpy as np
from datetime import datetime

SAVE_FOLDER = "user_defined_freezies"

class BinaryBoard(tk.Tk):
    def __init__(self, rows, cols, num_states=4):
        super().__init__()
        self.title("Create your own freezies!")

        self.rows = rows
        self.cols = cols
        self.num_states = num_states
        self.board = np.zeros((rows, cols), dtype=int)

        # Load images
        self.state_images = []
        self.state_images.append(tk.PhotoImage(file='envs/probs/tile_ims/empty.png'))
        self.state_images.append(tk.PhotoImage(file='envs/probs/tile_ims/solid.png'))
        self.state_images.append(tk.PhotoImage(file='envs/probs/tile_ims/player.png'))
        self.state_images.append(tk.PhotoImage(file='envs/probs/tile_ims/key.png'))

        self.buttons = [[None] * cols for _ in range(rows)]

        self.dragging = False
        self.toggled_during_drag = set()

        # Create a frame for the buttons
        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=0, sticky='nsew')

        for row in range(rows):
            for col in range(cols):
                btn = tk.Button(button_frame, image=self.state_images[0])
                btn.grid(row=row, column=col, sticky='news')
                btn.bind('<Button-1>', lambda e, row=row, col=col: self.on_click(row, col))
                btn.bind('<ButtonRelease-1>', self.on_release)
                self.buttons[row][col] = btn

        # Handle drag event at the tkinter.Tk level
        self.bind('<B1-Motion>', self.on_drag)

        # Add a Text widget to display the array, with more space allocated to it
        self.array_display = tk.Text(self, height=rows, width=40)  # Allocate more width
        self.array_display.grid(row=0, column=1, sticky='nsew')

        # Create a frame for Save and Copy and Reset buttons and place it below the Text widget
        frame = tk.Frame(self)
        frame.grid(row=1, column=0, columnspan=2, sticky='ew')  # span across both columns

        # Add Save, Copy, and Reset buttons to the frame
        copy_button = tk.Button(frame, text='Copy', command=self.copy_board)
        copy_button.pack(side='right', padx=5, pady=5)  

        save_button = tk.Button(frame, text='Save', command=self.save_board)
        save_button.pack(side='right', padx=5, pady=5)  

        reset_button = tk.Button(frame, text='Reset', command=self.reset_board)
        reset_button.pack(side='left', padx=5, pady=5)

        # Configure the grid to allocate more space to the Text widget
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)  # Allocate more space to the column containing the Text widget
        
        # Set the minimum window size and disable resizing
        self.minsize(600, 400)
        self.resizable(False, False)  # Disallow both horizontal and vertical resizing

    def on_click(self, row, col):
        self.board[row][col] = (self.board[row][col] + 1) % self.num_states  # Cycle through the states
        self.buttons[row][col].config(image=self.state_images[self.board[row][col]])
        self.toggled_during_drag.add((row, col))
        self.dragging = True
        self.display_board()

    def on_drag(self, event):
        if self.dragging:
            x, y = self.winfo_pointerxy()  # Get the global screen coordinates of the cursor
            for row in range(self.rows):
                for col in range(self.cols):
                    btn = self.buttons[row][col]
                    x0 = btn.winfo_rootx()  # Get the global screen x-coordinate of the left edge of the button
                    y0 = btn.winfo_rooty()  # Get the global screen y-coordinate of the top edge of the button
                    x1 = x0 + btn.winfo_width()  # Get the global screen x-coordinate of the right edge of the button
                    y1 = y0 + btn.winfo_height()  # Get the global screen y-coordinate of the bottom edge of the button
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        if (row, col) not in self.toggled_during_drag:
                            self.toggle_button(row, col)
                            self.toggled_during_drag.add((row, col))
                            self.display_board()  # Display updated board

    def on_release(self, event):
        self.dragging = False
        self.toggled_during_drag.clear()

    def toggle_button(self, row, col):
        self.board[row][col] = (self.board[row][col] + 1) % self.num_states  # Cycle through the states
        current_image = self.state_images[self.board[row][col]]
        self.buttons[row][col].config(image=current_image)

    def save_board(self):
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'binary_board_{timestamp_str}.npy'
        filename = os.path.join(SAVE_FOLDER, filename)
        np.save(filename, self.board)
        self.display_board()  # Display saved board
        self.array_display.insert(tk.END, f'\nNice! Your freezies is saved as {filename}')
        # print(f'Nice! Your freezies is saved as "{filename}"!')

    def copy_board(self):
        self.clipboard_clear()
        self.clipboard_append(str(self.board))
        self.update()

    def reset_board(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # Reset the board to all zeros
        for row in range(self.rows):
            for col in range(self.cols):
                self.buttons[row][col].config(image=self.state_images[0])  # Reset the button images to the first state
        self.display_board()  # Display the updated board

    def display_board(self):
        # Clear the Text widget and insert the new array
        self.array_display.delete('1.0', tk.END)
        self.array_display.insert(tk.END, str(self.board))


if __name__ == "__main__":
    app = BinaryBoard(16, 16)
    app.mainloop()
