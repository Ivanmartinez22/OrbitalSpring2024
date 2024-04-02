import tkinter as tk
from tkinter.scrolledtext import ScrolledText  # For auto-scrolling text widget
import sys

# Function to setup the GUI
def setup_gui():
    root = tk.Tk()
    root.title("Console Output in GUI")

    # Create a scrolled text widget
    text_widget = ScrolledText(root, state='disabled', height=10)
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    return root, text_widget

root, text_widget = setup_gui()

class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string)
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)  # Auto-scroll to the end

    def flush(self):
        pass

sys.stdout = StdoutRedirector(text_widget)
sys.stderr = StdoutRedirector(text_widget)

def demo_output():
    print("This is a test output to the console.")

# Add a button to demonstrate live output
demo_button = tk.Button(root, text="Print Test Output", command=demo_output)
demo_button.pack(pady=5)

root.mainloop()
