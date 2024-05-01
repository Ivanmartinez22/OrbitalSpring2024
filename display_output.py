
import tkinter as tk
from tkinter import scrolledtext
import subprocess
from threading import Thread

class RealTimeOutputGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Command Line Output")
        root.geometry("800x600")
        root.configure(bg = "#1A1E20")
        process_thread = Thread(target=self.run_process)
        process_thread.start()

        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=30, width=100, bg="#1A1E20", fg="#049364", font=("Menlo Regular", 20*-1))
        self.output_text.pack(padx=10, pady=10)

    def run_process(self):
      # Use subprocess to run the command
      process = subprocess.Popen(["python","main.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

      # Read and display real-time output in the GUI
      with process.stdout:
         for line in iter(process.stdout.readline, ''):
            self.output_text.insert(tk.END, line)
            self.output_text.yview(tk.END)
            self.root.update()


# Create the Tkinter root window
root = tk.Tk()

# Instantiate the RealTimeOutputGUI class
real_time_output_gui = RealTimeOutputGUI(root)

root.mainloop()