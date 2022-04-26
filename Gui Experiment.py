import tkinter.filedialog as filedialog
import tkinter as tk
from ocr import Ocr
master = tk.Tk()


def input():
    input_path2 = tk.filedialog.askopenfilename()
    input_entry.delete(0, tk.END)  # Remove current text in entry
    input_entry.insert(0, input_path2)  # Insert the 'path'


def start_point():
    path = input_entry.get()
    Ocr(path).main()


master.title("OCR Program")
top_frame = tk.Frame(master)
bottom_frame = tk.Frame(master)
line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')
input_path = tk.Label(top_frame, text="Input File Path:")
input_entry = tk.Entry(top_frame, text="", width=40)
browse1 = tk.Button(top_frame, text="Browse", command=input)
begin_button = tk.Button(bottom_frame, text='Execute!', command=start_point)
top_frame.pack(side=tk.TOP)
line.pack(pady=10)
bottom_frame.pack(side=tk.BOTTOM)

input_path.pack(pady=5)
input_entry.pack(pady=5)
browse1.pack(pady=5)

begin_button.pack(pady=20, fill=tk.X)

master.mainloop()

print("Program has closed.")

