



from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1182x652")
window.configure(bg = "#1A1E20")


canvas = Canvas(
    window,
    bg = "#1A1E20",
    height = 652,
    width = 1182,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    725.0,
    28.0,
    1182.0,
    651.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    742.0,
    46.0,
    1165.0,
    567.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    536.0,
    417.0,
    702.0,
    580.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    0.0,
    417.0,
    523.0,
    587.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    0.0,
    163.0,
    337.0,
    397.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    357.0,
    163.0,
    700.0,
    397.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    335.0,
    28.0,
    700.0,
    120.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_rectangle(
    0.0,
    28.0,
    323.0,
    120.0,
    fill="#1A1E20",
    outline="#049364")

canvas.create_text(
    3.0,
    0.0,
    anchor="nw",
    text="OPTIMIZATION",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    337.0,
    0.0,
    anchor="nw",
    text="COORDINATES",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    725.0,
    3.0,
    anchor="nw",
    text="OUTPUT",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    85.5,
    482.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_1.place(
    x=16.0,
    y=465.0,
    width=139.0,
    height=32.0
)


canvas.create_text(
    16.0,
    436.0,
    anchor="nw",
    text="duration",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    28.0,
    178.0,
    anchor="nw",
    text="a",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    28.0,
    250.0,
    anchor="nw",
    text="i",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    28.0,
    319.0,
    anchor="nw",
    text="ω",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    166.0,
    178.0,
    anchor="nw",
    text="e",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    167.0,
    250.0,
    anchor="nw",
    text="Ω",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    167.0,
    319.0,
    anchor="nw",
    text="ν",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    251.5,
    482.0,
    image=entry_image_2
)
entry_2 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_2.place(
    x=182.0,
    y=465.0,
    width=139.0,
    height=32.0
)

canvas.create_text(
    182.0,
    436.0,
    anchor="nw",
    text="max thrust",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    418.5,
    482.0,
    image=entry_image_3
)
entry_3 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_3.place(
    x=348.0,
    y=465.0,
    width=141.0,
    height=32.0
)

canvas.create_text(
    348.0,
    436.0,
    anchor="nw",
    text="date",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    251.5,
    558.0,
    image=entry_image_4
)
entry_4 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_4.place(
    x=182.0,
    y=542.0,
    width=139.0,
    height=30.0
)

canvas.create_text(
    182.0,
    511.0,
    anchor="nw",
    text="max speed",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    85.5,
    558.0,
    image=entry_image_5
)
entry_5 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_5.place(
    x=16.0,
    y=542.0,
    width=139.0,
    height=30.0
)

canvas.create_text(
    19.0,
    511.0,
    anchor="nw",
    text="mass",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    88.5,
    225.0,
    image=entry_image_6
)
entry_6 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_6.place(
    x=24.0,
    y=208.0,
    width=129.0,
    height=32.0
)

entry_image_7 = PhotoImage(
    file=relative_to_assets("entry_7.png"))
entry_bg_7 = canvas.create_image(
    89.5,
    295.5,
    image=entry_image_7
)
entry_7 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_7.place(
    x=25.0,
    y=279.0,
    width=129.0,
    height=31.0
)

entry_image_8 = PhotoImage(
    file=relative_to_assets("entry_8.png"))
entry_bg_8 = canvas.create_image(
    91.0,
    366.0,
    image=entry_image_8
)
entry_8 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_8.place(
    x=27.0,
    y=351.0,
    width=128.0,
    height=28.0
)

entry_image_9 = PhotoImage(
    file=relative_to_assets("entry_9.png"))
entry_bg_9 = canvas.create_image(
    230.5,
    225.0,
    image=entry_image_9
)
entry_9 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_9.place(
    x=166.0,
    y=208.0,
    width=129.0,
    height=32.0
)

entry_image_10 = PhotoImage(
    file=relative_to_assets("entry_10.png"))
entry_bg_10 = canvas.create_image(
    230.5,
    295.5,
    image=entry_image_10
)
entry_10 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_10.place(
    x=166.0,
    y=279.0,
    width=129.0,
    height=31.0
)

entry_image_11 = PhotoImage(
    file=relative_to_assets("entry_11.png"))
entry_bg_11 = canvas.create_image(
    231.5,
    366.0,
    image=entry_image_11
)
entry_11 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_11.place(
    x=167.0,
    y=351.0,
    width=129.0,
    height=28.0
)

canvas.create_text(
    3.0,
    134.0,
    anchor="nw",
    text="INITIAL POSITIONS",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    392.0,
    178.0,
    anchor="nw",
    text="a",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    392.0,
    250.0,
    anchor="nw",
    text="i",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    393.0,
    319.0,
    anchor="nw",
    text="ω",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    532.0,
    178.0,
    anchor="nw",
    text="e",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    532.0,
    250.0,
    anchor="nw",
    text="Ω",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

canvas.create_text(
    534.0,
    319.0,
    anchor="nw",
    text="ν",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

entry_image_12 = PhotoImage(
    file=relative_to_assets("entry_12.png"))
entry_bg_12 = canvas.create_image(
    454.5,
    225.0,
    image=entry_image_12
)
entry_12 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_12.place(
    x=390.0,
    y=208.0,
    width=129.0,
    height=32.0
)

entry_image_13 = PhotoImage(
    file=relative_to_assets("entry_13.png"))
entry_bg_13 = canvas.create_image(
    456.5,
    295.5,
    image=entry_image_13
)
entry_13 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_13.place(
    x=392.0,
    y=279.0,
    width=129.0,
    height=31.0
)

entry_image_14 = PhotoImage(
    file=relative_to_assets("entry_14.png"))
entry_bg_14 = canvas.create_image(
    456.5,
    366.0,
    image=entry_image_14
)
entry_14 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_14.place(
    x=392.0,
    y=351.0,
    width=129.0,
    height=28.0
)

entry_image_15 = PhotoImage(
    file=relative_to_assets("entry_15.png"))
entry_bg_15 = canvas.create_image(
    596.0,
    225.0,
    image=entry_image_15
)
entry_15 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_15.place(
    x=531.0,
    y=208.0,
    width=130.0,
    height=32.0
)


entry_image_16 = PhotoImage(
    file=relative_to_assets("entry_16.png"))
entry_bg_16 = canvas.create_image(
    597.5,
    295.5,
    image=entry_image_16
)
entry_16 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_16.place(
    x=532.0,
    y=279.0,
    width=131.0,
    height=31.0
)

entry_image_17 = PhotoImage(
    file=relative_to_assets("entry_17.png"))
entry_bg_17 = canvas.create_image(
    597.5,
    366.0,
    image=entry_image_17
)
entry_17 = Text(
    bd=0,
    bg="#1A1E20",
    fg="#049364",
    highlightthickness=0
)
entry_17.place(
    x=532.0,
    y=351.0,
    width=131.0,
    height=28.0
)

canvas.create_text(
    364.0,
    134.0,
    anchor="nw",
    text="FINAL POSITIONS",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=19.37432861328125,
    y=48.167572021484375,
    width=129.16204833984375,
    height=51.66482162475586
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=355.19561767578125,
    y=48.167572021484375,
    width=154.9944610595703,
    height=51.66482162475586
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=527.6270141601562,
    y=48.167572021484375,
    width=154.9944610595703,
    height=51.66482162475586
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat"
)
button_4.place(
    x=174.3687744140625,
    y=48.8133544921875,
    width=129.16204833984375,
    height=51.66482162475586
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_5 clicked"),
    relief="flat"
)
button_5.place(
    x=545.709716796875,
    y=431.7781982421875,
    width=147.24473571777344,
    height=51.66482162475586
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_6 clicked"),
    relief="flat"
)
button_6.place(
    x=545.709716796875,
    y=511.21307373046875,
    width=147.24473571777344,
    height=51.66482162475586
)

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_7 clicked"),
    relief="flat"
)
button_7.place(
    x=741.39013671875,
    y=586.1273803710938,
    width=147.24473571777344,
    height=51.66482162475586
)

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_8 clicked"),
    relief="flat"
)
button_8.place(
    x=1002.297607421875,
    y=586.1273803710938,
    width=160.16094970703125,
    height=51.66482162475586
)

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_9 clicked"),
    relief="flat"
)
button_9.place(
    x=2.583251953125,
    y=600.3355712890625,
    width=698.1209106445312,
    height=51.66482162475586
)


window.resizable(False, False)
window.mainloop()
