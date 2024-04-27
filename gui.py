



from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label

# Import to run when clicking "run"
# This will be running the Training later
# The "run" button is Button_9
import subprocess


from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3d plotting
from matplotlib.widgets import Slider

import pyorb
import time
import csv


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Initialize the subprocess variable to None
process = None
output_file = "output.txt"

def toggle_script():
    global process
    # If there's no process running, start it
    if process is None:
        # Update the button to say "Kill Script"
        button_9.config(image=button_image_kill)
        # Start the external script and keep its process
        with open(output_file, "a") as output:
            process = subprocess.Popen(["python","display_output.py"],stdout=output, stderr=subprocess.STDOUT)
        button_9['command'] = kill_script

def kill_script():
    global process
    # Terminate the process if it exists
    if process is not None:
        process.terminate()
        process = None
        # Update the button to say "Run Script"
        button_9.config(image=button_image_9, command=toggle_script)

num = 1000
ax_lims = 7000000
asymtote_limit = 0.99

# Reading in specific Kepler state file
with open('results/state/88082/88082_TD3_state_kepler_252.txt', 'r') as f:
    lines = f.readlines()[1:]

# Capture data
data = []
for i in range(len(lines)):
    a, e, i, omega, Omega, nu = [float(x) for x in lines[i].split(',')]
    data.append([a, e, i, omega, Omega, nu])

orbits = np.array(data).T

orb = pyorb.Orbit(
    M0 = 1.0,
    G = pyorb.get_G(length='AU', mass='Msol', time='y'),
    num = num,
    a = 1.0, 
    e = 0, 
    i = 0, 
    omega = 0, 
    Omega = 0, 
    anom = np.linspace(0, 360, num=num),
    degrees = True,
    type = 'true',
)
# for some optimization
orb.direct_update = False

target = pyorb.Orbit(
    M0 = 1.0,
    G = pyorb.get_G(length='AU', mass='Msol', time='y'),
    num = num,
    a = 6300000, 
    e = 0.23*4, 
    i = 5.3*4, 
    omega = 24.0*4, 
    Omega = 24.0*4, 
    anom = np.linspace(0, 360, num=num),
    degrees = True,
    type = 'true',
)

# def add_vel(ax):
#     r = orb._cart[:3, 0]
#     v = orb._cart[3:, 0]
#     vel = ax.quiver(
#         r[0], r[1], r[2],
#         v[0], v[1], v[2],
#         length=ax_lims*0.05,
#     )
#     return vel

def plot():
    fig = Figure(figsize=(7, 7))  # Adjust size as needed
    ax = fig.add_subplot(111, projection='3d')

    # Define initial plot setup
    r = orb.r
    t = target.r
    l, = ax.plot(r[0, :], r[1, :], r[2, :], '-b', label='Current State')
    fin, = ax.plot(t[0, :], t[1, :], t[2, :], '-r', label='Target State')
    dot, = ax.plot([r[0, 0]], [r[1, 0]], [r[2, 0]], 'ob')
    ax.plot([0], [0], [0], 'og', label='Earth')

    ax.legend(loc="upper left")
    ax.set_title('Orbit Visualization', fontsize=22)
    ax.set_xlabel('X-position [m]', fontsize=15, labelpad=20)
    ax.set_ylabel('Y-position [m]', fontsize=15, labelpad=20)
    ax.set_zlabel('Z-position [m]', fontsize=15, labelpad=20)
    ax.set_xlim([-ax_lims, ax_lims])
    ax.set_ylim([-ax_lims, ax_lims])
    ax.set_zlim([-ax_lims, ax_lims])

    # Embed the Figure in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.place(x=742, y=46, width=621, height=521)

    # Create and place sliders
    axcolor = 'lightgoldenrodyellow'
    ax_a = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_e = fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_i = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_omega = fig.add_axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_Omega = fig.add_axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_nu = fig.add_axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)

    s_a = Slider(ax_a, 'a [m]', 5400*1e3, 6400*1e3, valinit=6300000)
    s_e = Slider(ax_e, 'e [1]', 0, 1, valinit=0.23)
    s_i = Slider(ax_i, 'i [deg]', 0, 360, valinit=5.3)
    s_omega = Slider(ax_omega, 'omega [deg]', 0, 360, valinit=24)
    s_Omega = Slider(ax_Omega, 'Omega [deg]', 0, 360, valinit=24)
    s_nu = Slider(ax_nu, 'nu [deg]', 0, 360, valinit=180)

    def draw():
        r = orb.r
        t = target.r

        l.set_xdata(r[0, 1:])
        l.set_ydata(r[1, 1:])
        l.set_3d_properties(r[2, 1:])

        dot.set_xdata([r[0, 0]])
        dot.set_ydata([r[1, 0]])
        dot.set_3d_properties([r[2, 0]])

        fin.set_xdata(t[0, 1:])
        fin.set_ydata(t[1, 1:])
        fin.set_3d_properties(t[2, 1:])

        fig.canvas.draw_idle()

    def update_orb(val):
        a, e, i, omega, Omega, nu = val
        orb.a = a
        orb.e = e * 4 # x4 scaling to better observe slight element changes
        orb.i = i * 4
        orb.omega = omega * 4
        orb.Omega = Omega * 4
        orb._kep[5, 0] = nu * 4
        draw()

    def current_state():
        time.sleep(1)
        # Iterate through each state
        for i in range(len(lines)):
            a, e, i, omega, Omega, nu = [float(x) for x in lines[i].split(',')]
            update_orb([a, e, i, omega, Omega, nu])
            s_a.set_val(a)
            s_e.set_val(e)
            s_i.set_val(i)
            s_omega.set_val(omega)
            s_Omega.set_val(Omega)
            s_nu.set_val(nu)
            # plt.pause(0.1)

    current_state()

    # Initial draw of the plot
    canvas.draw_idle()

window = Tk()

window.geometry("1400x652")
window.configure(bg = "#1A1E20")


# Export Button Functionality
def export_to_csv(filename="exported_data.txt"):
    # Not implemented as a separate thread...
    command = "python main.py"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        with open(filename, 'w') as file: file.write(result.stdout)
    except subprocess.CalledProcessError as e:
        print("AHHHHH: ", e.output)
    except Exception as e:
        print( "BAKBKUSBAFkuVAFu: ", str(e))




# Saved Data is in format:
'''
data_set_#
coordinates
intial, a, e, i, omega, w, v
final, a, e, i, omega, w, v
duration, max_thrust, max_speed
date_1, date_2, date_3, date_4, date_5, date_6
mass_1, mass_2
'''

file_path = "saved_data.txt"

def parse_data(file_path):
    parsed_data = {
        "data_set_number": None,
        "coordinate_type": None,
        "coordinates": {
            "initial": {},
            "final": {}
        },
        "stats": {
            "duration": None,
            "max_thrust": None,
            "max_speed": None
        },
        "date": [],
        "masses": []
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    print("LINES: ", lines)
    
    for i, line in enumerate(lines):
        parts = line.strip().split(', ')
        
        if parts[0] == '1':
            parsed_data["data_set_number"] = parts[0]
        elif parts[0] in ['keplerian','cartesian']:
            parsed_data["coordinate_type"] = line
        elif parts[0] in ['initial', 'final']:
            parsed_data["coordinates"][parts[0]] = {
                "a": eval(parts[1]),
                "e": float(parts[2]),
                "i": float(parts[3]),
                "omega": float(parts[4]),
                "w": float(parts[5]),
                "v": float(parts[6])
            }
        elif parts[0] == 'results':
            parsed_data["stats"] = {
                "duration": eval(parts[1]),
                "max_thrust": float(parts[2]),
                "max_speed": float(parts[3])
            }
        elif parts[0] == 'date':
            parsed_data["date"] = [parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]]
        elif parts[0] == 'mass':
            parsed_data["masses"] = [parts[1], parts[2]]
    
    return parsed_data

parsed_data = parse_data('saved_data.txt')
print("STATS: ", parsed_data["date"])




canvas = Canvas(
    window,
    bg = "#1A1E20",
    height = 652,
    width = 1400,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    725.0,
    28.0,
    1382.0,
    651.0,
    fill="#1A1E20",
    outline="#049364")

# Visualizer Rectangle Location
canvas.create_rectangle(
    742.0,
    46.0,
    1350.0,
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

# Entry_1 is DURATION
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
entry_1.insert("end", str(parsed_data["stats"]["duration"]))


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

# Entry_2 is MAX_THRUST
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
entry_2.insert("end", "MAX_THRUST")

canvas.create_text(
    182.0,
    436.0,
    anchor="nw",
    text="max thrust",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

# Entry_3 is DATE
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
entry_3.insert("end", parsed_data["date"])

canvas.create_text(
    348.0,
    436.0,
    anchor="nw",
    text="date",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

# Entry_4 is MAX_SPEED 
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
entry_4.insert("end", "MAX_SPEED")

canvas.create_text(
    182.0,
    511.0,
    anchor="nw",
    text="max speed",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

# Entry_5 is MASSES
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
entry_5.insert("end", parsed_data["masses"])

canvas.create_text(
    19.0,
    511.0,
    anchor="nw",
    text="mass",
    fill="#049364",
    font=("Menlo Regular", 20 * -1)
)

# Entry_6 is Initial a
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
entry_6.insert("end", parsed_data["coordinates"]["initial"]['a'])

# Entry_7 is initial i
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
entry_7.insert("end", parsed_data["coordinates"]["initial"]['i'])

# Entry_8 is Initial w
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
entry_8.insert("end", parsed_data["coordinates"]["initial"]['w'])


# Entry_9 is Initial e
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
entry_9.insert("end", parsed_data["coordinates"]["initial"]['e'])

# Entry_10 is Initial Omega
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
entry_10.insert("end", parsed_data["coordinates"]["initial"]['omega'])

# Entry_11 is Initial v
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
entry_11.insert("end", parsed_data["coordinates"]["initial"]['v'])


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

# Entry_12 is Final a
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
entry_12.insert("end", parsed_data["coordinates"]["final"]['a'])

# Entry_13 is final i
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
entry_13.insert("end", parsed_data["coordinates"]["final"]['i'])

# Entry_14 is Final w
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
entry_14.insert("end", parsed_data["coordinates"]["final"]['w'])

# Entry_15 is Final e
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
entry_15.insert("end", parsed_data["coordinates"]["final"]['e'])


# Entry_16 is Final omega
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
entry_16.insert("end", parsed_data["coordinates"]["final"]['omega'])

# Entry_17 is final v
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
entry_17.insert("end", parsed_data["coordinates"]["final"]['v'])


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

##visualizer
button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    # command=lambda: print("button_7 clicked"),
    command = plot,
    relief="flat"
)
button_7.place(
    x=741.39013671875,
    y=586.1273803710938,
    width=147.24473571777344,
    height=51.66482162475586
)

# Export Button
button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    #command=lambda: export_to_csv(),
    relief="flat"
)
button_8.place(
    x=1182.297607421875,
    y=586.1273803710938,
    width=160.16094970703125,
    height=51.66482162475586
)

# Load iimage for Kill Command
button_image_kill = PhotoImage(file=relative_to_assets("button_9_kill.png"))  # Your "Kill Script" button image


button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    # command=lambda: print("button_9 clicked"),
    command=lambda: toggle_script(),
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
