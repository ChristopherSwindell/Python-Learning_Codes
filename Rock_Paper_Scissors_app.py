import tkinter as tk
from tkinter import font
import random

HEIGHT = 569
WIDTH = 600

out_list = []

def clear_count():
    out_list.clear()
    
def play_game(your_hand):
    numgen=random.randint(0,3)

    if numgen == 1:
        opp_hand = 'Rock'
    elif numgen == 2:
        opp_hand = 'Paper'
    elif numgen == 3:
        opp_hand = 'Scissors'



    if opp_hand == your_hand:
        outcome = 'tied'
    elif opp_hand == 'Rock':
        if your_hand == 'Paper':
            outcome = 'lost'
        elif your_hand == 'Scissors':
            outcome = 'won'
    elif opp_hand == "Paper":
        if your_hand == "Rock":
            outcome = 'lost'
        elif your_hand == "Scissors":
            outcome = 'won'
    elif opp_hand == "Scissors":
        if your_hand == "Rock":
            outcome = 'won'
        elif your_hand == "Papaer":
            outcome = 'lost'

    out_list.append(outcome)
    your_choice = "You chose " + your_hand + ". " + "Your opponent chose "+ opp_hand + ". You " + outcome + ". You have won " + str(out_list.count('won')) + " out of " + str(len(out_list)) + " games."
    label.config(text = your_choice, wraplength=450)


    
root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width = WIDTH)
canvas.pack()

background_image = tk.PhotoImage(file='Rock_Paper_Scissors.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

choice_frame = tk.Frame(root, bg = '#80c1ff', bd=5)
choice_frame.place(relx=0.5, rely = 0.1,relwidth=.75, relheight=.1, anchor='n')

entry = tk.Entry(choice_frame, font=('Courier', 18))
entry.place(relwidth=0.45, relheight=1)

play_button = tk.Button(choice_frame, text="Play", font=('Courier', 12), command=lambda: play_game(entry.get()))
play_button.place(relx=0.48, relheight=1, relwidth=0.25)

clear_button = tk.Button(choice_frame, text="Clear", font=('Courier', 12), command=lambda: clear_count())
clear_button.place(relx=0.75, relheight=1, relwidth=0.25)

game_frame = tk.Frame(root, bg = '#80c1ff', bd=10)
game_frame.place(relx=0.5,rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')

label = tk.Label(game_frame, font=('Courier', 18), anchor='nw', justify='left', bd=4)
label.place(relwidth=1, relheight=1)

root.mainloop()

##Use cmd to save app to be used by other people
##pyinstaller.exe --onefile Rock_Paper_Scissors_app.py
