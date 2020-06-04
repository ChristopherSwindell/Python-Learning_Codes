import random

out_list = []

while True:
    numgen=random.randint(0,3)

    if numgen == 1:
        opp_hand = 'Rock'
    elif numgen == 2:
        opp_hand = 'Paper'
    elif numgen == 3:
        opp_hand = 'Scissors'


    your_hand = input("Rock, Paper, or Scissors?     ")
    print("Your opponent chose", opp_hand,".")

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
    print("You", outcome,".")
    print("You have won",out_list.count('won'), "out of",len(out_list),"games.")

    if input("Do you want to continue? Y/N     ") not in ['Y','y','Yes','yes','YES']:
        break

