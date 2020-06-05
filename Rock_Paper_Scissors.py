import random

out_list = []

while True:
    hands = ['rock','paper','scissors']
    opp_hand=random.choice(hands)
    your_hand = input("Rock, Paper, or Scissors?     ")
    global outcome

    if opp_hand == your_hand:
        outcome = 'tied'
    elif opp_hand+your_hand in ['rockpaper','paperscissors','scissorsrock']:
        outcome = 'won'
    else:
        outcome = 'lost'

    out_list.append(outcome)

    print("Your opponent chose " + str(opp_hand) + ".")
    print("You " + str(outcome) + ".")
    print("You have won",out_list.count('won'), "out of",len(out_list),"games.")

    if input("Do you want to continue? Y/N     ") not in ['Y','y','Yes','yes','YES']:
        break
