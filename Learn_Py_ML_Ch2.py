import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D #for 3D projections

'''Distributions'''
##b = ss.distributions.binom
##n = ss.distributions.norm
##for flips in [5, 10, 20, 40, 80]:
##    success = np.arange(flips)
##    our_distribution = b.pmf(success, flips, .5)
##    plt .hist(success, flips, weights=our_distribution)
##    mu = flips * 0.5,
##    std_dev = np.sqrt(flips* 0.5 * (1-0.5))
##    norm_x = np.linspace(mu-3*std_dev, mu+3*std_dev)
##    norm_y = n.pdf(norm_x, mu, std_dev)
##    plt.plot(norm_x, norm_y, 'k');
##    
##plt.xlim(0,55)
##plt.show()

'''Linear Combinations, Weighted Sums, and Dot Products'''
##Method 1: Traditional Python
##quantity = [2, 12, 3]
##costs = [12.5, .5, 1.75]
##partial_cost = []
##for q, c in zip(quantity, costs):
##    partial_cost.append(q*c)
##    print(q, " * ", c, " = ", q*c)
##print("----------------")
##print(sum(partial_cost))

##Method 2: Also traditional Python
##quantity = [2, 12, 3]
##costs = [12.5, .5, 1.75]
##total = sum(q*c for q,c in zip(quantity, costs))
##print(total)

##Method 3: Using numpy arrays
##quantity = np.array([2, 12, 3])
##costs = np.array([12.5, .5, 1.75])
##total = np.sum(quantity * costs)
##print(total)

##Method 4: Also using numpy arrays
##quantity = np.array([2, 12, 3])
##costs = np.array([12.5, .5, 1.75])
##for q_i, c_i in zip(quantity, costs):
##    print("{:2d} {:5.2f} --> {:5.2f}".format(q_i, c_i, q_i * c_i))
##
##print("Total:", sum(q*c for q,c in zip(quantity, costs)))

'''Weighted Average'''
##values = np.array([10,20,30])
##weights = np.full_like(values, 1/3, dtype=np.double) #repeated (1/3)
##print("weights:", weights)
##print("via mean:", np.mean(values))
##print("via weights and dot:", np.dot(weights, values))

##values = np.array([10,20,30])
##weights = np.array([.5,.25,.25])
##wa = np.dot(weights, values)
##print(wa)

##Payoff example
##payoffs = np.array([1.0, -.5])
##probs = np.array([.5,.5])
##expval = np.dot(payoffs, probs)
##print(expval)

##Another Payoff example
##def is_even(n):
##    return n % 2 == 0
##win_records = []
##for i in range(10):
##    winnings = 0 
##    for toss_ct in range(10000):
##        die_toss = np.random.randint(1,7)
##        winnings += 1 if is_even(die_toss) else -.5
##    win_records.append(winnings)
##print(win_records)
##print("average: ", np.mean(win_records))

'''Sums of Squares'''
##values = np.array([5, -3, 2, 1])
##squares = values * values
##print(squares, np.sum(squares), np.dot(values,values), sep = "\n")

'''Sum of Squared Errors'''
##errors = np.array([5, -5, 3.2, -1.1])
##sq_errors = pd.DataFrame({'errors': errors,
##                      'squared': errors*errors})
##print(sq_errors)
##print(np.dot(errors,errors))

'''Lines'''
##people = np.arange(1,11)
##total_cost = np.ones_like(people) * 40.0
##ax = plt.gca()
##ax.plot(people, total_cost)
##ax.set_xlabel("# People")
##ax.set_ylabel("Cost\n(Parking Only)")
##plt.show()

##people = np.arange(1,11)
##total_cost = 80 * people + 40
##tot_cost_df = pd.DataFrame({'total_cost':total_cost.astype(np.int)},
##                           index=people).T
##print(tot_cost_df)
##ax = plt.gca()
##ax.plot(people,total_cost, 'bo')
##ax.set_ylabel("Total Cost")
##ax.set_xlabel("People")
##plt.show()

'''Beyond Lines'''
##number_people = np.arange(1,11) #1 to 10 people
##number_rbs = np.arange(0,20) #0 to 20 rootbeers
##number_people, number_rbs = np.meshgrid(number_people, number_rbs)
##total_cost = 80 * number_people + 10 * number_rbs + 40
##fig,axes = plt.subplots(2, 3,
##                        subplot_kw={'projection':'3d'},
##                        figsize=(9,6))
##angles = [0, 45, 90, 135, 180]
##for ax,angle in zip(axes.flat, angles):
##    ax.plot_surface(number_people, number_rbs, total_cost)
##    ax.set_xlabel("People")
##    ax.set_ylabel("RootBeers")
##    ax.set_zlabel("TotalCost")
##    ax.azim = angle
##axes.flat[-1].axis('off')
##fig.tight_layout()
##plt.show()

'''Notation and the Plus-One Trick'''
'''Polynomials and Nonlinearity'''
##fig, axes = plt.subplots(2,2)
##fig.tight_layout()
##titles = ["$y=c_0$",
##          "$y=c_1x+c_0$",
##          "$y=c_2x^2+c_1x+c_0$",
##          "$y=c_3x^3+c_2x^2+c_1x+c_0$"
##          ]
##xs= np.linspace(-10,10,100)
##for power, (ax, title) in enumerate(zip(axes.flat, titles), 1):
##    coeffs = np.random.uniform(-5,5,power)
##    poly = np.poly1d(coeffs)
##    ax.plot(xs, poly(xs))
##    ax.set_title(title)
##plt.show()

##plt.Figure((2, 1.5))
##xs = np.linspace(-10, 10, 101)
##coeffs = np.array([2, 3, 4])
##ys = np.dot(coeffs, [xs**2, xs**1, xs**0])
##plt.plot(xs, ys);
##plt.show()

'''NumPy versus All the Maths'''
##oned_vec = np.arange(5)
##print(oned_vec, "-->", oned_vec * oned_vec)
##print("self dot:", np.dot(oned_vec, oned_vec))

##row_vec = np.arange(5).reshape(1, 5)
##col_vec = np.arange(0, 50, 10).reshape(5, 1)
##print("row vec:", row_vec,
##      "colvec:", col_vec,
##      "dot:", np.dot(row_vec, col_vec), sep='\n' )
##out = np.dot(col_vec, row_vec)
##print(out)

'''1D versus 2D'''
##col_vec = np.arange(0, 50, 10).reshape(5, 1)
##row_vec = np.arange(0, 5).reshape(1,5)
##oned_vec = np.arange(5)
##out = np.dot(oned_vec, col_vec)
##print(col_vec)
##print(row_vec)
##print(out)

##D = np.array([[1,3],
##              [2,5],
##              [2,7],
##              [3,2]])
##w = np.array([1.5, 2.5])
##print(np.dot(D,w))
##def rdot(arr,brr):
##    'reversed-argument verion of np.dot'
##    return np.dot(brr,arr)
##print(rdot(w, D))

'''Floating Point Issues'''
print(1.1 + 2.2 == 3.3)
print(type(1.1),type(2.2),type(1.1+2.2),type(3.3))
#Are the numbers close enough?
print(np.allclose(1.1 + 2.2, 3.3))
