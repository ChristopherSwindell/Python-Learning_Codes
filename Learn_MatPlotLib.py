
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##Create a histogram
population_ages = [22,55, 62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()

plt.show()

##Create a scatter plot
x=[1, 2, 3, 4, 5, 6, 7, 8]
y=[5, 2, 4, 2, 1, 4, 5, 2]

plt.scatter(x,y, label='skitscat', color = 'k', marker = '*', s = 100)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


##Create a stack plot

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8, 5, 7, 8, 13]

plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)

plt.stackplot(days, sleeping, eating, working, playing, colors = ['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()



##Create a plot for stock prices

TSLA = pd.read_csv("TSP.txt",
                   parse_dates = ['Date'])

'Look at the first 5 rows of data'
print(TSLA.head(5))
'Check dates formatted properly'
#print(type(TSLA['Date'][0]))

'Save file when done'
#df.to_csv('TSLA_modified.txt')


#Scatter Plot
'''
def scatter_graph(stock):
    plt.scatter(TSLA.Date,TSLA.CloseP, label='Prices', color='k', s=25, marker="o")
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Prices')
    plt.legend()
    plt.show()
scatter_graph(TSLA)
'''

#Graph
def graph_data(stock):
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))

    ax1.plot_date(TSLA.Date, TSLA.CloseP, '-',label = "Price")

    ax1.plot([],[],linewidth=5, label = 'loss', color='r', alpha=0.5)
    ax1.plot([],[],linewidth=5, label = 'gain', color='g', alpha=0.5)
    ax1.axhline(TSLA.CloseP[0], color='k', linewidth=5)

    ax1.fill_between(TSLA.Date, TSLA.CloseP,TSLA.CloseP[0],where=(TSLA.CloseP > TSLA.CloseP[0]), facecolor='g', alpha=0.5)
    ax1.fill_between(TSLA.Date, TSLA.CloseP,TSLA.CloseP[0],where=(TSLA.CloseP < TSLA.CloseP[0]), facecolor='r', alpha=0.5)

    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)#, color='g', linestyle='-', linewidth=1)
##    ax1.xaxis.label.set_color('c')
##    ax1.yaxis.label.set_color('r')
##    ax1.set_yticks([0,25,50,75])

    ax1.spines['left'].set_color('c')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_linewidth(5)
    ax1.tick_params(axis='x', colors='#f06215')
    
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(stock)
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()

graph_data('TSLA')
