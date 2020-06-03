import pandas as pd

##df  = pd.read_csv('pokemon_data.csv')

##print(df.head(3))
##print(df.tail(3))

##Read Headers
##print(df.columns)

##Read each Column
##print(df[['Name','Type 1','HP']][0:5])

##Read each Row
##print(df.iloc[1:4])
##for index, row in df.iterrow():
##    print(index,row['Name'])
##print(df.loc[df['Type 1'] == "Grass"])

##Read a specific Locatio (R,C)
##print(df.iloc[2,1])

##Basic Descriptive Statistics
##print(df.describe())

##Sort By Name
##print(df.sort_values(['Type 1', 'HP'], ascending=[1,0]))

##Create a Total column
##df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
##print(df.head(5))

##Drop the column
##df = df.drop(columns=['Total'])
##print(df.head(5))

##Create a Total column a different way
##df['Total'] = df.iloc[:, 4:10].sum(axis=1)
##
##cols = list(df.columns.values)
##df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
##
##print(df.head(5))

##Save updated file
##df.to_csv('modified_pokemon_data.csv', index = False)
##df.to_excel('modified_pokemon_data.xlsx', index = False)
##df.to_csv('modified_pokemon_data.txt', index = False, sep='\t')

##Show just Grass and Poison Then Grass or Poison
##print(df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') &(df['HP'] > 70)])
##print(df.loc[(df['Type 1'] == 'Grass') | (df['Type 2'] == 'Poison')])

##Reset index for new dataframe
##new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') &(df['HP'] > 70)]
##new_df.reset_index(drop = True, inplace = True)
##print(new_df.head(5))
##new_df.to_csv('filtered.csv')

##Filter out names that contain Mega
##df.loc[~df['Name'].str.contains('Mega')]
##print(df)

##Is Type 1 equal to fire or grass?
##import re
##print(df.loc[df['Type 1'].str.contains('fire|grass', flags = re.I, regex = True)])

##Get Pokemon names that start with PI
##import re
##print(df.loc[df['Name'].str.contains('^pi[a-z]*', flags = re.I, regex = True)])

##Conditional changing values and filtering
##df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
##df.loc[df['Type 1'] == 'Fire', 'Legendary'] = True

##Change multiple parameters
##df.loc[df['Total'] > 500, ['Generation','Legendary']] = 'Test Value'

##Alt
##df.loc[df['Total'] > 500, ['Generation','Legendary']] = ['Test 1', 'Test 2']

##Group By function for aggregate statistics
##print(df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False))
##print(df.groupby(['Type 1']).sum().sort_values('Defense', ascending=False))
##print(df.groupby(['Type 1']).count().sort_values('Defense', ascending=False))

##If we want to make sure that we count blank rows:
##df['count']=1
##print(df.groupby(['Type 1']).count()['count'])
##print(df.groupby(['Type 1','Type 2']).count()['count'])

##For very large datasets we can read a portion of the data at a time
##for df in pd.read_csv('pokemon_data.csv', chunksize=5):

##new_df = pd.DataFrame(columns = df.columns)
##for df in pd.read_csv('modified.csv', chunksize=5):
##    results = df.groupby(['Type 1']).count()
##
##    new_df = pd.concat([new_df,results])
