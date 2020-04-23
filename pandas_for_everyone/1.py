import pandas as pd

## loads data from a csv file as a DataFrame
df = pd.read_csv('gapminder.tsv', sep='\t')

## you can check the type of an object using 'type(obj)'
#print(type(df))
## DataFrame


## prints the first few rows and columns
print(df.head())

## obj.shape prints the dimensions of a DataFrame, i.e the rows and columns
#print(df.shape)

## calls an Index of all the columns
#print(df.columns)

## prints the object types for each column of the dataframe
#print(df.dtypes)

## you're able to isolate a series of the dataframe by referencing its container
df_country = df['country'] # type Series

## you can them manipulate the container
# print(df_country.head())
# print(df['country'][-5:])


## you can select multiple rows by passing a list into the dataframe
# df_mixed = df[['country','continent','pop']]
## this creates another DataFrame object
# print(df_mixed.head())
# print(type(df_mixed))

# print(len(df))
## prints the location of the first row
# print(df.loc[1])

## prints the location of the indexed row at '1', the second value
# print (df.iloc[1])

## iloc allows us to pass -1 to get the last value, whereas this wasn't possible with 'loc'
# print (df.iloc[-1])

## prints cols 0-6, step 2
print(df.iloc[:, 0:6:2])



#### Grouped and Aggregated Calculations
# print(df.head(n=10))

### Grouped Means
## Try to find the average life expectancy for each year
# print(df.groupby('year')['lifeExp'].mean())
# print(df.groupby('year').head()) 
# print(df.dtypes)

multi_group_var = df.groupby(['year', 'continent'])[['lifeExp','gdpPercap']].mean()
print(multi_group_var)

