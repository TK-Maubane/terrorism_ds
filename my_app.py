import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import streamlit as st

df = pd.read_csv("globalterrorismdb_0718dist.tar.bz2", compression="bz2")

df.head()
df.tail()


select_columns = """success 
suicide 
attacktype1
attacktype1_txt 
targtype1_txt 
targsubtype1_txt 
target1 
natlty1_txt 
gname 
gsubname 
nperps 
weaptype1_txt 
weapsubtype1_txt 
nkill
nkillus """
df.columns.all
print(df.columns.tolist())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df.tail()

df.loc[(df['region_txt']=='Central America & Caribbean') & (df['iyear']>=2008)][['iyear', 'region_txt']]
df2 = df.groupby(['iyear'], as_index=True).count()[['eventid']]

df2.plot.bar(xlabel='Year', ylabel='Number of attacks', title='Number of attacks per year (1970-2017)', figsize=(15, 8))
df.region_txt.unique()
df3 = df[['region_txt', 'iyear', 'eventid']].groupby(['region_txt', 'iyear']).count()
df3
df3.unstack(level=0)['eventid'].plot(kind='line', figsize=(15,11), grid=True)
plt.legend(title='Regions')
plt.title('Number of attacks per region, per year')

# show the graph
plt.show()
df3.unstack(level=0)
df3.tail(19)
#df3.loc['East Asia',2000][0]
df3.loc['Western Europe', 2000]['eventid']
df3.loc['Western Europe', 2000]['eventid']
# pretty sure there's a more elegant way to produce the below table, using .pivot_table
df4 = pd.DataFrame(index=df.iyear.unique(), columns=df.region_txt.unique())
df4.head()
for col in df4.columns:
    for ind in df4.index:
        try:
            df4.loc[ind][col] = df3.loc[col, ind]['eventid']
        except:
            df4.loc[ind][col]  = 0

df4.head(50)
def add_row_mean(df):
    # compute row-wise mean
    glob_mean = df.mean(axis=1)
    # add new column to DataFrame
    df['global_mean'] = glob_mean

add_row_mean(df4)
df4.sort_index().tail(30)
df4 = df4.sort_index()
df4.plot(figsize=(18, 15), grid=True, style=['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '--'])
plt.show()
df4[['Central America & Caribbean', 'Southeast Asia', 'Western Europe']].plot(grid=True)
plt.show()
## Question 1: How has the number of terrorist activities changed over the years? Are there certain regions where this trend is different from the global averages?

#A: Overall, there has been an increase in the number of attacks in the last 20 years. Before 1997, South Asia, Western Europe and South America had the most attacks. There was never more than 1500 in any of those regions though. Around 2001, South Asia started having more attacks, however the standout regions is by far the Middle East and North Africa. 2014 experienced the most attacks in a single region, and was the most active year for terrorist attacks. Sub-Saharn Africa started climbing above the global average around 2011. It's most active year was also 2014.
### Question 2

#### Q: Is the number of incidents and the number of casualties correlated? Can you spot any irregularities or outliers?


columns_of_interest = df[['iyear', 'imonth', 'iday', 'country', 'region', 'success', 'suicide', 'attacktype1', 'natlty1', 'claimed', 'weaptype1', 'nkill', 'nkillus']]
df_corr = df[['iyear', 'imonth', 'iday', 'country', 'region', 'success', 'suicide', 'attacktype1', 'natlty1', 'claimed', 'weaptype1', 'nkill', 'nkillus']].corr()

fig, ax = plt.subplots(figsize=(10,12))
im = ax.imshow(df_corr, interpolation='nearest')
fig.colorbar(im, orientation='vertical', fraction = 0.05)

# Show all ticks and label them with the dataframe column name
ax.set_xticks(np.arange(len(columns_of_interest.columns)))
ax.set_yticks(np.arange(len(columns_of_interest.columns)))

ax.set_xticklabels(columns_of_interest.columns, rotation=65, fontsize=15)
ax.set_yticklabels(columns_of_interest.columns, rotation=0, fontsize=15)

# Loop over data dimensions and create text annotations
for i in range(len(columns_of_interest.columns)-1):
    for j in range(len(columns_of_interest.columns)-1):
        text = ax.text(j, i, round(df_corr.to_numpy()[i, j], 2),
                       ha="center", va="center", color="white")

st.write(plt.show())
df5 = df[['region_txt', 'attacktype1_txt', 'eventid']].groupby(['region_txt', 'attacktype1_txt']).count().sort_values(['region_txt', 'eventid'], ascending=[True, False])
df5
# df6 = df[['iyear', 'attacktype1_txt', 'eventid']].groupby(['iyear', 'attacktype1_txt']).count().sort_values(['iyear', 'eventid'], ascending=False)
#df6 = df[['iyear', 'attacktype1_txt', 'eventid']].groupby(['iyear', 'attacktype1_txt']).count().sort_values(['iyear', 'eventid'], ascending=False)
df[['iyear', 'attacktype1_txt', 'eventid']].groupby(['iyear', 'attacktype1_txt']).count().sort_values(['iyear', 'eventid'], ascending=[True, False])
#need to redact the output above. 
#add diagrams to explain the data better
## Question 3: What are the most common methods of attacks? Does it differ in various regions or in time?

#Answer: The most common methods of attacks are bombings and armed assualt. Almost all regions have bombings in their top 2 attack types, throughout 1970-2017, the prefered terrorist attack type is explosions.  
st.title("Another attempt")
st.write(df4)
st.line_chart(data=df4, x=df4.index.all(), y=pd.Series(list(range(0, 5000, 1000))).all())