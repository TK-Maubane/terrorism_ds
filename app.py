import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px

import bar_chart_race as bcr
import streamlit as st

st.set_page_config(layout="wide")
st.title("Terrorism Data Exploration")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["About", "Question 1", "Question 2", "Question 3", "Question 4"])


@st.cache_data
def load_data():
    df = pd.read_csv("globalterrorismdb_0718dist.tar.bz2", compression="bz2", low_memory=False)
    return df

df = load_data()

with tab1:
    st.markdown("## The goal of this EDA is to explore the global terrorist attack trends between 1970 - 2017")

    st.dataframe(df, use_container_width=True)


with tab2:
    st.header("Question 1:")
    st.write("How has the number of terrorist activities changed over the years?")
    st.write("Are there certain regions where this trend is different from the global averages?")
    st.markdown("\n")
    st.write("""A: Overall, there has been an increase in the number of attacks in the last 20 years. Before 1997, South Asia, Western Europe and South America had the most attacks. 
    There was never more than 1500 in any of those regions though. Around 2001, South Asia started having more attacks, however the standout regions is by far the Middle East and North Africa. 
    2014 experienced the most attacks in a single region, and was the most active year for terrorist attacks. Sub-Saharn Africa started climbing above the global average around 2011. 
    It's most active year was also 2014.""")

    df2 = df.groupby(['iyear'], as_index=True).count()[['eventid']]

    fig = px.bar(data_frame=df2, title='Number of attacks per year (1970-2017)',labels={
                     "eventid": "Number of Attacks (thousands)",
                     "iyear": "Year"
                 })
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Number of attacks")
    st.write(fig)

    # defining the dataframe to chart
    df3 = df[['region_txt', 'iyear', 'eventid', 'nkill']].groupby(['region_txt', 'iyear']).agg({'eventid':'count', 'nkill':'sum'})

    fig_regions = px.line(data_frame=df3.unstack(level=0)['eventid'], 
                      title='Number of attacks per region, per year'
                      )
    fig_regions.update_yaxes(title="Number of attacks")

    # display chart
    st.write(fig_regions)

    # pretty sure there's a more elegant way to produce the below table, using .pivot_table
    df4 = pd.DataFrame(index=df.iyear.unique(), columns=df.region_txt.unique())

    for col in df4.columns:
        for ind in df4.index:
            try:
                df4.loc[ind][col] = df3.loc[col, ind]['eventid']
            except:
                df4.loc[ind][col]  = 0

    def add_row_mean(df):
        # compute row-wise mean
        glob_mean = df.mean(axis=1)
        # add new column to DataFrame
        df['global_mean'] = glob_mean

    add_row_mean(df4)
    df4 = df4.sort_index()

    df4.index.name = 'year'
    df_race = df4.apply(pd.to_numeric, errors='coerce', axis=1)
    cumulative_attacks_df = df_race.cumsum()

    # Race Bar Chart to better show the change in attacks numbers over the years for different regions
    
    @st.cache_data
    def show_bcr():
        chart = bcr.bar_chart_race(df=cumulative_attacks_df, filename=None,
                period_length=1000, 
                title="Cumulative count of terrorist attacks per region (1970 and 2017)",  
                perpendicular_bar_func='median')
        return chart

    st.write(show_bcr())
    

with tab3:
    st.header("Question 2:")
    st.write("Is the number of incidents and the number of casualties correlated?")
    st.write(" Can you spot any irregularities or outliers?")
    st.markdown("\n")
    st.write("The correlation is: 0.83")
    st.write("This means that there is very high correlation between the number of terrorist attacks in a region, and the number of confirmed deaths due to the attacks.")
    st.write("Outliers are East Asia and North America, with North America being the only extreme outlier. It actually has a negative, albeit neglegible correlation between its attack number and number of kills.")
    
    columns_of_interest = df[['iyear', 'imonth', 'iday', 'country', 'region', 'success', 'suicide', 'attacktype1', 'natlty1', 'claimed', 'weaptype1', 'nkill', 'nkillus']]
    df_corr = df[['iyear', 'imonth', 'iday', 'country', 'region', 'success', 'suicide', 'attacktype1', 'natlty1', 'claimed', 'weaptype1', 'nkill', 'nkillus']].corr()

    fig, ax = plt.subplots(figsize=(6,7))
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

    st.write(fig)
    all_region = df3.index.levels[0].tolist()
    tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13,tab14,tab15,tab16,tab17, = st.tabs(df3.index.levels[0].tolist())

    all_regions_correlations = []
    for region in df3.index.levels[0]:
        fig_2 = px.line(data_frame=df3.loc[region, :], title=f"{region}: Number of attacks vs number of kills")
        # may change back to use original st.write(fig_2) if line_chart disappoints
        all_regions_correlations.append(fig_2)
        
    
    with tab6:
        st.write(all_regions_correlations[0])
    with tab7:
        st.write(all_regions_correlations[1])
    with tab8:
        st.write(all_regions_correlations[2])
    with tab9:
        st.write(all_regions_correlations[3])
    with tab10:
        st.write(all_regions_correlations[4])
    with tab11:
        st.write(all_regions_correlations[5])
    with tab12:
        st.write(all_regions_correlations[6])
    with tab13:
        st.write(all_regions_correlations[7])
    with tab14:
        st.write(all_regions_correlations[8])
    with tab15:
        st.write(all_regions_correlations[9])
    with tab16:
        st.write(all_regions_correlations[10])
    with tab17:
        st.write(all_regions_correlations[11])


    # with st.sidebar:
    # region_chosen = st.radio("Select region to view:", options=df3.index.levels[0], horizontal=True)
    # st.write(all_regions_correlations[region_chosen])



    #correlation outliers section 
    correlation_df = pd.DataFrame(index=df3.index.levels[0], columns=['Correlation'])

    for region in df3.index.levels[0]:
        correlation_df.loc[region]['Correlation'] = df3.loc[region, :]['eventid'].corr(df3.loc[region, :]['nkill'])

    # I'm going to use the approach that a value is an outlier if it is more that 1 std deviation from the median, 
    # and an extreme outlier is a value more than 2 std deviation away for the median

    std_deviation = correlation_df['Correlation'].std()
    median = correlation_df['Correlation'].median()

    correlation_df['is_outlier'] = correlation_df['Correlation'] < median - std_deviation 
    correlation_df['is_extreme_outlier'] = correlation_df['Correlation'] < median - 2*std_deviation 

    # dataframe with nominal correlation values
    st.write(correlation_df)

    # scatterplot to correlation 
    corr_plot = px.scatter(data_frame=correlation_df['Correlation'])
    st.write(corr_plot)


with tab4:
    st.title("Question 3:")
    st.header(" What are the most common methods of attacks? Does it differ in various regions or in time?")

    st.write(""" Answer:
        The most common methods of attack are overwhelmingly Bombings and Exposions. Almost 50% of all terrorist attacks in between 1970 - 2017. 
        45 out of the 47 years in the study(96%) have bombings as the most common attack type, and the 2 years where the attack types aren't bombings also have the least amount of top attacks of all the years. 
        Regarding terms of regions, 5/6 regions have bombings as their most common attack type. Central America & The Caribbean have Armed Assault as its most common type, with bombings coming in second. 

        I imagine bombings/explosions are the prefered method of attack due to the efficacy of it, and that one can execute the attack/s remotely. """)
    
    attack_type_count_df = df[['iyear', 'attacktype1_txt', 'eventid']].groupby(['iyear', 'attacktype1_txt']).count().sort_values(['iyear', 'eventid'], ascending=[True, False])
    st.write(attack_type_count_df)

    # pie chart 
    attack_type_pie_chart = df[['attacktype1_txt', 'eventid']].groupby(['attacktype1_txt']).count().sort_values(['eventid'], ascending=False).plot.pie(subplots=True,
    autopct=lambda p: format(p, '.2f') if p > 4 else None) 

    st.write(attack_type_pie_chart)

    # no idea why the hell i named this df "overtime"
    overtime = pd.DataFrame(index=list(attack_type_count_df.index.levels[0]),
                                   columns=[['Top Attack Type', 'Amount']])
    
    
    for i in list(attack_type_count_df.index.levels[0]):
        # populate Amount column with top value for each year
        overtime.loc[i]['Amount'] = attack_type_count_df.loc[i].max()['eventid']

        # populate top type of attack for each year
        overtime.loc[i]['Top Attack Type'] = attack_type_count_df.loc[i].iloc[0].name
    
    st.write(overtime)

    # popular attack types by region 
    attack_type_count_regions = df[['region_txt', 'attacktype1_txt', 'eventid']].groupby(['region_txt', 'attacktype1_txt']).count().sort_values(['region_txt', 'eventid'], ascending=[True, False])
    st.write(attack_type_count_regions)


with tab5:
    st.title("Question 4")
    st.write("Plot the locations of attacks on a map to visualize their regional spread")        

    locations_df = df[['iyear', 'country_txt', 'region_txt', 'latitude', 'longitude', 'weaptype1_txt']]

    start_year, end_year = st.select_slider(
    'Select a range of years to view',
    options=df.iyear.unique().tolist(),
    value=(1970, 1970))
    geo_df = df.loc[(df['iyear'] >= start_year) & (df['iyear'] <= end_year)]

    fig = px.scatter_geo(geo_df,lat='latitude',lon='longitude', symbol='region_txt',
                     hover_name="country_txt",
                     hover_data=['weaptype1_txt', 'region_txt'])
    fig.update_layout(title = 'World map', title_x=0.5)

    st.write(fig)