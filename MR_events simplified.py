# Import libraries
import os
import sys
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import pandas as pd

os.environ['Cdf'] = 'Users\sauge\PycharmProjects\pythonProject1\venv\Lib\site-packages'
from spacepy import pycdf

# Input year and date format to search
Format1 = input('Entrez une date mmdd')
Filepath=input('Copy paste the path to the directory of Dataframes')
Filepath=Filepath.replace('\\','/')
Filepathshort=input('Copy paste the filepath to where you want to save the plots for the short timewindow')
Filepathshort=Filepathshort.replace('\\','/')
Filepathlong=input('Copy paste the filepath to where you want to save the plots for the long timewindow')
Filepathlong=Filepathlong.replace('\\','/')
Filepathday=input('Copy paste the filepath to where you want to save the plots for the day')
Filepathday=Filepathday.replace('\\','/')

Desired_Table=pd.read_pickle(str(Filepath)+'/Desired_Table'+Format1)
# Group the datas by a certain amount of time and calculate the mean
Drop_Grouped = Desired_Table.groupby(pd.Grouper(freq='5min', origin='start_day')).mean()

# The frequency here is different to get a thin parameter however the condition on deltaBr Bt and Bn would be always satisfied ,but it has to be not too fast to get the maximum of events
Drop_Groupedmax = Desired_Table.groupby(pd.Grouper(freq='1min', origin='start_day')).max()
Drop_Groupedmin = Desired_Table.groupby(pd.Grouper(freq='1min', origin='start_day')).min()

# We reindex to make the conditions works and to plot later on
Drop_Grouped = Drop_Grouped.reindex(Desired_Table.index, method='ffill')
Drop_Groupedmax = Drop_Groupedmax.reindex(Desired_Table.index, method='ffill')
Drop_Groupedmin = Drop_Groupedmin.reindex(Desired_Table.index, method='ffill')

# For the condition of the  jumps of each component of magnetic field
Drop_Grouped['DeltaBr'] = np.abs(Drop_Groupedmax['Br'] - Drop_Groupedmin['Br'])
Drop_Grouped['DeltaBt'] = np.abs(Drop_Groupedmax['Bt'] - Drop_Groupedmin['Bt'])
Drop_Grouped['DeltaBn'] = np.abs(Drop_Groupedmax['Bn'] - Drop_Groupedmin['Bn'])
Drop_Grouped['DeltaVR'] = np.abs(Drop_Grouped['VR'] - Desired_Table['VR'])
# Define parameters to select the events(here the fluctuations of DeltaVR have to be >1/2*Alfvén speed and the fluctuations for at least one component of Magnetic field has to be >20nT
Drop_Grouped = Drop_Grouped.loc[(Drop_Grouped['DeltaVR'] > 1 / 2 * Drop_Grouped['Alfvén_speed']) & (
            (Drop_Grouped['DeltaBr'] > 20) | (Drop_Grouped['DeltaBt'] > 20) | (Drop_Grouped['DeltaBn'] > 20))]

# Print the number of events without grouping them
print('nombre d évenements', len(Drop_Grouped))

# Reset index
Drop_Grouped = Drop_Grouped.reset_index()
Drop_Grouped['Time'] = Drop_Grouped['Absolute_Time_index']
Drop_Grouped = Drop_Grouped.set_index(Drop_Grouped['Absolute_Time_index'])

# Group the same events together and keep only the first occurence of them
Drop_Grouped_60s = Drop_Grouped.groupby(pd.Grouper(freq='60S')).first().dropna()

# Print the number of events when they are grouped
print('nombre d évenements', len(Drop_Grouped_60s))

# This column of zero will be used  for the vectors
Desired_Table['Zero'] = np.zeros((len(Desired_Table['Br']), 1))
# loop where python looks at each events in the index and will plot with a certain time window
for values in Drop_Grouped_60s.index:
    print(values)
    values = pd.to_datetime(values)

    # Timewindow for the short timewindow graph
    Timewindowstart = values - timedelta(seconds=15)
    Timewindowend = values + timedelta(seconds=15)
    # Timewindow for the long time window graph
    Timewindowstartlong = values - timedelta(seconds=150)
    Timewindowendlong = values + timedelta(seconds=150)
    # Take only the datas in the main Dataframe according to the timewindow (here the short one) to plot variables
    MR_Time = Desired_Table.loc[Timewindowstart: Timewindowend]

    # SHORT GRAPH
    fig, axes = plt.subplots(15, 1, sharex=True, figsize=(11, 11))
    for nn, ax in enumerate(axes):
        # Number of ticks set to be precise as possible
        yticks = matplotlib.ticker.MaxNLocator(6, min_n_ticks=5)
        ax.yaxis.set_major_locator(yticks)
        ax.tick_params(axis='y', labelsize=8)
    # Plot variables
    axes[0].plot_date(MR_Time['Absolute_Time'], MR_Time['|B|'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='purple')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Br'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='red', label='Br')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Bt'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='Bt')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Bn'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='green', label='Bn')
    axes[2].plot_date(MR_Time['Absolute_Time'], MR_Time['VR'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='black')
    axes[3].plot_date(MR_Time['Absolute_Time'], MR_Time['VT'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='VT')
    axes[3].plot_date(MR_Time['Absolute_Time'], MR_Time['VN'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='orange', label='VN')
    axes[4].plot_date(MR_Time['Absolute_Time'], MR_Time['n1'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='brown')
    axes[5].plot_date(MR_Time['Absolute_Time'], MR_Time['Temperature'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='red')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['Alfvén_speed'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='black', label='|Va|')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaR'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='red', label='VaR')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaT'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='VaT')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaN'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='orange', label='VaN')
    axes[7].plot_date(MR_Time['Absolute_Time'], MR_Time['PlasmaBeta'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='cyan')
    axes[8].plot_date(MR_Time['Absolute_Time'], MR_Time['Total_Pressure'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='orange')
    axes[9].plot_date(MR_Time['Absolute_Time'], MR_Time['Vtot'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='yellow')
    axes[10].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVaR'], MR_Time['DeltaVaT'], angles='uv',
                    width=0.001)
    axes[11].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVaR'], MR_Time['DeltaVaN'], angles='uv',
                    width=0.001)
    axes[12].plot_date(MR_Time['Absolute_Time'], MR_Time['sigmac'], linestyle='solid', linewidth=1, markersize=0.000001,
                       color='red')
    axes[13].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVR'], MR_Time['DeltaVT'], angles='uv',
                    width=0.001)
    axes[14].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVR'], MR_Time['DeltaVN'], angles='uv',
                    width=0.001)
    # Legend the plots for when there are several datas plotted at the same time in the same graph
    axes[1].legend(loc='upper right', prop={"size": 7})
    axes[3].legend(loc='upper right', prop={"size": 7})
    axes[6].legend(loc='upper right', prop={"size": 6})
    fig.suptitle(str(Format1), fontsize=9)

    # Labels to see what is ploted
    axes[0].set_ylabel('|B|\n(nT)', fontsize=7)
    axes[1].set_ylabel('Brtn\n(nT)', fontsize=7)
    axes[2].set_ylabel('VR\n(km/s)', fontsize=7)
    axes[3].set_ylabel('VT and VN\n (km/s', fontsize=7)
    axes[4].set_ylabel('Density \n(m^3)', fontsize=7)
    axes[5].set_ylabel('Temperature\n(K)', fontsize=7)
    axes[5].tick_params(axis='y', which='major', labelsize=7)
    axes[6].set_ylabel('Alfvén\nspeed(Km/s)', fontsize=7)
    axes[7].set_ylabel('PlasmaBeta\n(SI)', fontsize=7)
    axes[8].set_ylabel('Total \nPressure(Pa)', fontsize=7)
    axes[9].set_ylabel('Vdiff\n(Km/s)', fontsize=7)
    axes[10].set_ylabel('δb R-T', fontsize=7)
    axes[11].set_ylabel('δb R-N', fontsize=7)
    axes[12].set_ylabel('sigmac\n (unitless)', fontsize=7)
    axes[13].set_ylabel('δV R-T', fontsize=7)
    axes[14].set_ylabel('δV R-N', fontsize=7)

    # To get the information on the datetime rotation is needed unless datetime covers each others
    plt.xticks(rotation=90, size=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # Save the files by replacing forbidden character
    values = str(values)
    values = values.replace(':', '-')
    values = values.replace('.', '_')

    # Choose where do you want to save it
    plt.savefig(Filepathshort+'/' + str(values) + '.pdf', format='pdf')
    plt.show()

    # LONG GRAPH
    # Replace the previous MR_Time with a .loc on a larger time window
    MR_Time = Desired_Table.loc[Timewindowstartlong: Timewindowendlong]
    fig, axes = plt.subplots(15, 1, sharex=True, figsize=(11, 11))
    for nn, ax in enumerate(axes):
        # Number of ticks set to be precise as possible
        yticks = matplotlib.ticker.MaxNLocator(6, min_n_ticks=5)
        ax.yaxis.set_major_locator(yticks)
        ax.tick_params(axis='y', labelsize=8)
    axes[0].plot_date(MR_Time['Absolute_Time'], MR_Time['|B|'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='purple')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Br'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='red', label='Br')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Bt'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='Bt')
    axes[1].plot_date(MR_Time['Absolute_Time'], MR_Time['Bn'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='green', label='Bn')
    axes[2].plot_date(MR_Time['Absolute_Time'], MR_Time['VR'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='black')
    axes[3].plot_date(MR_Time['Absolute_Time'], MR_Time['VT'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='VT')
    axes[3].plot_date(MR_Time['Absolute_Time'], MR_Time['VN'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='orange', label='VN')
    axes[4].plot_date(MR_Time['Absolute_Time'], MR_Time['n1'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='brown')
    axes[5].plot_date(MR_Time['Absolute_Time'], MR_Time['Temperature'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='red')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['Alfvén_speed'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='black', label='|Va|')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaR'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='red', label='VaR')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaT'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='blue', label='VaT')
    axes[6].plot_date(MR_Time['Absolute_Time'], MR_Time['VaN'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='orange', label='VaN')
    axes[7].plot_date(MR_Time['Absolute_Time'], MR_Time['PlasmaBeta'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='cyan')
    axes[8].plot_date(MR_Time['Absolute_Time'], MR_Time['Total_Pressure'], linestyle='solid', linewidth=1,
                      markersize=0.000001, color='orange')
    axes[9].plot_date(MR_Time['Absolute_Time'], MR_Time['Vtot'], linestyle='solid', linewidth=1, markersize=0.000001,
                      color='yellow')
    axes[10].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVaR'], MR_Time['DeltaVaT'], angles='uv',
                    width=0.001)
    axes[11].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVaR'], MR_Time['DeltaVaN'], angles='uv',
                    width=0.001)
    axes[12].plot_date(MR_Time['Absolute_Time'], MR_Time['sigmac'], linestyle='solid', linewidth=1, markersize=0.000001,
                       color='red')
    axes[13].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVR'], MR_Time['DeltaVT'], angles='uv',
                    width=0.001)
    axes[14].quiver(MR_Time['Absolute_Time'], MR_Time['Zero'], MR_Time['DeltaVR'], MR_Time['DeltaVN'], angles='uv',
                    width=0.001)
    axes[1].legend(loc='upper right', prop={"size": 7})
    axes[3].legend(loc='upper right', prop={"size": 7})
    axes[6].legend(loc='upper right', prop={"size": 6})
    fig.suptitle(str(Format1), fontsize=9)

    # Labels to see what is ploted
    axes[0].set_ylabel('|B|\n(nT)', fontsize=7)
    axes[1].set_ylabel('Brtn\n(nT)', fontsize=7)
    axes[2].set_ylabel('VR\n(km/s)', fontsize=7)
    axes[3].set_ylabel('VT and VN\n (km/s', fontsize=7)
    axes[4].set_ylabel('Density \n(m^3)', fontsize=7)
    axes[5].set_ylabel('Temperature\n(K)', fontsize=7)
    axes[5].tick_params(axis='y', which='major', labelsize=7)
    axes[6].set_ylabel('Alfvén\nspeed(Km/s)', fontsize=7)
    axes[7].set_ylabel('PlasmaBeta\n(SI)', fontsize=7)
    axes[8].set_ylabel('Total \nPressure(Pa)', fontsize=7)
    axes[9].set_ylabel('Vdiff\n(Km/s)', fontsize=7)
    axes[10].set_ylabel('δb R-T', fontsize=7)
    axes[11].set_ylabel('δb R-N', fontsize=7)
    axes[12].set_ylabel('sigmac\n (unitless)', fontsize=7)
    axes[13].set_ylabel('δV R-T', fontsize=7)
    axes[14].set_ylabel('δV R-N', fontsize=7)
    plt.xticks(rotation=90, size=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.savefig(Filepathlong+'/Graphelong' + str(values) + '.pdf',format='pdf')
    plt.show()

# Making the subplots FOR DAYGRAPH
figuree, graph = plt.subplots(15, 1, sharex=True, figsize=(11, 11))
for nn, ax in enumerate(graph):
    # Number of ticks set to be precise as possible
    locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
    yticks = matplotlib.ticker.MaxNLocator(6, min_n_ticks=5)
    ax.yaxis.set_major_locator(yticks)
    ax.xaxis.set_major_locator(locator)
    ax.tick_params(axis='y', labelsize=8)

graph[0].plot_date(Desired_Table['Absolute_Time'], Desired_Table['|B|'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='purple')
graph[1].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Br'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='red', label='Br')
graph[1].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Bt'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, label='Bt')
graph[1].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Bn'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, label='Bn')
graph[2].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VR'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='black')
graph[3].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VT'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='blue', label='VT')
graph[3].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VN'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='orange', label='VN')
graph[4].plot_date(Desired_Table['Absolute_Time'], Desired_Table['n1'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='brown')
graph[5].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Temperature'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='red')
graph[6].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Alfvén_speed'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='black', label='|Va|')
graph[6].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VaR'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='red', label='VaR')
graph[6].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VaT'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='blue', label='VaT')
graph[6].plot_date(Desired_Table['Absolute_Time'], Desired_Table['VaN'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='orange', label='VaN')
graph[7].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Vtot'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='green')
graph[8].plot_date(Desired_Table['Absolute_Time'], Desired_Table['PlasmaBeta'], linestyle='solid', linewidth=0.2,
                   markersize=0.0001, color='cyan')
graph[9].plot_date(Desired_Table['Absolute_Time'], Desired_Table['Total_Pressure'], linestyle='solid', linewidth=0.2,
                   markersize=0.000001, color='orange')
graph[10].quiver(Desired_Table['Absolute_Time'], Desired_Table['Zero'], Desired_Table['DeltaVaR'],
                 Desired_Table['DeltaVaT'], angles='uv', width=0.001)
graph[11].quiver(Desired_Table['Absolute_Time'], Desired_Table['Zero'], Desired_Table['DeltaVaR'],
                 Desired_Table['DeltaVaN'], angles='uv', width=0.001)
graph[12].plot_date(Desired_Table['Absolute_Time'], Desired_Table['sigmac'], linestyle='solid', linewidth=0.01,
                    markersize=0.000001, color='red')
graph[13].quiver(Desired_Table['Absolute_Time'], Desired_Table['Zero'], Desired_Table['DeltaVR'],
                 Desired_Table['DeltaVT'], angles='uv')
graph[14].quiver(Desired_Table['Absolute_Time'], Desired_Table['Zero'], Desired_Table['DeltaVR'],
                 Desired_Table['DeltaVN'], angles='uv')

graph[1].legend(loc='upper right', prop={"size": 7})
graph[3].legend(loc='upper right', prop={"size": 7})
graph[6].legend(loc='upper right', prop={"size": 6})
figuree.suptitle(str(Format1), fontsize=9)

# Labels to see what is ploted
graph[0].set_ylabel('|B|\n(nT)', fontsize=7)
graph[1].set_ylabel('Brtn\n(nT)', fontsize=7)
graph[2].set_ylabel('VR\n(km/s)', fontsize=7)
graph[3].set_ylabel('VT VN\n(km/s)', fontsize=7)
graph[4].set_ylabel('Density \n(m^3)', fontsize=7)
graph[5].set_ylabel('Temperature\n(K)', fontsize=7)
graph[5].tick_params(axis='y', which='major', labelsize=8)
graph[6].set_ylabel('Alfvén\nspeed(Km/s)', fontsize=7)
graph[7].set_ylabel('Vdiff\n(Km/s)', fontsize=7)
graph[8].set_ylabel('PlasmaBeta\n(SI)', fontsize=7)
graph[9].set_ylabel('Total \nPressure(Pa)', fontsize=7)
graph[10].set_ylabel('δb R-T', fontsize=7)
graph[11].set_ylabel('δb R-N', fontsize=7)
graph[12].set_ylabel('sigmac\n (unitless)', fontsize=7)
graph[13].set_ylabel('δV R-T', fontsize=7)
graph[14].set_ylabel('δV R-N', fontsize=7)
# Date format unless we only have the year of the values on the plot
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# Rotation is important without, the datas are covered
plt.xticks(rotation=90, size=8)
plt.savefig(Filepathday+'/' + str(Format1) + '.pdf', format='pdf')
plt.show()
plt.close()
