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
Année = input('Entrez une année')
Format1 = input('Entrez une date mmdd')
Filepathshort=input('Copy paste the filepath to where you want to save the plots for the short timewindow')
Filepathshort=Filepathshort.replace('\\','/')
Filepathlong=input('Copy paste the filepath to where you want to save the plots for the long timewindow')
Filepathlong=Filepathlong.replace('\\','/')
Filepathday=input('Copy paste the filepath to where you want to save the plots for the day')
Filepathday=Filepathday.replace('\\','/')
# Import cdf from FIELD(with magnetic field)
cdf1 = pycdf.CDF(
    'C:/Users/sauge/PycharmProjects/pythonProject1/Donnees/mag_rtn/' + Année + '/psp_fld_l2_mag_rtn_' + Année + Format1 + '00_v01.cdf')
cdf2 = pycdf.CDF(
    'C:/Users/sauge/PycharmProjects/pythonProject1/Donnees/mag_rtn/' + Année + '/psp_fld_l2_mag_rtn_' + Année + Format1 + '06_v01.cdf')
cdf3 = pycdf.CDF(
    'C:/Users/sauge/PycharmProjects/pythonProject1/Donnees/mag_rtn/' + Année + '/psp_fld_l2_mag_rtn_' + Année + Format1 + '12_v01.cdf')
cdf4 = pycdf.CDF(
    'C:/Users/sauge/PycharmProjects/pythonProject1/Donnees/mag_rtn/' + Année + '/psp_fld_l2_mag_rtn_' + Année + Format1 + '18_v01.cdf')

# Make a copy of cdf file
with cdf1, cdf2, cdf3, cdf4:
    cdf1 = cdf1.copy()
    cdf2 = cdf2.copy()
    cdf3 = cdf3.copy()
    cdf4 = cdf4.copy()

# Import cdf from SWEAP (Velocity,density,Thermal speed...)
data1 = pycdf.CDF(
    'C:/Users/sauge/PycharmProjects/pythonProject1/Donnees/spdf.gsfc.nasa.gov_spc/pub/data/psp/sweap/spc/l3/l3i/' + Année + '/psp_swp_spc_l3i_' + Année + Format1 + '_v02.cdf')

# Make dataframes of each components of magnetic field for the whole day
df_magnetic_field_modified1 = pd.DataFrame(cdf1['psp_fld_l2_mag_RTN'], columns=['Br', "Bt", 'Bn'])
df_magnetic_field_modified1['|B|'] = np.sqrt(
    df_magnetic_field_modified1['Br'] ** 2 + df_magnetic_field_modified1['Bt'] ** 2 + df_magnetic_field_modified1[
        'Bn'] ** 2)
df_magnetic_field_modified2 = pd.DataFrame(cdf2['psp_fld_l2_mag_RTN'], columns=['Br', "Bt", 'Bn'])
df_magnetic_field_modified2['|B|'] = np.sqrt(
    df_magnetic_field_modified1['Br'] ** 2 + df_magnetic_field_modified1['Bt'] ** 2 + df_magnetic_field_modified1[
        'Bn'] ** 2)
df_magnetic_field_modified3 = pd.DataFrame(cdf3['psp_fld_l2_mag_RTN'], columns=['Br', "Bt", 'Bn'])
df_magnetic_field_modified3['|B|'] = np.sqrt(
    df_magnetic_field_modified1['Br'] ** 2 + df_magnetic_field_modified1['Bt'] ** 2 + df_magnetic_field_modified1[
        'Bn'] ** 2)
df_magnetic_field_modified4 = pd.DataFrame(cdf4['psp_fld_l2_mag_RTN'], columns=['Br', "Bt", 'Bn'])
df_magnetic_field_modified4['|B|'] = np.sqrt(
    df_magnetic_field_modified1['Br'] ** 2 + df_magnetic_field_modified1['Bt'] ** 2 + df_magnetic_field_modified1[
        'Bn'] ** 2)

# To see columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5000)
# concatenate the dataframe of time and magnetic field and merge dataframes together to get the dataframe for the day

df_time1 = pd.DataFrame(pd.to_datetime(cdf1['epoch_mag_RTN']), columns=['Time_datetime'])
df_time1['Time'] = df_time1['Time_datetime']
df_magnetic_field_modified1 = df_time1.join(df_magnetic_field_modified1)
df_time2 = pd.DataFrame(pd.to_datetime(cdf2['epoch_mag_RTN']), columns=['Time_datetime'])
df_time2['Time'] = df_time2['Time_datetime']
df_magnetic_field_modified2 = df_time2.join(df_magnetic_field_modified2)
df_time3 = pd.DataFrame(pd.to_datetime(cdf3['epoch_mag_RTN']), columns=['Time_datetime'])
df_time3['Time'] = df_time3['Time_datetime']
df_magnetic_field_modified3 = df_time3.join(df_magnetic_field_modified3)
df_time4 = pd.DataFrame(pd.to_datetime(cdf4['epoch_mag_RTN']), columns=['Time_datetime'])
df_time4['Time'] = df_time4['Time_datetime']
df_magnetic_field_modified4 = df_time4.join(df_magnetic_field_modified4)
final_df = pd.concat([df_magnetic_field_modified1, df_magnetic_field_modified2, df_magnetic_field_modified3,
                      df_magnetic_field_modified4], ignore_index=True)
final_df = final_df.set_index(final_df['Time'])

# Transform the time datas,velocity,density and thermal speed to dataframe with the correct units

time_velo = pd.to_datetime(data1['Epoch'])
velocity = data1['vp_moment_RTN']
density = data1['np_moment'][::]
thermal_speed = data1['wp_moment'][::]
df_time_velo = pd.DataFrame(time_velo, columns=['Time1'])
df_velocity = pd.DataFrame(velocity, columns=['VR', 'VT', "VN"])
df_density = pd.DataFrame(density, columns=['n1'])
df_density['n1'] = df_density['n1'] * 10 ** 6
df_thermal_speed = pd.DataFrame(thermal_speed, columns=['Thermal speed'])
df_thermal_speed['Thermal speed'] = df_thermal_speed['Thermal speed'] * 10 ** 3

# Drop the absurd values
df_velocity['VR'] = np.where(((df_velocity.VR > 10 ** 9) | (df_velocity.VR < 10 ** -9)), np.NaN, df_velocity['VR'])
df_velocity['VT'] = np.where(((df_velocity.VT > 10 ** 9) | (df_velocity.VT < 10 ** -9)), np.NaN, df_velocity['VT'])
df_velocity['VN'] = np.where(((df_velocity.VN > 10 ** 9) | (df_velocity.VN < 10 ** -9)), np.NaN, df_velocity['VN'])
df_density = df_density.drop(df_density[df_density.n1 < 0].index)
df_thermal_speed = df_thermal_speed.drop(df_thermal_speed[df_thermal_speed['Thermal speed'] < 10 ** -3].index)
df_thermal_speed = df_thermal_speed.drop(df_thermal_speed[df_thermal_speed['Thermal speed'] > 10 ** 5].index)

# Flags are used to prevent bad measurements
flags = pd.DataFrame(data1['DQF'][:, 16], columns=['Flag'])

# concatenate all these tables to a single one
final_df_velo = pd.concat([df_time_velo, df_velocity, df_density, df_thermal_speed, flags], axis=1)

# Use the flag in order to get rid of the bad measurements
final_df_velo['VR'] = np.where((final_df_velo.Flag == 1), np.NaN, final_df_velo['VR'])
final_df_velo['VT'] = np.where((final_df_velo.Flag == 1), np.NaN, final_df_velo['VT'])
final_df_velo['VN'] = np.where((final_df_velo.Flag == 1), np.NaN, final_df_velo['VN'])
final_df_velo['Thermal speed'] = np.where((final_df_velo.Flag == 1), np.NaN, final_df_velo['Thermal speed'])
final_df_velo['n1'] = np.where((final_df_velo.Flag == 1), np.NaN, final_df_velo['n1'])

# calculate Temperature and norm of velocity
final_df_velo['Temperature'] = (((final_df_velo['Thermal speed']) ** 2) * (1.673 * 10 ** -27)) / ((1.380 * 10 ** (-23)))
final_df_velo['|V|'] = np.sqrt(final_df_velo['VR'] ** 2 + final_df_velo['VT'] ** 2 + final_df_velo['VN'] ** 2)

# reset the index
final_df_velo['Time_Velocity'] = final_df_velo['Time1']
Desired_Table_Velocity = final_df_velo
Desired_Table_Velocity = Desired_Table_Velocity.reset_index(drop=True)
Desired_Table_Velocity = Desired_Table_Velocity.set_index(Desired_Table_Velocity['Time_Velocity'])

# Create a table with the magnetic field dataframe and the velocity,temperature...dataframe
tableau_modifié = pd.concat([final_df, Desired_Table_Velocity], axis=1, join='outer')
tableau_modifié = tableau_modifié.drop(['Time_datetime'], axis=1)

# Interpolation to the velocity index of time :
tableau_modifié = final_df.reindex(Desired_Table_Velocity.index, method='ffill')
tableau_modifié = pd.concat([tableau_modifié, Desired_Table_Velocity], axis=1)

# Create a column of all the datetime that we have to plot and set it back to index
tableau_modifié = tableau_modifié.rename_axis('Absolute_Time').reset_index()
tableau_modifié['Absolute_Time_index'] = tableau_modifié['Absolute_Time']
tableau_modifié = tableau_modifié.set_index('Absolute_Time_index')
Desired_Table = tableau_modifié

# PLASMA PARAMETERS
Desired_Table['Alfvén_speed'] = ((Desired_Table['|B|'] * 10 ** -9) * (10 ** -3)) / (
    np.sqrt(4 * np.pi * (10 ** -7) * Desired_Table['n1'] * (1.6726 * 10 ** -27)))
Desired_Table['VaR'] = ((Desired_Table['Br'] * 10 ** -9) * (10 ** -3)) / (
    np.sqrt(4 * np.pi * (10 ** -7) * Desired_Table['n1'] * (1.6726 * 10 ** -27)))
Desired_Table['VaT'] = ((Desired_Table['Bt'] * 10 ** -9) * (10 ** -3)) / (
    np.sqrt(4 * np.pi * (10 ** -7) * Desired_Table['n1'] * (1.6726 * 10 ** -27)))
Desired_Table['VaN'] = ((Desired_Table['Bn'] * 10 ** -9) * (10 ** -3)) / (
    np.sqrt(4 * np.pi * (10 ** -7) * Desired_Table['n1'] * (1.6726 * 10 ** -27)))
Desired_Table['Pressure'] = ((Desired_Table['|B|'] * 10 ** -9) ** 2) / (2 * 4 * np.pi * (10 ** -7))
Desired_Table['electron_gyrofrequency'] = ((1.6 * 10 ** -19) * Desired_Table['|B|'] * 10 ** -9) / (
            (9.1094 * 10 ** -31) * 3 * 10 ** 8)
Desired_Table['ion_gyrofrequency'] = ((1.6 * 10 ** -19) * Desired_Table['|B|'] * 10 ** -9) / ((1.6726 * 10 ** -27))
Desired_Table['ion_gyroradius'] = Desired_Table['Thermal speed'] / Desired_Table['ion_gyrofrequency']
Desired_Table['PlasmaBeta'] = (Desired_Table['n1'] * (1.38064852 * 10 ** (-23)) * Desired_Table['Temperature']) / (
    Desired_Table['Pressure'])
Desired_Table['ion_plasma_frequency'] = np.sqrt(
    (Desired_Table['n1'] * ((1.6 * 10 ** -19) ** 2)) / (((1.6726 * 10 ** -27)) * (8.85418782 * (10 ** -12))))
Desired_Table['Total_Pressure'] = (Desired_Table['n1'] * (1.38064852 * 10 ** (-23)) * Desired_Table['Temperature']) + \
Desired_Table['Pressure']
Desired_Table['ion_inertial_length'] = (3 * 10 ** 8) / Desired_Table['ion_plasma_frequency']

# HOWTOFIND Magnetic reconnexion events
# Group the datas by a certain amount of time and calculate the mean
Drop_Grouped = Desired_Table.groupby(pd.Grouper(freq='5min', origin='start_day')).mean()

# The frequency here is different to get a thin parameter however the condition on deltaBr Bt and Bn would be always satisfied ,but it has to be not too fast to get the maximum of events
Drop_Groupedmax = Desired_Table.groupby(pd.Grouper(freq='1min', origin='start_day')).max()
Drop_Groupedmin = Desired_Table.groupby(pd.Grouper(freq='1min', origin='start_day')).min()

# We reindex to make the conditions works and to plot later on
Drop_Grouped = Drop_Grouped.reindex(Desired_Table.index, method='ffill')
Drop_Groupedmax = Drop_Groupedmax.reindex(Desired_Table.index, method='ffill')
Drop_Groupedmin = Drop_Groupedmin.reindex(Desired_Table.index, method='ffill')

# calculate the fluctuations of each components of velocity (for the vectors)
Desired_Table['DeltaVR'] = Drop_Grouped['VR'] - Desired_Table['VR']
Desired_Table['DeltaVT'] = Drop_Grouped['VT'] - Desired_Table['VT']
Desired_Table['DeltaVN'] = Drop_Grouped['VN'] - Desired_Table['VN']
Desired_Table['Delta|V|'] = Desired_Table['|V|'] - Drop_Grouped['|V|']

# Calculate the fluctuations of each components of Alfvén speed for the vectors
Desired_Table['DeltaVaR'] = Drop_Grouped['DeltaVaR'] = Desired_Table['VaR'] - Drop_Grouped['VaR']
Desired_Table['DeltaVaT'] = Drop_Grouped['DeltaVaT'] = Desired_Table['VaT'] - Drop_Grouped['VaT']
Desired_Table['DeltaVaN'] = Drop_Grouped['DeltaVaN'] = Desired_Table['VaN'] - Drop_Grouped['VaN']
Desired_Table['Delta|Va|'] = Drop_Grouped['DeltaVa'] = Drop_Grouped['Alfvén_speed'] - Desired_Table['Alfvén_speed']

# For the condition of the  jumps of each component of magnetic field
Drop_Grouped['DeltaBr'] = np.abs(Drop_Groupedmax['Br'] - Drop_Groupedmin['Br'])
Drop_Grouped['DeltaBt'] = np.abs(Drop_Groupedmax['Bt'] - Drop_Groupedmin['Bt'])
Drop_Grouped['DeltaBn'] = np.abs(Drop_Groupedmax['Bn'] - Drop_Groupedmin['Bn'])
Drop_Grouped['DeltaVR'] = np.abs(Drop_Grouped['VR'] - Desired_Table['VR'])
##########SIGMAC

# First of all we have to create a table with only the velocity and then we calculate deltaV
Deltavelo = pd.concat([Desired_Table['VR'], Desired_Table['VT'], Desired_Table['VN']], axis=1)
DeltaVGrouped = pd.concat([Drop_Grouped['VR'], Drop_Grouped['VT'], Drop_Grouped['VN']], axis=1)
DeltaV = Deltavelo.sub(DeltaVGrouped)

# Same for deltab
DeltaVa = pd.concat([Desired_Table['VaR'], Desired_Table['VaT'], Desired_Table['VaN']], axis=1)
DeltaBGrouped = pd.concat([Drop_Grouped['VaR'], Drop_Grouped['VaT'], Drop_Grouped['VaN']], axis=1)
DeltaB = DeltaVa.sub(DeltaBGrouped)

# We rename columns to use them later
DeltaV = DeltaV.rename(columns={'0': 'VR', '1': 'VT', '2': 'VN'})
DeltaB = DeltaB.rename(columns={'0': 'VaR', '1': 'VaT', '2': 'VaN'})

# Calcul DeltaV . DeltaB
AlfvénxVelo = pd.DataFrame(columns=['R', 'T', 'N'])
AlfvénxVelo['R'] = DeltaB['VaR'] * DeltaV['VR']
AlfvénxVelo['T'] = DeltaB['VaT'] * DeltaV['VT']
AlfvénxVelo['N'] = DeltaB['VaN'] * DeltaV['VN']

# We calculate this |DeltaV|²+|DeltaB|²
DeltaVnormpower = (np.sqrt(DeltaV['VR'] ** 2 + DeltaV['VT'] ** 2 + DeltaV['VN'] ** 2)).pow(2)
DeltaBnormpower = (np.sqrt(DeltaB['VaR'] ** 2 + DeltaB['VaT'] ** 2 + DeltaB['VaN'] ** 2)).pow(2)
DeltaAlfvénplusVelo = DeltaBnormpower + DeltaVnormpower

# Then we use the formula for sigmac wich is 2*AlfvénxVelo/DeltaAlfvénplusVelo
Sigma = pd.DataFrame(columns=['R', 'T', 'N'])
Sigma['R'] = (2 * AlfvénxVelo['R']) / DeltaAlfvénplusVelo
Sigma['T'] = (2 * AlfvénxVelo['T']) / DeltaAlfvénplusVelo
Sigma['N'] = (2 * AlfvénxVelo['N']) / DeltaAlfvénplusVelo

# We calculate the norm as in the paperand we merge our principal dataframe with the dataframe of sigmac to plot it
sigmac = np.sqrt(Sigma['R'] ** 2 + Sigma['T'] ** 2 + Sigma['N'] ** 2)
sigmac = pd.DataFrame(sigmac, columns=['sigmac'])
Desired_Table = pd.concat([Desired_Table, sigmac], axis=1)

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
