import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

df = pl.read_parquet("./Data/08102025Endurance1_FirstHalf.parquet")

df = df.filter(pl.col("VDM_GPS_VALID1") != 0)

df_analysis = df.select([
    pl.col("SME_TRQSPD_Speed").alias("Motor_RPM"),
    pl.col("SME_TEMP_BusCurrent").alias("Motor_Current"),
    pl.col("SME_TEMP_MotorTemperature").alias("Motor_Temp")
])

df_clean = df_analysis.filter(
    (pl.col("Motor_RPM") > 100) &
    (pl.col("Motor_Current") > 1.0) &
    (pl.col("Motor_Temp") > 0)
)

rpm = df_clean["Motor_RPM"].to_numpy()
current = df_clean["Motor_Current"].to_numpy()
temp = df_clean["Motor_Temp"].to_numpy()


#Create the graph
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 7))

#create a scatter plot where color represents the 3rd var. (temp)
scatter = plt.scatter(
    x=rpm,
    y=current,
    c=temp,
    cmap='inferno', #colormap for temperature
    s=10, # Marker size
    alpha=0.6, #transparency to see density
)

#Color Bar
cbar = plt.colorbar(scatter)
cbar.set_label('Motor Temperature (°C)', rotation=270, labelpad=15)

#Labels and Titles
plt.title('Current vs. RPM Colored by Temperature', fontsize=14)
plt.xlabel('Motor Speed (RPM)', fontsize=12)
plt.ylabel('DC Bus Current (Amps)', fontsize=12)


plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
