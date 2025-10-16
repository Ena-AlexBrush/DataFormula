import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

df = pl.read_parquet("./Data/08102025Endurance1_FirstHalf.parquet")

df = df.filter(pl.col("VDM_GPS_VALID1") != 0) #filter for valid GPS data

#extract & convert colms to numpy arrays
#need to use .to_numpy() for Matplotlib compatibility
time_index = df.select(pl.Series(range(len(df)))).to_numpy().flatten()
gps_speed = df["VDM_GPS_SPEED"].to_numpy()
brake_voltage = df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"].to_numpy()
torque_demand = df["SME_THROTL_TorqueDemand"].to_numpy()
motor_temp = df["SME_TEMP_MotorTemperature"].to_numpy()

#matplotlib Plotting
fig = plt.figure(figsize=(11.5, 10))
plt.suptitle("Driver Input and Car Response", fontsize=16)

#AX1 (Top left): GPS speed over time
ax1 = fig.add_subplot(221)
#plotting only gps speed
ax1.plot(time_index, gps_speed, label="GPS Speed", color='blue')
ax1.set_title("GPS Speed over Time")
ax1.set_ylabel("Speed (m/s)")
ax1.set_xlabel("Time (Data Index)")
ax1.grid(True)

#AX2 (Top right): Speed v brake pedal vol (scatter)
ax2 = fig.add_subplot(222)

#filter for when the car is moving & braking is active
moving_brakes = (gps_speed > 1.0) & (brake_voltage > 0.1)
ax2.scatter(brake_voltage[moving_brakes], gps_speed[moving_brakes], s=5, color='darkred')
ax2.set_title("Vehicle Speed vs. Brake Pedal Voltage")
ax2.set_xlabel("Brake Sense Voltage (V)")
ax2.set_ylabel("Speed (m/s)")
ax2.grid(True)

#AX3 (Bottom left): torque demand > time
ax3 = fig.add_subplot(223)
ax3.plot(time_index, torque_demand, label="Torque Demand", color='orange')
ax3.set_title("Driver's Throttle/Torque Demand")
ax3.set_xlabel("Time")
ax3.set_ylabel("Torque Demand (%)")
ax3.grid(True)

#AX4 (bottom right): motor temp > time
ax4 = fig.add_subplot(224)
ax4.plot(time_index, motor_temp, label="Motor Temperature", color='red')
ax4.set_title("Motor Temperature Response")
ax4.set_xlabel("Time (Data Index)")
ax4.set_ylabel("Temperature (°C)")
ax4.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) #adjust layout to make room for suptitle
plt.show()
