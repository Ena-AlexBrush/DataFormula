#✨Worked with Tribhuvani Ayyagari, Naga Gunukula, and Nikitha Rambothula✨
import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#File CdA = Drag Coefficient × Frontal Area
df = pl.read_parquet("C:/Users/chanc/PycharmProjects/data_project/Data/08102025Endurance1_FirstHalf.parquet")
df = df.sort("VDM_UTC_TIME_SECONDS")

n = len(df)
dt_est = 0.01
df = df.with_columns(pl.Series("time_s", np.arange(n) * dt_est))

pdf = df.select(["VDM_GPS_SPEED", "time_s"]).drop_nulls().to_pandas()

speed = pdf["VDM_GPS_SPEED"].values
time = pdf["time_s"].values

if np.nanmax(speed) > 100:
    speed = speed / 3.6


accel = np.gradient(speed, time, edge_order=2)

mass = 250

Crr = 0.015
g = 9.81

F_total = -mass * accel

F_roll = mass * g * Crr

v_sq = speed**2
mask = (speed > 5) & (speed < np.nanmax(speed)*0.9)

X = v_sq[mask].reshape(-1, 1)
y = F_total[mask] - F_roll

model = LinearRegression()
model.fit(X, y)
k = model.coef_[0]
intercept = model.intercept_

rho = 1.225
CdA = 2 * k / rho

print(f"Slope: {k:.5f} N/(m/s)^2")
print(f"Intercept (N): {intercept:.2f}")
print(f"Estimated CdA: {CdA:.4f} m^2")
print(f"Rolling resistance Crr: {Crr:.5f}")


plt.scatter(v_sq[mask], y, s=2, label="Data")
plt.plot(v_sq[mask], model.predict(X), color='red', label="Linear fit")
plt.xlabel("Velocity² (m²/s²)")
plt.ylabel("Resistive Force (N)")
plt.title("Drag Force Fit — CdA Estimation")
plt.legend()
plt.grid(True)
plt.show()
