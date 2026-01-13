set.seed(1234567890)

library(geosphere)

stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps    <- read.csv("temps50k.csv")

# Merge measurements with station coordinates
st <- merge(stations, temps, by = "station_number")

# Convert date/time columns
st$date <- as.Date(st$date)                     
st$time <- strptime(st$time, "%H:%M:%S")       

# Kernel bandwidths (hyperparameters chosen by me)
h_distance <- 100000    # metres
h_date     <- 20        # days
h_time     <- 4         # hours

# Forecast location (latitude a, longitude b)
a <- 58.4274            
b <- 14.826             
date <- "2013-11-04"   
forecast_date <- as.Date(date)

# Take times every 2 hours
times <- c("04:00:00","06:00:00","08:00:00","10:00:00",
           "12:00:00","14:00:00","16:00:00","18:00:00",
           "20:00:00","22:00:00")

# Vectors to store predictions for sum and product kernels
temp_add  <- numeric(length(times))
temp_mult <- numeric(length(times))

# Gaussian kernel in one dimension
gaussian_kernel <- function(x, h) {
  exp(-(x^2) / (2 * h^2))
}

# Spatial kernel: kernel value vs physical distance (metres)
x_dist <- seq(0, 300000, length.out = 1000)
plot(x_dist,
     gaussian_kernel(x_dist, h_distance),
     type = "l",
     xlab = "Physical distance (m)",
     ylab = "Kernel value",
     main = "Gaussian physical distance kernel")
grid()

# Date kernel: kernel value vs difference in days
x_date <- seq(0, 60, by = 0.5)
plot(x_date,
     gaussian_kernel(x_date, h_date),
     type = "l",
     xlab = "Distance in days",
     ylab = "Kernel value",
     main = "Gaussian date kernel")
grid()

# Time kernel: kernel value vs difference in hours
x_time <- seq(0, 18, by = 0.25)
plot(x_time,
     gaussian_kernel(x_time, h_time),
     type = "l",
     xlab = "Distance in hours",
     ylab = "Kernel value",
     main = "Gaussian time kernel")
grid()

# Day difference on calendar year
date_diff_ignoring_year <- function(date1_char, date2_char) {
  d1 <- strsplit(date1_char, "-")[[1]]
  d2 <- strsplit(date2_char, "-")[[1]]
  date1 <- as.Date(paste("2000", d1[2], d1[3], sep = "-"))
  date2 <- as.Date(paste("2000", d2[2], d2[3], sep = "-"))
  
  diff <- abs(as.numeric(date1 - date2))  # absolute difference in days
  
  # wrap around year if we crossed the year boundary
  if (diff >= 183) diff <- 366 - diff
  diff
}

# Hour difference on a 24h circle 
time_diff_ignoring_day <- function(time1, time2) {
  diff <- abs(as.numeric(difftime(time1, time2, units = "hours")))
  if (diff > 12) diff <- 24 - diff
  diff
}

# Kernel regression loop over forecast times
for (k in seq_along(times)) {
  time_char <- times[k]
  time      <- strptime(time_char, "%H:%M:%S")  # forecast time of day
  
  # Use only measurements not later than forecast date/time (no future data)
  st_temp <- st[st$date < forecast_date |
                  (st$date == forecast_date & st$time < time), ]
  
  
  # Spatial kernel for each measurement using distHaversine
  distance_kernels <- mapply(function(lat, lon) {
    d <- distHaversine(c(b, a), c(lon, lat))
    gaussian_kernel(d, h_distance)
  }, st_temp$latitude, st_temp$longitude)
  
  # Date kernel for each measurement
  date_kernels <- sapply(as.character(st_temp$date), function(d_i) {
    dd <- date_diff_ignoring_year(date, d_i)
    gaussian_kernel(dd, h_date)
  })
  
  # Time-of-day kernel for each measurement 
  time_kernels <- sapply(st_temp$time, function(t_i) {
    dt <- time_diff_ignoring_day(time, t_i)
    gaussian_kernel(dt, h_time)
  })
  
  # Combine the three kernels: sum and product
  kernels_add  <- distance_kernels + date_kernels + time_kernels
  kernels_mult <- distance_kernels * date_kernels * time_kernels
  
  # Kernel regression prediction: weighted average of air_temperature
  temp_add[k]  <- sum(kernels_add  * st_temp$air_temperature) /
    sum(kernels_add)
  temp_mult[k] <- sum(kernels_mult * st_temp$air_temperature) /
    sum(kernels_mult)
}

# Plots of the predicted temperature curves
# Forecast using sum of kernels
plot(temp_add,
     type = "o",
     xaxt = "n",
     xlab = "Time",
     ylab = "Temperature",
     main = "Added kernels")
axis(1, at = seq_along(times), labels = times)
grid()

# Forecast using product of kernels (oral defense)
plot(temp_mult,
     type = "o",
     xaxt = "n",
     xlab = "Time",
     ylab = "Temperature",
     main = "Multiplied kernels")
axis(1, at = seq_along(times), labels = times)
grid()

# Plot analysis 
# Gaussian physical distance kernel
# Nearby stations get high weight; weight drops as distance grows.
#Beyond roughly 200 km, stations contribute almost nothing.

#Gaussian date kernel
#Same-day measurements get weight 1; weight fades as days apart increase.
#After about a month, old data has very little influence.

#Gaussian time kernel
#Same hour gets full weight; a few hours away gets much less weight.
#So the method mainly uses measurements from similar times of day.

#Added kernels (sum)
#Temperature changes smoothly over the day: coolest early, warmest around midday, then cooling.
#Summing kernels averages over many observations, giving a very smooth curve.

#Multiplied kernels (product)
#The shape is similar but values differ slightly because fewer observations get high weight.
#Multiplying kernels focuses on data that are close in place, date, and time simultaneously.

# Q: Diff between adding and multiplying kernels?
# A: product kernel gives a high weight only when all three kernels are high at the same time (close in place, date, and hour).
# Sum -> many points get some weight.
# Product -> only a small number counts (since all prediction points must be close in every way)