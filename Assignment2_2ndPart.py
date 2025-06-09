# Import necessary modules.
import numpy as np
from datetime import datetime
from astropy.time import Time, TimeDelta, TimeGPS


def gpst2sod(epoch):
    """ This function converts from epoch to second of day."""
    
    # UTC to GPS Time.
    gps_time = epoch.to_value(format="gps")
    
    # Find the GPS week num.
    gps_week = gps_time//(86400*7)
    
    # Find the second of week.
    sec_of_week = gps_time -(gps_week*86400*7)
    
    # Day of GPS week.
    day_of_week = sec_of_week // 86400
    
    # Second of day.
    sec_of_day = sec_of_week - (day_of_week*86400)
    
    return sec_of_day

def retrieve_epoch(sat_num, epoch, fpath):
    
    file = open(fpath, 'r')
    lines = file.readlines()
    file.close()
    
    start = 22
    finish = 55
    
    ten_crs = []
    
    while True:
        coords = lines[start:finish]
        line_1 = coords[0][1:].split()
        t = [int(round(float(i), 0)) for i in line_1]
        message_epoch = Time(datetime(t[0], t[1], t[2], t[3], t[4], t[5]), scale="utc")
        sec_of_day = gpst2sod(message_epoch)

        difference = epoch - sec_of_day 

        if difference in range(0, 901):
            for i in coords:
                sat_info = i.split()
                if sat_num == sat_info[0]:
                    index = coords.index(i)
                    for j in [-4,-3,-2,-1,0,1,2,3,4,5]:
                        start_new = start+(j * 33)
                        finish_new = finish+(j * 33)
                        crs = lines[start_new:finish_new]
                        epoch_line = crs[0][1:].split()
                        t = [int(round(float(i), 0)) for i in epoch_line]
                        epoch = gpst2sod(Time(datetime(t[0], t[1], t[2], t[3], t[4], t[5]), scale="utc"))
                        sat_inew = crs[index].split()[1:4]
                        crs_new = [float(xyz) for xyz in sat_inew]
                        crs_new.insert(0, epoch)
                        ten_crs.append(crs_new)

                    return np.array([ten_crs])
                    
        start+=33
        finish+=33

        if finish>=len(lines):
            break    
        
sp3_path = "IGS0OPSFIN_20240610000_01D_15M_ORB.SP3"        
epochs = retrieve_epoch("PG08", 12*1860, sp3_path)


def lagrange(eph, dat):

    x = dat[:, 0] # x values
    y = dat[:, 1] # y values
    n = len(x)    # n number

    res = 0.0
    for i in range(n):
        total = y[i]
        for j in range(n):
            if j != i:
                total = total*(eph - x[j]) / (x[i] - x[j])
        res += total

    return res

print(lagrange(12*1860, epochs))
        
