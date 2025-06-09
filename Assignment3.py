### GMT312
### Student: Oğuzhan TOSUN, 2200674044

# Import necessary modules.
import numpy as np
from datetime import datetime
from astropy.time import Time, TimeDelta, TimeGPS

def lagrange(eph, dat, index):

    x = dat[:, index[0]] # x values
    y = dat[:, index[1]] # y values
    n = len(x)    # n number

    res = 0.0
    for i in range(n):
        total = y[i]
        for j in range(n):
            if j != i:
                total = total*(eph - x[j]) / (x[i] - x[j])
        res += total

    return res

def retrieve_epoch(sat_num, epoch, fpath):
    """
    Parameters
    ----------

    sat_num: str
            Satellite name or number.
    epoch: int
            Second of day.
    fpath: str
            File path of SP3 file.
    
    """
    
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
                        sat_inew = crs[index].split()[1:5] # X, Y, Z and Satellite Clock Correction
                        crs_new = [float(xyzs) for xyzs in sat_inew]
                        crs_new.insert(0, epoch)
                        ten_crs.append(crs_new)

                    return np.array([ten_crs])
                    
        start+=33
        finish+=33

        if finish>=len(lines):
            break 
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


def RotationMatrix(angle, axis=int):
    if axis==3:
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
        return R3
    elif axis==1:
        R1 = np.array([[1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])      
        return R1
        

def emis(trec, pc, clk):
    """
    Parameters
    ----------
    
    trec: float, int
        Second of day of the epoch.
    pc: float
        R value of the satellite from RNX file.
    clk: np.array
        10x2 matrix after Lagrange interpolation for epoch.

    Returns
    -------
    
    tems: float
        The signal emission time computed using precise ephemerides.
    
    """
    
    c = 299792458.0 # light speed (meter/sec)

    first_lag = lagrange(my_epoch, clk, [0,3])
    clk_error = first_lag[4]*1e-6

    tems = trec - ((pc/c)+clk_error)

    return tems

def sat_pos(trec, pc, sp3, r_apr):

    we = 7.2921151467e-5
    c = 299792458.0 # light speed (meter/sec)

    t_emis = emis(trec, pc, sp3)

    first_lag = lagrange(trec, sp3, [0,3])
    
    rsat_apprx = first_lag[1:4]*1000 # meter
    
    rrcv_0 = r_apr

    delta_t = np.sqrt((rsat_apprx[0]-rrcv_0[0])**2+(rsat_apprx[1]-rrcv_0[1])**2+(rsat_apprx[2]-rrcv_0[2])**2)/c
    
    print(t_emis-trec)

    R3 = np.array([[np.cos(we*delta_t), np.sin(we*delta_t), 0],
                   [-np.sin(we*delta_t), np.cos(we*delta_t), 0],
                   [0, 0, 1]])
    
    fpos = R3 @ rsat_apprx

    return fpos 

my_epoch = (2+2+6+7+4+4+4)*930
sp3_path = "IGS0OPSFIN_20240610000_01D_15M_ORB.SP3"

first = retrieve_epoch("PG02", my_epoch, fpath=sp3_path)
second = retrieve_epoch("PG06", my_epoch, fpath=sp3_path)


print(emis((2+2+6+7+4+4+4)*930, 26163864.643, first))
print(emis((2+2+6+7+4+4+4)*930, 25098315.102, second))

print(sat_pos((2+2+6+7+4+4+4)*930, 26163864.643, first, np.array([4239146.6414, 2886967.1245, 3778874.4800])))
print(sat_pos((2+2+6+7+4+4+4)*930, 25098315.102, second, np.array([4239146.6414, 2886967.1245, 3778874.4800])))
