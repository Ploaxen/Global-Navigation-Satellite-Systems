"""
GMT312 Assignment-4
Author: Oğuzhan TOSUN - 2200674044
"""

# Import necessary modules.
import numpy as np
from datetime import datetime
from astropy.time import Time
from Ion_Klobuchar import Ion_Klobuchar # type: ignore
from trop_SPP import trop_SPP # type: ignore

def lagrange(eph, dat, index):
    
    x = dat[:, index[0]]  # x değerleri
    y = dat[:, index[1]]  # y değerleri

    if len(x) != len(y):
        raise ValueError("x and y value lengths must be equal.")
    
    n = len(x)  # n sayısı

    res = 0.0
    for i in range(n):
        total = y[i]
        for j in range(n):
            if j != i:
                total *= (eph - x[j]) / (x[i] - x[j])
        res += total

    return res

def gpst2sod(epoch):
    """ 
    This function converts from epoch to second of day. When convertion, uses GPS time.
    
    Parameters
    ----------
    
    epoch: astropy.time
        AstroPy time function.
        
    Returns
    -------
    
    sec_of_day: float
        Second of day in float.
    
    """
    
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
    """
    This function takes ten(10) epoch, X, Y, Z and clock error column from "*.sp3" file.
    
    If you have observation file, please use hatanaka to sp3 convertation software.
    
    Parameters
    ----------

    sat_num: str
            Satellite name or number.
    epoch: int
            Second of day.
    fpath: str
            File path of SP3 file.
            
    Returns
    -------
    
    ten_crs: np.array
        Previous 5 epoch, next 5 epoch from observation epoch with X, Y, Z and clock error.
    
    """
    
    
    # Take the epoch from first line.
    def parse_time_line(line):
        t = [int(round(float(i), 0)) for i in line[1:].split()]
        return Time(datetime(t[0], t[1], t[2], t[3], t[4], t[5]), scale="utc")

    # Takes other informations.
    def get_satellite_data(coords, sat_num, index):
        sat_info = coords[index].split()[1:5]
        return [float(xyzs) for xyzs in sat_info]

    # Read the "*.sp3" file
    with open(fpath, 'r') as file:
        lines = file.readlines()

    # Every epoch length. TR: yani her bir epochtaki toplam mesaj uzunluğu. 
    start, finish = 22, 55
    ten_crs = [] # will include epochs.

    while finish < len(lines):
        coords = lines[start:finish]
        if len(coords) < 1:
            break
        
        # Message epoch.
        message_epoch = parse_time_line(coords[0])
        sec_of_day = gpst2sod(message_epoch)
        # Difference btw. epoch and defined epoch.
        difference = epoch - sec_of_day

        # 15 minute check;
        if difference in range(0, 901):
            for i, line in enumerate(coords):
                sat_info = line.split()
                if sat_num == sat_info[0]:
                    index = i
                    # TR: bu for döngüsü, ilgili epochtaki uyduyu bulduktan sonra başlangıçta belirlemiş olduğum start, finish değerlerini kullanarak
                    # önceki ve sonraki 5 epochu içeren mesajları çekmemi sağlıyor. sonra uyduya ait verileri çekme işlemini her bir mesaj için tekrar uyguluyor.
                    # bu kısıma girdikten sonra uyduya ait bilgileri dosyanın tamamında değil, çarpım işlemi sonrasında elde ettiğimiz yeni mesjalarda arıyor. 
                    for j in range(-4, 6):
                        start_new = start + (j * 33)
                        finish_new = finish + (j * 33)
                        # bu kısım ise sadece kontrol işlemi görüyor.
                        if start_new < 0 or finish_new > len(lines):
                            continue
                        crs = lines[start_new:finish_new]
                        if len(crs) < 1:
                            continue

                        epoch_line = parse_time_line(crs[0])
                        epoch = gpst2sod(epoch_line)
                        crs_new = get_satellite_data(crs, sat_num, index)
                        crs_new.insert(0, epoch)
                        ten_crs.append(crs_new)

                    return np.array(ten_crs)

        start += 33
        finish += 33

    return np.array(ten_crs)

def xyz2plh(x, y, z, radian=True):
    """
    Converts cartesian coordinates (x, y, z) to ellipsoidal coordinates (phi, lambda, h) using
    an iterative procedure and the GRS80 reference ellipsoid.
    
    Parameters
    ----------
    x : float
        Cartesian x-coordinate in meters.
    y : float
        Cartesian y-coordinate in meters.
    z : float
        Cartesian z-coordinate in meters.
    radian: boolean
        If it is True, the function returns values in radian. Else converts to degree and returns.
    
    Returns
    -------
    phi : float
        Ellipsoidal latitude in degrees.
    lambda_ : float
        Ellipsoidal longitude in degrees.
    h : float
        Ellipsoidal height in meters.
        
    Resources
    ---------
    1) GMT225-3 slide
    2) https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    3) http://www.jpz.se/Html_filer/wgs_84.html
    
    """
    # Parameters of GRS80 Ellipsoid.
    a = 6378137.0  # meter
    b = 6356752.3141  # meter
    
    # Eccentricity (Attention! e²)
    e_2 = (pow(a, 2) - pow(b, 2)) / pow(a, 2)
    
    p = np.sqrt(pow(x, 2) + pow(y, 2)) # sqrt(x² + y²)
    
    # Initial values for iterative procedure.
    phi = np.arctan2(z, (1 - e_2) * p)
    lambda_ = np.arctan2(y, x)
    h = 0 # height

    while True:
        # Radius of curvature in the prime vertical
        N = a / np.sqrt(1 - e_2 * pow(np.sin(phi), 2))
        h_new = (p / np.cos(phi)) - N
        phi_new = np.arctan2(z + e_2 * N * np.sin(phi), p)
        
        # Check the precision
        if abs(phi_new - phi) < 1e-12:
            phi = phi_new
            h = h_new
            # If precision is lower than 10^(−12) stop the while loop.
            break
        
        # Change the variables values with new values.
        phi = phi_new
        h = h_new
    
    if radian:
        return phi, lambda_, h
    else:
        # Convert radian to degree and return.
        return np.degrees(phi), np.degrees(lambda_), h
    
def local(x0, y0, z0, x1, y1, z1):
    """
    Converts cartesian coordinates (x, y, z) to ellipsoidal coordinates (phi, lambda, h) using
    an iterative procedure and the GRS80 reference ellipsoid.
    
    Parameters
    ----------
    x0 : float
        Cartesian x-coordinate in meters of center point.
    y0 : float
        Cartesian y-coordinate in meters of center point.
    z0 : float
        Cartesian z-coordinate in meters of center point.
    x1 : float
        Cartesian x-coordinate in meters of center point.
    y1 : float
        Cartesian y-coordinate in meters of center point.
    z1 : float
        Cartesian z-coordinate in meters of center point.
    
    Returns
    -------
    az : float
        Azimuth angle from topocenter to target point in degrees.
    zen : float
        Zenith angle from topocenter to target point in degrees.
    slantd : float
        Distance between two point in meters.
        
    """  
    
    # Difference between two coordinate.
    delta_xyz = np.array([x1 - x0,
                          y1 - y0,
                          z1 - z0])

    # Slant (radial) range from the topocenter to the target
    slantd = np.sqrt(delta_xyz[0] ** 2 + delta_xyz[1] ** 2 + delta_xyz[2] ** 2)
    
    # Latitude, longitude and height of topocenter.
    lat, lon, h = xyz2plh(x0, y0, z0, radian=True)
    
    # Rotation matrix.
    rotation_matrix = np.array([[-np.sin(lat), -np.cos(lat) * np.sin(lon), np.cos(lat) * np.cos(lon)],
                                [np.cos(lat), -np.sin(lat) * np.sin(lon), np.sin(lat) * np.cos(lon)],
                                [0, np.cos(lon), np.sin(lat)]])
    
    # Convert to ENU coordinates.
    ENU = rotation_matrix @ delta_xyz
    
    # ENU coordinates for azimuth and zenith angle.
    east, north, up = ENU[0], ENU[1], ENU[2]
    
    # Zenith angle.
    zen = 90 - np.rad2deg(np.arcsin(up / np.sqrt(east ** 2 + north ** 2 + up ** 2)))
    
    # Azimuth angle.
    az = np.rad2deg(np.arctan2(north, east))
    
    # Return azimuth and zenith angle with slant distance.
    return np.array([az, zen, slantd])

def emis(trec, pc, clk):
    
    c = 299792458.0  # light speed (meter/sec)
    
    clk_error = lagrange(trec, clk, [0, 3]) * 1e-6  # clock error in seconds
    
    tems = trec - ((pc / c) + clk_error)
    
    return tems

def sat_pos(trec, pc, sp3, r_apr):
    """
    This function calculates final satellite position.
    
    Parameters
    ----------
    
    trec: float
        Second of day for defined epoch.
        
    pc: float
        Code pseudorange observation from the observation file in meter.
    
    sp3: np.array (10x5 matrix)
        A matrix (or corresponding array) composed of the time tags (t); corresponding satellite coordinates (X, Y, Z); and the satellite clock corrections for ten consecutive epochs (five epochs after and before the reception time) which are obtained from the related precise ephemeris.
        
    r_apr: np.array (3x1 matrix)
        Approximate receiver coordinates in ECEF coordinates (meter). 
        
    Returns
    -------
    
    fpos: np.array (3x1 matrix) 
        Final satellite coordinates in ECEF coordinates (meter). 
    """
    
    we = 7.2921151467e-5 # Earth rotation rate (rad/sec)
    c = 299792458.0  # light speed (meter/sec)

    # emission time
    t_emis = emis(trec, pc, sp3)


    # Calculation positions with Lagrangre enterpolation.
    x_pos = lagrange(t_emis, sp3, [0, 1])
    y_pos = lagrange(t_emis, sp3, [0, 2])
    z_pos = lagrange(t_emis, sp3, [0, 3])

    rsat_apprx = np.array([x_pos, y_pos, z_pos]) * 1000  # convertion from km to meter

    rrcv_0 = r_apr

    # Time difference.
    delta_t = np.linalg.norm(rsat_apprx - rrcv_0) / c

    alpha = we * delta_t

    # Rotation matrix around 3rd axis (z)
    R3 = np.array([[np.cos(alpha), np.sin(alpha), 0],
                   [-np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])

    # Final satellite position.
    fpos = np.dot(R3, rsat_apprx)
    
    # return final position.
    return fpos

def atmos(doy, trec, trecw, C1, rec, sp3, alpha, beta):
    
    # ECEF to phi, lat and height.
    ell = xyz2plh(rec[0], rec[1], rec[2])
    
    lat, lon, h = ell # latitude, longitude and height in radian.
    
    # calculate final satellite position in defined epoch.
    sat = sat_pos(trec, C1, sp3, rec)
    
    # calculate azimuth, zenith and slant distance.
    az, zen, slantd = local(rec[0][0], rec[1][0], rec[2][0], sat[0], sat[1], sat[2])
    
    # Azimuth and Elevation angle convertion degree to radian.
    az_d = np.deg2rad(az)
    elv_d = np.deg2rad(90-zen)
    
    dion = Ion_Klobuchar(lat, lon, elv_d, az_d, alpha, beta, trecw)
    Trzd, Trzw, ME = trop_SPP(lat, doy, h, elv_d)
    
    TrD = Trzd * ME
    TrW = Trzw * ME
    
    return az, zen, slantd, dion, TrD, TrW

# Closest position from RINEX file for my epoch of the satellite receiver.
r_apr = np.array([[-11530.195316], 
                  [18662.983875], 
                  [14391.063983]])

doy = 61 # day of the year
trec = (2+2+6+7+4+4+4)*930 # the sum of my student ID num
sow = 458970 # second of week 
C1 = 26156940.038 # meter
rec = r_apr

sp3_path = "data/IGS0OPSFIN_20240610000_01D_15M_ORB.SP3"
sp3_1 = retrieve_epoch("PG02", trec, fpath=sp3_path)
alpha = np.array([[0.2328e-07],[0.0000e+00],[-0.1192e-06],[0.1192e-06]]) #ION ALPHA parameters
beta = np.array([[0.1290e+06],[-0.1638e+05],[-0.6554e+05],[-0.5243e+06]]) #ION BETA parameters

az, zen, slantd, IonD, TrD, TrW = atmos(doy, trec, sow, C1, rec, sp3_1, alpha, beta)

print(f"Azimuth Angle: {az}\nZenith Angle: {zen}\nSlant (radial) distance(kilometer): {slantd}\nIonospheric delay for the related signal(meter): {IonD}\nTropospheric dry (hydrostatic) delay for the related signal (meter): {TrD}\nTropospheric wet delay for the related signal: {TrW}")