# Import necessary modules.
import numpy as np
from datetime import datetime
from astropy.time import Time, TimeDelta, TimeGPS

class RinexMessage():
    def __init__(self, num, epoch, crs, delta_n, Mo, cuc, e, cus, sqrt_a, t0e, cic, omega0, cis, i0, crc, omega, omega_dot, idot, all_values, index):
        
        self.num = num # Num of satellite. str
        self.epoch = epoch # Message epoch.
        self.crs = float(crs) # crs
        self.delta_n = float(delta_n)
        self.Mo = float(Mo) # Mo angle
        self.cuc = float(cuc) # Argument of Latitude Correction Cosinus Component / radians
        self.e = float(e) # eccentricity
        self.cus = float(cus) # Argument of Latitude Correction Sinus Component / radians
        self.sqrt_a = float(sqrt_a) # Square root of the orbit semi major axis / m^0.5
        self.t0e = float(t0e) # TOE: Time Of Ephemeris / Seconds of GPS week
        self.cic = float(cic) # Inclination Correction Cosinus Component / radians
        self.omega0 = float(omega0) # OMEGA (Right Ascension of Ascending Node) Angle / radians
        self.cis = float(cis) # Inclination Correction Sinus Component / radians
        self.i0 = float(i0) # inital inclination / radians
        self.crc = float(crc) # Radius Correction Cosinus Component / meters
        self.omega = float(omega) # Radius Correction Cosinus Component / radians
        self.omega_dot = float(omega_dot) # Radius Correction Cosinus Component / (radians/second)
        self.idot = float(idot) # inclination rate / (radians/second)
        self.all_values = all_values # all values of message in list.
        self.index = index # index values.
        
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
    
    
def message_finder(sat_num, epoch, brdc_path=str):
    
    start = 8
    finish = 16
    file = open(brdc_path, 'r')
    lines = file.readlines()
    file.close()
    
    while True:
        # Rinex message.
        brdc_message = lines[start:finish]
        # first line for sat num, epoch etc.
        line_1 = brdc_message[0]
        t2 = list(map(float, line_1[:22].split())) # str to float
        t = list(map(int, t2)) # float to int because of AstroPy library function.

    
        if sat_num == t[0]:
            # epoch. 
            sat_epoch = Time(datetime(t[1]+2000,t[2],t[3],t[4],t[5],t[6]), scale="utc")     # define the epoch
            # Second Of Day.
            obs_sod = gpst2sod(epoch)
            sat_sod = gpst2sod(sat_epoch)

            # Difference between observation epoch and satellite epoch.
            delta_time = obs_sod - sat_sod

            # 15 minute difference.
            if delta_time>0 and delta_time<900:     
                # Lines for other parameters.
                line_2 = brdc_message[1]
                line_3 = brdc_message[2]
                line_4 = brdc_message[3]
                line_5 = brdc_message[4]
                line_6 = brdc_message[5]
                
                # 16 parameters from broadcast message.
                val_list = [line_2[22:41], line_2[41:60], line_2[60:79], 
                            line_3[0:22], line_3[22:41], line_3[41:60], line_3[60:79],
                            line_4[0:22], line_4[22:41], line_4[41:60], line_4[60:79], 
                            line_5[0:22], line_5[22:41], line_5[41:60], line_5[60:79], 
                            line_6[0:22]]
                
                
                # Change D character with e because D is not defined in Python. After   that, convert string to float.
                v_list = [float(i.replace('D','e')) for i in val_list]
                

                # Message object.
                sat = RinexMessage(num=t[0], epoch=sat_epoch,
                                  crs=v_list[0], delta_n=v_list[1], Mo=v_list[2],
                                  cuc=v_list[3], e=v_list[4], cus=v_list[5],   sqrt_a=v_list[6], 
                                  t0e=v_list[7], cic=v_list[8], omega0=v_list[9],   cis=v_list[10],
                                  i0=v_list[11], crc=v_list[12], omega=v_list[13],   omega_dot=v_list[14],
                                  idot=v_list[15], all_values=brdc_message, index=(start, finish))
                del lines
                return sat
            else:
                break
        else:
            start += 8
            finish += 8
def true_anomaly(eccentricity, mean_anomaly, tolerance=1e-6):
  """
  Verilen eksantriklik ve ortalama anomaliden true anomaly hesaplar.

  Args:
    eccentricity: Elipsin eksantrikliği.
    mean_anomaly: Ortalama anomalinin radyan cinsinden değeri.
    tolerance: Hata toleransı (varsayılan: 10^-6).

  Returns:
    True anomaly'nin radyan cinsinden değeri.
  """

  if eccentricity == 0:
    # Dairesel yörünge için true anomaly = mean anomaly
    return mean_anomaly

  e = eccentricity
  M = mean_anomaly

  def f(E):
    return E - e * np.sin(E) - M

  def df(E):
    return 1 - e * np.cos(E)

  E0 = M + np.pi * (1 - e) / (1 - e**2)  # İlk tahmin

  while abs(f(E0)) > tolerance:
    E1 = E0 - f(E0) / df(E0)
    E0 = E1

  return E0            

def cal_brdc(epoch, brd):
    
    omega_earth = 7.2921151467e-5 # rad/s
    GM = 3.986005e+14
    
    t = gpst2sod(epoch) # sec of day

    t0e = brd.t0e # sec of day

    t0e_sod = np.mod(brd.t0e,86400)
    tk = t - t0e_sod

    # Mo
    Mo = brd.Mo
    Mu = np.sqrt(GM)/np.sqrt(((brd.sqrt_a**2)**3))
    Mk = Mo + (Mu + brd.delta_n)*tk

    # Mo
    Mo = brd.Mo
    Mu = np.sqrt(GM)/np.sqrt(((brd.sqrt_a**2)**3))
    Mk = Mo + (Mu + brd.delta_n)*tk

    print(f"Mk: {Mk}")
    E_i = true_anomaly(brd.e, Mk)
    print(f"E_i: {E_i}")       
    
    # True anomaly.
    Vk = np.arctan(np.sqrt(1-brd.e**2)*np.sin(E_i)/np.cos(E_i)-brd.e)
    
    uk = brd.omega + Vk + brd.cuc*np.cos(2*(brd.omega+Vk)) + brd.cus*np.sin(2*(brd.omega+Vk))

    #rK value
    rk = (brd.sqrt_a**2)*(1-(brd.e*np.cos(E_i))) + brd.crc*np.cos(2*(brd.omega+Vk)) + brd.crs*np.sin(2*(brd.omega+Vk))

    # inclination
    ik = brd.i0 + brd.idot*tk + brd.cic*np.cos(2*(brd.omega+Vk)) + brd.cis*np.sin(2*(brd.omega+Vk))

    # Longitude
    lon_k = brd.omega0 + (brd.omega_dot-omega_earth)*tk - (omega_earth*t0e_sod)

    pos_vec = np.array([[rk], [0], [0]])

    #position vector.
    spos =  np.dot(np.dot(RotationMatrix(-uk,3), RotationMatrix(-ik, 1)), RotationMatrix(-lon_k, 3)) @ pos_vec
    
    return spos


my_epoch = Time(datetime(2024,3,1,9,15,42))
brdc_fp = "brdc0610.24n"   

brdc_message = message_finder(13, my_epoch, brdc_path="brdc0610.24n")
print(cal_brdc(my_epoch, brdc_message))    
