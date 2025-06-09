"""
GNSS - Assignment 1st
author @oguzhantosun


"""
# Import necessary modules.
import numpy as np
from math import sin, cos, atan2, sqrt, degrees


def xyz2plh(x,y,z, radian=bool):
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
    radian: bool
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
    a = 6378137.0       # meter
    b = 6356752.3141    # meter
    
    # Eccentricity (Attention! e²)
    e_2 = (pow(a, 2) - pow(b, 2)) / pow(a, 2)
    
    p = sqrt(pow(x, 2) + pow(y, 2)) # sqrt(x² + y²)
    
    # Initial values for iterative procedure.
    
    phi = atan2(z, (1- e_2)*p)
    lambda_ = atan2(y, x)
    h = 0                           # height
    
    while True:
        # Radius of curvature in the prime vertical
        N = a / sqrt(1 - e_2 * pow(sin(phi), 2))
        h_new = (p / cos(phi)) - N 
        phi_new = atan2((z/p), (1 - (e_2 * phi / phi+h)), )        
        
        # Check the precision
        if abs(phi_new - phi) < 1e-12:
            # If precision is lower than 10^(−12) stop the while loop.
            break        
        
        # Change the variables values with new values.
        phi = phi_new
        h = h_new
        
    if radian==True:
        return phi, lambda_, h
    else:
        # Convert radian to degree and return.
        return degrees(phi), degrees(lambda_), h
        
def calc_az_zen(x0, y0, z0, x1, y1, z1):
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
    delta_x = np.array([[x1-x0], 
                        [y1-y0], 
                        [z1-z0]])
    
    # Slant (radial) range from the topocenter to the target
    slantd = np.sqrt(delta_x[0]**2 + delta_x[1]**2 + delta_x[2]**2)
    
    # Latitude, longitude and height of topocenter.
    lat, lon, h = xyz2plh(x0, y0, z0, radian=True)
    
    # Rotation matrix.
    rotation_matrix = np.array([[-np.sin(lat), -np.cos(lat)*np.sin(lon), np.cos(lat)*np.cos(lon)], 
                                [np.cos(lat), -np.sin(lat)*np.sin(lon), np.sin(lat)*np.cos(lon)], 
                                [0, np.cos(lon), np.sin(lat)]])
    
    # Convert to ENU coordinates.
    ENU = np.dot(rotation_matrix, delta_x)
    
    # ENU coordinates for azimuth and zenith angle.
    east, north, up = ENU[0], ENU[1], ENU[2]
    
    # Zenith angle.
    zen = 90 - np.rad2deg(np.arcsin(up/np.sqrt(east**2 + north**2 + up**2)))
    
    # Azimuth angle.
    az = np.rad2deg(np.arctan2(north, east))
    
    # Return azimuth and zenith angle with distance.
    return az, zen, slantd
    
print(calc_az_zen(90,40,50,54,45,10))