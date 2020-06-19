import numpy as np

# Luminosity index legend
#  0 = bolometric luminosity
#  1 = Johnsons U
#  2 = Johnsons B
#  3 = Johnsons V
#  4 = Johnsons R
#  5 = Johnsons I
#  6 = Cousins J
#  7 = Cousins H
#  8 = Cousins K
#  9 = Sloan u
# 10 = Sloan g
# 11 = Sloan r
# 12 = Sloan i
# 13 = Sloan z

colors_available = ['bol','U','Johnsons U','B','Johnsons B','V','Johnsons V','R','Johnsons R','I','Cousins I','J','Cousins J','H','Cousins H','K','Cousins K','u','Sloan u','sloan u','g','Sloan g','sloan g','r','Sloan r','sloan r','i','Sloan i','sloan i','z','Sloan z','sloan z']


# get band ID from a band name/ID
def get_band_ID_single(band):

    if type(band) is int:
        band_ID = 0 if (band<0) or (band>13) else band
    elif (band=='bol'):
        band_ID = 0
    elif (band=='U') or (band=='Johnsons U'):
        band_ID = 1
    elif (band=='B') or (band=='Johnsons B'):
        band_ID = 2
    elif (band=='V') or (band=='Johnsons V'):
        band_ID = 3
    elif (band=='R') or (band=='Johnsons R'):
        band_ID = 4
    elif (band=='I') or (band=='Johnsons I'):
        band_ID = 5
    elif (band=='J') or (band=='Cousins J'):
        band_ID = 6
    elif (band=='H') or (band=='Cousins H'):
        band_ID = 7
    elif (band=='K') or (band=='Cousins K'):
        band_ID = 8
    elif (band=='u') or (band=='Sloan u') or (band=='sloan u'):
        band_ID = 9
    elif (band=='g') or (band=='Sloan g') or (band=='sloan g'):
        band_ID = 10
    elif (band=='r') or (band=='Sloan r') or (band=='sloan r'):
        band_ID = 11
    elif (band=='i') or (band=='Sloan i') or (band=='sloan i'):
        band_ID = 12
    elif (band=='z') or (band=='Sloan z') or (band=='sloan z'):
        band_ID = 13
    else:
        band_ID = 0

    return band_ID

get_band_ID = np.vectorize(get_band_ID_single)


# get band effective wavelength/frequency
import colors_sps.colors_table as ctab

def get_band_nu_eff_single(band):

    band_ID = get_band_ID(band)
    nu_eff = ctab.colors_table(0, 0, BAND_ID=band_ID, RETURN_NU_EFF=1)

    return nu_eff

get_band_nu_eff = np.vectorize(get_band_nu_eff_single)


def get_band_lambda_eff_single(band):

    band_ID = get_band_ID(band)
    lambda_eff = ctab.colors_table(0, 0, BAND_ID=band_ID, RETURN_LAMBDA_EFF=1)

    return lambda_eff

get_band_lambda_eff = np.vectorize(get_band_lambda_eff_single)


# get luminosity-to-mass ratio for a stellar population
def colors_table(age_in_Gyr, metallicity_in_solar_units, band=0, SALPETER_IMF=0,
                 CHABRIER_IMF=1, UNITS_SOLAR_IN_BAND=1):

    band_ID = get_band_ID_single(band)
    return ctab.colors_table(age_in_Gyr, metallicity_in_solar_units, BAND_ID=band_ID,
                             SALPETER_IMF=SALPETER_IMF, CHABRIER_IMF=CHABRIER_IMF,
                             UNITS_SOLAR_IN_BAND=UNITS_SOLAR_IN_BAND)
