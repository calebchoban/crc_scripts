from . import dust_obs
import numpy as np

# This is the list of observations which have a file that can be loaded by a corresponding function.
# Each function will read a file, rename columns and create data produce, and then return a Pandas DataFrame with the file data.
# Update this as more files are added
SUPPORTED_FILES = {'HiZ_compilation': dust_obs.load_HiZ_compilation,
				   'galactic_DZ_compilation': dust_obs.load_galactic_DZ_compilation}

class Observational_Data(object):

	def __init__(self, obs_name, **kwargs):

		if obs_name not in SUPPORTED_FILES:
			print("Observation not supported.")
			return
		
		self.obs_name = obs_name
		# Load in pandas data frame
		self.data_frame = SUPPORTED_FILES[obs_name](**kwargs)

	
	def get_data(self, prop):
		if prop not in self.data_frame:
			print("%s is not in the specified date frame"%prop)
			return None
		else:
			return self.data_frame[prop].values
		
	
	def get_data_w_errors(self, prop):
		if prop not in self.data_frame:
			print("%s is not in the specified date frame"%prop)
			return None, None
		else:
			if prop+"_err_low" not in self.data_frame or prop+"_err_high" not in self.data_frame:
				print("%s errors are not in the specified date frame so they will be set to zero"%prop)
				return self.data_frame[prop].values, np.zeros((2,len(self.data_frame[prop].values)))
			else:
				return self.data_frame[prop].values, np.array([self.data_frame[prop+"_err_low"].values,self.data_frame[prop+"_err_high"].values])
		
	
	def available_data(self):
		print(self.data_frame.columns)