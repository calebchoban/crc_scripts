# crc_scripts
Personal routines for reading in FIRE snapshots with dust evolution modules from C. R. Choban et al. (2022). This includes function for plotting dust properties, dust observations included in my recent publications, and Jupyter notebook templates.

To install, first make a new folder in your home directory where you will store all of your codes. I call mine /codes. 
Then, git clone this repository
```console
cd codes
git clone https://github.com/calebchoban/crc_scripts.git
```
and then use pip to install it to your Python/Conda environment
```bash
 pip install -e .
```

Note that all compiled observational data is not included in this repository since some of the data may not be published and/or is from other authors. Please reach out to me if you want to use any of the observational data.

For a brief tutorial on using these scripts to read FIRE snapshot data, check out the snapshot_tutorial Jupiter notebook.
