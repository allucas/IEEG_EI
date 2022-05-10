# IEEG_EI

This repository provides a simple Python-based GUI for calculating the Epileptogenicity Index (EI) on data stored on iEEG.org. It uses the implementation of the EI proposed here: https://www.frontiersin.org/articles/10.3389/fninf.2021.773890/full#F4

Make sure you have an active ieeg.org account: https://www.ieeg.org/

# Required Packages

Make sure you have:
- PySimpleGUI: https://pysimplegui.readthedocs.io/en/latest/
- SciPy: https://scipy.org/
- Numpy: https://numpy.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- ieegpy: https://github.com/ieeg-portal/ieegpy

Installing the dependencies:
```
pip install numpy
pip install scipy
pip install PySimpleGUI
pip install seaborn
pip install matplotlib
pip install git+https://github.com/ieeg-portal/ieegpy.git
```

If you installed Python through Anaconda, then you probably only need:
```
pip install git+https://github.com/ieeg-portal/ieegpy.git
pip install PySimpleGUI
```

# Running the IEEG_EI GUI

## Setting Up Output Folder
Before running, change line `1` in `ei_main_gui.py` to:

```
output_folder = 'OUTPUT_FOLDER_PATH'
```

Such that `OUTPUT_FOLDER_PATH` is the path to the output folder you want to save the calculated EI values to. For example:

```
output_folder = '~/allucas/ei_gui_outputs'
```

Note the lack of a trailing '/' in the above example.

## Running the GUI
To run the IEEG_EI GUI type:

```
python ei_main_gui.py
```

# Example Usage

## Visualizing a seizure epoch

1. Once the GUI is open enter your IEEG.org username and password, as well as the name of a dataset that you have access to on the server

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step1.png?raw=true)

2. If the username, password and datasets are correct, you should be able to click the `Update List` button which will pull the names of all the electrodes available in that dataset

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step1.png?raw=true)

3. Enter the time, in seconds, of a known seizure onset within that dataset

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step3.png?raw=true)

4. Select an electrode that you would like to see the timeseries of.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step4.png?raw=true)

5. Press the `Plot` button, which will show the epoch at the selected channel. 

By default the algorithm plots 100 seconds prior to the specified seizure onset time, and 200 seconds after the seizure onset time

The baseline is set by default to an 60 second window starting 100 seconds prior to the seizure onset.

The target (which must contain the transition from pre-ictal to ictal) is set by default from the end of the baseline (i.e. 40s prior to seizure onset) to  60s after seizure onset.

The range for the baseline and target regions can be manually set in the provided boxes. The values must be provided in samples, not seconds.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step5.png?raw=true)

## Calculating the EI of the chosen epoch
After following the steps from the previous section, an epoch with a seizure in it was selected. Now the EI can be calculated across all channels for this particular epoch.

1. Prior to calculating the EI, electrodes that we wish to exclude can be selected. Make sure to press the `Exclude` button after selection.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step6.png?raw=true)

2. Electrodes that belong to the seizure onset zone can also be selected. This will not affect the calculation of the EI, but will overlay those electrodes in the final EI plot for ease of visualization.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step7.png?raw=true)

3. Finally, by pressing the `Get EI` button, the EI algorithm will run.

A plot of the distribution of EI values will be generated. Note that the values are normalized by the maximum EI calculated, generating a range between 0-1. A sorted table with these values is also generated for visualization purposes.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step8.png?raw=true)

4. In order to go back to the main window and perform more actions. Please close the table window. In order to save the calculated EI values into a `.csv` file, enter a filename and press the `Save` button. The output directory is defined in line `1` of the script as per the setup instructions.

![alt text](https://github.com/allucas/IEEG_EI/blob/main/images/step9.png?raw=true)


