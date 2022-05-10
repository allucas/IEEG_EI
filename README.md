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

# Running IEEG_EI

## Setting Up Output Folder
Before running, change line 444 in `ei_main_gui.py`:

```
save_ei('OUTPUT_FOLDER',values['-SAVE_NAME-'], ei, list(np.array(dataset.get_channel_labels())[ch_keep]))
```

Such that `OUTPUT_FOLDER` is the path to the output folder you want to save the calculated EI values to. For example:

```
save_ei('~/allucas/ei_gui_outputs',values['-SAVE_NAME-'], ei, list(np.array(dataset.get_channel_labels())[ch_keep]))
```

## Running the GUI
To run the IEEG_EI GUI simply type:

```
python ei_main_gui.py
```

