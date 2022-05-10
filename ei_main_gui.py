import PySimpleGUI as sg
# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ieeg.auth import Session
from scipy.signal import convolve2d
from scipy import stats
from scipy import signal

#%% ei functions

def compute_hfer(target_data, base_data, fs):
    '''
    :param target_data: (Channels x Time) data with pre-ictal to ictal transition
    :param base_data: (Channels x Time) pre-ictal baseline data
    :param fs: sampling frequency
    :return: normalized high frequency energy for baseline and target data
    '''
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy=convolve2d(target_sq,np.ones((1,window)),'same')
    base_energy=convolve2d(base_sq,np.ones((1,window)),'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    target_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, target_energy.shape[1]))
    base_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, base_energy.shape[1]))
    norm_target_energy = target_energy / target_de_matrix.astype(np.float32)
    norm_base_energy = base_energy / base_de_matrix.astype(np.float32)
    return norm_target_energy, norm_base_energy

def get_threshold(norm_base_data, sd_val=10):
    '''
    :param norm_base_data: (channels x time) normalized baseline energy data
    :param sd_val: (int) how many standard deviations above the mean baseline energy to define the threshold per channel
    :return: thresh: threshold per channel
    '''
    thresh = np.max(norm_base_data,axis=1) + (sd_val*np.std(norm_base_data,axis=1, ddof=1))
    return thresh

def get_onset(norm_target_energy, norm_base_energy):

    # get the threshold for each channel
    thresh = get_threshold(norm_base_energy, 10)

    # define the onset vector
    onset = []

    # compute the onset time for each channel
    for i in range(len(thresh)):
        if np.sum(norm_target_energy[i,:] > thresh[i])>0:
            onset.append(np.argwhere(norm_target_energy[i,:] > thresh[i])[0][0])
        else:
            # if no onset is found, assign the onset time to the length of the signal
            onset.append(len(norm_target_energy[i,:]))

    rank = stats.rankdata(onset)+1
    #rank = np.argsort(onset)+1
    tc = 1/rank
    print(np.min(onset))
    return onset, tc

def calculate_ei(norm_target_energy, fs, onset, tc):
    ec = np.mean(norm_target_energy[onset:onset + int(fs * 0.250)]) # energy coefficient, mean of energy 250ms after detection
    return np.sqrt(ec*tc)

def get_ei_all(target, base, fs):
    norm_target_energy, norm_base_energy = compute_hfer(target, base, fs)
    onset, tc = get_onset(norm_target_energy, norm_base_energy)

    ei_vec = []
    for i in range(len(onset)):
        ei_vec.append(calculate_ei(norm_target_energy[i,:],fs,np.min(onset), tc[i]))
    ei_vec = np.array(ei_vec)

    if np.isnan(np.nanmax(ei_vec)):
        ei_vec = np.zeros(len(ei_vec))
    else:
        ei_vec[np.isnan(ei_vec)] = 0
        ei_vec = ei_vec / np.nanmax(ei_vec)
    return ei_vec

#% Extra functions
def determine_threshold_onset(target, base):
    base_data = base.copy()
    target_data = target.copy()
    sigma = np.std(base_data, axis=1, ddof=1)
    channel_max_base = np.max(base_data, axis=1)
    thresh_value = channel_max_base + 10 * sigma
    onset_location = np.zeros(shape=(target_data.shape[0],))
    for channel_idx in range(target_data.shape[0]):
        logic_vec = target_data[channel_idx, :] > thresh_value[channel_idx]
        if np.sum(logic_vec) == 0:
            onset_location[channel_idx] = len(logic_vec)
        else:
            onset_location[channel_idx] = np.where(logic_vec != 0)[0][0]
    return onset_location


def compute_ei_index(target, base, fs):
    target, base = compute_hfer(target, base, fs)
    ei = np.zeros([1, target.shape[0]])
    hfer = np.zeros([1, target.shape[0]])
    onset_rank = np.zeros([1, target.shape[0]])
    channel_onset = determine_threshold_onset(target, base)
    seizure_location = np.min(channel_onset)
    onset_channel = np.argmin(channel_onset)
    hfer = np.sum(target[:, int(seizure_location):int(seizure_location + 0.25 * fs)], axis=1) / (fs * 0.25)
    print(hfer)
    onset_asend = np.sort(channel_onset)
    time_rank_tmp = np.argsort(channel_onset)
    onset_rank = np.argsort(time_rank_tmp) + 1
    onset_rank = np.ones((onset_rank.shape[0],)) / np.float32(onset_rank)
    ei = np.sqrt(hfer * onset_rank)
    for i in range(len(ei)):
        if np.isnan(ei[i]) or np.isinf(ei[i]):
            ei[i] = 0
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return ei

#%%


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}

file_list_column = [
    [
        sg.Text("IEEG Username"),
        sg.In(size=(25,1), enable_events=True, key='-USER-'),
    ],
    [
        sg.Text("IEEG Password"),
        sg.In(size=(25, 1), enable_events=True, key='-PW-',password_char='*'),
    ],
    [
        sg.Text("Dataset Name"),
        sg.In(size=(25,1), enable_events=True, key='-DATASET-')
    ],
    [
        sg.Text("Seizure start time (in seconds)"),
        sg.In(size=(25,1), enable_events=True, key='-START-')
    ],
    [
        sg.Text("Electrode Name"),
        sg.In(size=(25,1), enable_events=True, key='-ELEC-')
    ],
    [
        sg.Listbox(size=(10,5),values=[], key='-ELEC_LIST-'),
        sg.Button('Update List', key='-UPDATE_ELEC-')
    ],
    [
      sg.Text("Baseline Times (in samples)"),
      sg.In(size=(10,1), enable_events=True, key='-BL_START-'),
      sg.In(size=(10,1), enable_events=True, key='-BL_END-'),
      sg.Button('Reset Baseline',key='-RESET_BASELINE-')
    ],
    [
        sg.Text("Target Times (in samples)"),
        sg.In(size=(10, 1), enable_events=True, key='-TARG_START-'),
        sg.In(size=(10, 1), enable_events=True, key='-TARG_END-'),
        sg.Button('Reset Target', key='-RESET_TARGET-')
    ],
    [
        sg.Text("Electrodes to Exclude"),
        sg.In(size=(25, 1), enable_events=True, key='-EXCLUDE_ELECTRODES-'),

    ],
    [
        sg.Listbox(size=(10,5),values=[], key='-ELEC_LIST_EXCLUDE-', select_mode='extended'),
        sg.Button('Exclude', key='-EXCLUDE-')
    ],
    [
        sg.Text("SOZ Electrodes")
    ],
    [
        sg.Listbox(size=(10, 5), values=[], key='-ELEC_LIST_SOZ-', select_mode='extended'),
    ],
    [
        sg.Button('Plot',key='-PLOT-'),
        sg.Button('Get EI', key='-EI-'),
        sg.Button('Re-Plot EI', key='-REPLOT-')
    ],
    [
        sg.In(size=(25, 1), enable_events=True, key='-SAVE_NAME-'),
        sg.Button('Save EI', key='-SAVE-')
    ],
]

image_viewer_column = [
    [sg.Canvas(key='figCanvas')],
]

# -- full layout --
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column(image_viewer_column)
    ]
]

_VARS['window'] = sg.Window("Image Viewer", layout, finalize=True)

# -- IEEG --
def open_ieeg_session(pw_path):
    with open('/Users/allucas/Documents/research/CNT/P23_Epileptogenicity_Index/source_data/us_pw_ieee', 'r') as f: session = Session('allucas', f.read())
    return session

def open_ieeg_session_us_pw(us,pw):
    session = Session(us, pw)
    return session

def get_ieeg_dataset(session,dataset_name):
    dataset = session.open_dataset(dataset_name)
    return dataset

def get_ieeg_timeseries(dataset, start, ch_idx):
    data = dataset.get_data((start*1e6)-100e6,300e6, [ch_idx])
    return data.reshape(-1)

def get_ieeg_timeseries_all(dataset, start, list_elect):
    data = dataset.get_data((start*1e6)-100e6,300e6, list_elect)
    return data

def get_electrode_index(dataset, ch_name):
    elect = np.array(dataset.get_channel_labels(), dtype=object)
    return np.where(elect==ch_name)[0][0]

def get_electrode_index_from_reduced(ch_all, ch_name, ch_skip):
    ch_all_new = []
    for i in range(len(ch_all)):
        if ch_all[i] in ch_skip:
            continue
        else:
            ch_all_new.append(ch_all[i])
    elect = np.array(ch_all_new, dtype=object)
    return np.where(elect==ch_name)[0][0]

def exclude_electrodes(original_list, remove):
    electrode_index = []
    for i in range(len(original_list)):
        keep = True
        for j in range(len(remove)):
            if original_list[i]==remove[j]:
                keep=False
        if keep:
            electrode_index.append(i)
    return electrode_index

def get_electrode_index_list(dataset, ch_names, ch_skip):
    index_list = []
    for i in range(len(ch_names)):
        index_list.append(get_electrode_index_from_reduced(dataset.get_channel_labels(), ch_names[i], ch_skip))
    return index_list

# -- EI --
def get_ei_from_data(data, fs,  bl_range, target_range):
    data = data.T
    # filter the signal
    if int(fs/2)<140:
        b, a = signal.butter(4, [70,int(fs/2)-1], 'bandpass', fs=fs)
    else:
        b, a = signal.butter(4, [70, 140], 'bandpass', fs=fs)
    data_filt = signal.filtfilt(b, a, data)

    #%
    #ei = get_ei_all(data_filt[:,20000:60000],data_filt[:,:20000],fs=fs)
    ei = compute_ei_index(data_filt[:, target_range[0]:target_range[1]], data_filt[:, bl_range[0]:bl_range[1]], fs=fs)
    return ei
# -- plots --


# Initialize empty figure
fig = plt.figure()
# Instead of plt.show
_VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

def plot_ieeg_timeseries(data, fs, bl_range, target_range, onset=100):
    _VARS['fig_agg'].get_tk_widget().forget()
    # make fig and plot
    fig = plt.figure()
    plt.plot(data, 'k', linewidth=0.2)
    plt.vlines(x=bl_range, ymin=np.min(data), ymax=np.max(data), color='blue')
    plt.vlines(x=target_range, ymin=np.min(data), ymax=np.max(data), color='green')
    plt.vlines(x=onset*fs, ymin=np.min(data), ymax=np.max(data), color='red')
    sns.despine()
    # Instead of plt.show
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

def plot_ei(ei,ch_names, soz=[]):
    _VARS['fig_agg'].get_tk_widget().forget()
    # make fig and plot
    fig = plt.figure()
    plt.bar(np.arange(len(ei)), ei)
    if soz!=[]:
        plt.bar(soz,ei[soz])
    plt.xticks(np.arange(len(ei)), ch_names, rotation=90, fontsize=6)
    sns.despine()
    plt.ylabel('EI')
    plt.xlabel('Electrode')
    plt.legend(['Non-SOZ', 'SOZ'])
    # Instead of plt.show
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

def save_ei(directory, fname, ei, ch_names):
    ei_table = []
    for i in range(len(ei)):
        ei_table.append([ch_names[i], ei[i]])

    ei_table = np.array(ei_table, dtype=object)
    np.savetxt(directory+'/'+fname,ei_table, delimiter=',', fmt='%s')

# -- Window with table of EI values --
def open_EI_table(ei, ch_names):

    ei_table = []

    ei_idx_sorted = np.argsort(ei)
    for i in range(len(ei)):
        ei_table.append([ch_names[ei_idx_sorted[i]],ei[ei_idx_sorted[i]]])

    headings = ['Electrode','EI']

    layout = [
        [sg.Table(values=ei_table,headings=headings,
                  auto_size_columns=True,
                  display_row_numbers=True,
                  num_rows=10,
                  key='-TABLE-')]
    ]

    window = sg.Window("EI Table", layout, modal=True)
    choice = None
    event, values = window.read()
    # while True:
    #     event, values = window.read()
    #     if event == "Exit" or event == sg.WIN_CLOSED:
    #         break

    #window.close()

# -- main loop --

bl_range = []
target_range = []
exclude_list = []

while True:
    event, values = _VARS['window'].read()
    if event == 'Exit' or event==sg.WIN_CLOSED:
        break
    if event == '-FOLDER-':
        continue
    if event == '-PLOT-':



        #pw_file = values['-FOLDER-']
        session = open_ieeg_session_us_pw(values['-USER-'], values['-PW-'])
        dataset = get_ieeg_dataset(session, values['-DATASET-'])

        fs = int(dataset.get_time_series_details(dataset.get_channel_labels()[0]).sample_rate)

        # Default values for baseline and target regions in signal
        if bl_range==[]:
            bl_range = [0, int(60 * fs)]
        if target_range==[]:
            target_range = [int(60 * fs),int(160 * fs)]

        if values['-ELEC_LIST-'] == []:
            electrode_to_plot = values['-ELEC-']
        else:
            electrode_to_plot = values['-ELEC_LIST-'][0]

        data = get_ieeg_timeseries(dataset, np.float(values['-START-']), get_electrode_index(dataset,electrode_to_plot))
        plot_ieeg_timeseries(data, fs, bl_range, target_range)

    if event=='-RESET_BASELINE-':
        bl_range = [np.int(values['-BL_START-']),np.int(values['-BL_END-'])]

    if event=='-UPDATE_ELEC-':
        #pw_file = values['-FOLDER-']
        session = open_ieeg_session_us_pw(values['-USER-'], values['-PW-'])
        dataset = get_ieeg_dataset(session, values['-DATASET-'])
        _VARS['window'].FindElement('-ELEC_LIST-').Update(values=dataset.get_channel_labels())
        _VARS['window'].FindElement('-ELEC_LIST_EXCLUDE-').Update(values=dataset.get_channel_labels())
        _VARS['window'].FindElement('-ELEC_LIST_SOZ-').Update(values=dataset.get_channel_labels())

    if event == '-RESET_TARGET-':
        target_range = [np.int(values['-TARG_START-']), np.int(values['-TARG_END-'])]

    if event == '-EXCLUDE-':
        #exclude_list = values['-EXCLUDE_ELECTRODES-'].split(',')
        exclude_list = values['-ELEC_LIST_EXCLUDE-']
        #pw_file = values['-FOLDER-']
        session = open_ieeg_session_us_pw(values['-USER-'], values['-PW-'])
        dataset = get_ieeg_dataset(session, values['-DATASET-'])

        ch_all = dataset.get_channel_labels()
        ch_keep = exclude_electrodes(ch_all, exclude_list)

    if event == '-EI-':
        #pw_file = values['-FOLDER-']
        session = open_ieeg_session_us_pw(values['-USER-'], values['-PW-'])
        dataset = get_ieeg_dataset(session, values['-DATASET-'])

        fs = int(dataset.get_time_series_details(dataset.get_channel_labels()[0]).sample_rate)

        # Default values for baseline and target regions in signal
        if bl_range==[]:
            bl_range = [0, int(60 * fs)]
        if target_range==[]:
            target_range = [int(60 * fs), int(160 * fs)]

        if exclude_list==[]:
            ch_keep = list(np.arange(len(dataset.get_channel_labels())))

        data = get_ieeg_timeseries_all(dataset, np.float(values['-START-']), ch_keep)
        ei = get_ei_from_data(data, fs, bl_range, target_range)
        plot_ei(ei, list(np.array(dataset.get_channel_labels())[ch_keep]),
                soz=get_electrode_index_list(dataset,values['-ELEC_LIST_SOZ-'],values['-ELEC_LIST_EXCLUDE-']))
        open_EI_table(ei, list(np.array(dataset.get_channel_labels())[ch_keep]))

    if event =='-REPLOT-':
        try:
            plot_ei(ei, list(np.array(dataset.get_channel_labels())[ch_keep]),
                    soz=get_electrode_index_list(dataset,values['-ELEC_LIST_SOZ-'],values['-ELEC_LIST_EXCLUDE-']))
            open_EI_table(ei, list(np.array(dataset.get_channel_labels())[ch_keep]))
        except:
            continue

    if event=='-SAVE-':
        save_ei('/Users/allucas/Documents/research/CNT/P23_Epileptogenicity_Index/outputs',values['-SAVE_NAME-'], ei, list(np.array(dataset.get_channel_labels())[ch_keep]))
_VARS['window'].close()