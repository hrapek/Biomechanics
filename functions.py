#standard
import os
import glob

#third-part
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

#local
from utils import joint_dict

def load_pickles(folder='pickles', files=None):

    '''
    Loads pickle files from folder. Either specified list in files or all pkl files inside the folder.

    INPUTS
    ------
        folder : str, default 'pickles'
            Directory where pickle files are stored.
        files : list, default None
            List of files to load.

    OUTPUTS
    -------
        model_outputs : dict
            Dictionary of loaded pickle results in PHALP schema (which are also dicts).
    '''


    #storage dict
    model_outputs = {}

    #only specified files
    if files:
        for file in files:
            name = os.path.splitext(file)[0]
            model_outputs[name] = joblib.load(f'{folder}/{file}')

    #everything in pickles directory
    else:
        pickle_files = glob.glob(f'{folder}/*.pkl')
        for file in pickle_files:
            name = os.path.splitext(os.path.basename(file))[0]
            model_outputs[name] = joblib.load(file)

    return model_outputs

def extract_joints(model_output):

    '''
    Extracts joints from pickle file for every detected person.

    Basically transforms dictionary obtained through PHALP into more useful schema:

    tracklets = {

        #dict for each person
        'person_id' : {
            #dict for each joint
            'joint_name' : {
                #dict for each dimension
                'dimension':    List[numpy.float32]     #value for every detected frame
            }
        }

    }

    INPUTS
    ------
        model_output : dict
            phalp_outputs dictionary of single video.

    OUTPUTS
    -------
        tracklets : dict
            Dictionary of all tracklets in the video with their 3d_joints in 3 dimensions.
    '''

    #check how many people are in the video:
    n_tracklets = 0
    for frame in model_output.values():
        n_tracklets = max(n_tracklets, len(frame['tracked_ids']))

    #initialize joints dict with empty list for every detection
    tracklets = {n: {joint: {dim: np.empty(0) for dim in range(3)} for joint in joint_dict} for n in range(n_tracklets)}

    #loop over frames and extract joints
    for frame in model_output.values():
        for person in frame['tracked_ids']:
            person -= 1
            for joint in joint_dict:
                for dim in range(3):
                    tracklets[person][joint][dim] = np.append(tracklets[person][joint][dim], frame['3d_joints'][person][joint_dict[joint]][dim])

    return tracklets

def plot_joints_trajectory(tracklets, person_id, joint_list, dim):

    '''
    Plots joint trajectories for specified video, person, joint list and dimension.

    INPUTS
    ------
        tracklets : dict
            Dict of tracklets from video obtained thorugh extract_joints().
        person_id : int
            Index of tracked person to plot.
        joint_list : list
            List of joints to plot.
        dim : int
            Dimension to plot from camera point of view (0 - x axis, 1 - y axis, 2 - z axis).

    '''

    trajectories = {}

    for joint in joint_list:
        trajectories[joint] = tracklets[person_id][joint][dim]
        plt.plot(range(len(trajectories[joint])), trajectories[joint])

    #for title purposes
    joint_names = ' & '.join(joint_list)

    plt.legend(joint_list)
    plt.title(joint_names)
    plt.show()

def moving_average(data, window_size=3):
    """
    Apply moving average smoothing to the data.

    Parameters:
    - data: pandas Series or DataFrame
        The data to be smoothed.
    - window_size: int
        The size of the moving window.

    Returns:
    - pandas Series or DataFrame
        The smoothed data.
    """
    return data.rolling(window=window_size).mean()

def reject_outliers(data, m=1):

    cleaned_data = data[abs(data - np.mean(data)) < m * np.std(data)]
    indices = abs(data - np.mean(data)) < m * np.std(data)

    return cleaned_data, indices

def get_step_metrics(tracklets, video, person_id, dim, joint='Heel', smoothing=False, _print=False):

    '''
    Caculcates bunch of step related biometrics for heels or ankles.

    INPUTS
    ------
        tracklets : dict
            Dictionary of all tracklets in the video with their 3d_joints in 3 dimensions.
        video : str
            Name of video.
        person_id : int
            Index of tracked person.
        dim_step : int
            Dimension to plot from camera point of view (0 - x axis, 1 - y axis, 2 - z axis).
        dim_asym : int
            Dimension used to calculate asymmetry
        joint : str
            Joint on which metrics are based. Must be either Heel or Ankle.
        smoothing : bool, default False
            If True then moving average smoothing is applied with window_size 3.
        print : bool, default False
            If True then prints function output

    OUTPUTS
    -------
        steps_length : np.array
            Array of step lengths.
        avg_step_length : float
            Average step length.
        speed : float
            Average speed from first to last step in m/timeframes.
        time : int
            Time from first to last step in timeframes.
        distance : float
            Distance travelled from first to last step in meters.
    '''

    assert joint in ['Heel','Ankle'], 'Joint must be heel or ankle'

    #get ankles
    r = tracklets[video][person_id][f'R{joint}'][dim]
    l = tracklets[video][person_id][f'L{joint}'][dim]

    if smoothing:
        r = np.array(moving_average(pd.DataFrame(r))).ravel()
        l = np.array(moving_average(pd.DataFrame(l))).ravel()

    #calculate difference between ankles in the dimension
    difference = np.absolute(np.subtract(r, l))

    #indices of local maxima
    maxima = argrelextrema(difference, np.greater)[0]

    #steps length
    steps_length = difference[maxima]

    #get rid of outliers
    steps_length, indices = reject_outliers(steps_length)

    #speed
    time = np.ptp(maxima[indices])
    distance = np.sum(steps_length[1:])
    speed = distance/time #meters per timeframe

    #average step
    avg_step_length = np.average(steps_length)
    if _print:
        print(f'Length of steps: \n{steps_length}')
        print(f'Average step length: {avg_step_length}')
        print(f'Speed: {speed}')
        print(f'Time (timeframes): {time}')
        print(f'Distance: {distance}')
    
    return steps_length, avg_step_length, speed, time, distance

def get_asymmetry(tracklets, video, person_id, dim, joint='Hip', smoothing=False, _print=False):
    
    '''
    Caculcates asymmetry based on the hips.

    INPUTS
    ------
        tracklets : dict
            Dictionary of all tracklets in the video with their 3d_joints in 3 dimensions.
        video : str
            Name of video.
        person_id : int
            Index of tracked person.
        dim : int
            Dimension to plot from camera point of view (0 - x axis, 1 - y axis, 2 - z axis).
        joint : str
            Joint on which metrics are based. Must be either Heel or Ankle.
        smoothing : bool, default False
            If True then moving average smoothing is applied with window_size 3.
        print : bool, default False
            If True then prints function output

    OUTPUTS
    -------
        asymmetry : float
            Asymmetry value.
    '''

    assert joint in ['Heel','Ankle', 'Hip'], 'Joint must be heel, ankle or hip'

    # get joints
    r = tracklets[video][person_id][f'R{joint}'][dim]
    l = tracklets[video][person_id][f'L{joint}'][dim]
    
    # absoulte values
    r = np.absolute(r)
    l = np.absolute(l)
    
    if smoothing:
        r = np.array(moving_average(pd.DataFrame(r))).ravel()
        l = np.array(moving_average(pd.DataFrame(l))).ravel()
    
    # get rid of nans
    r = r[~np.isnan(r)]
    l = l[~np.isnan(l)]

    asymmetry = 1 - (np.trapz(r) / np.trapz(l)) # > 0: asymmetry towards right, < 0: asymmetry towards left
    
    if _print:
        print(f'Assymetry: {asymmetry}')
    
    return asymmetry

def process_biometrics_df(folder='pickles'):
    
    '''
    Produces dataframe with caculated biometrics for specified pickle files.

    INPUTS
    ------
        folder : str, default 'pickles'
            Directory where pickle files are stored.
        files : list, default None
            List of files to load.
        smoothing : bool, default False
            If True then moving average smoothing is applied with window_size 3.

    OUTPUTS
    -------
        df : pd.DataFrame
            DataFrame with calculated biomechanics.
    '''
    
    model_outputs = load_pickles(folder=folder)

    tracklets_dict = {}

    for video_name, video_results in model_outputs.items():
        tracklets_dict[video_name] = extract_joints(video_results)
    
    output_dict = {"walking_type": [], "video_id": [], "person_id": [], "camera_type": [], "steps_length": [], 
                   "avg_step_length": [], "speed": [], "time": [], "distance": [], "asymmetry": []}
    
    for video in tracklets_dict:
        n_tracklets = 0
        video_name = video.split("demo_")[1]
        walking_type, video_id, camera_type = video_name.split("-")
        if camera_type == "side":
            dim_step = 0
            dim_asym = 2
        else:
            dim_step = 2
            dim_asym = 0

        person_id = 0
        steps_length, avg_step_length, speed, time, distance = get_step_metrics(tracklets_dict, video, person_id=person_id, 
                                                                                dim=dim_step, joint='Heel', smoothing=False)
        asymmetry = get_asymmetry(tracklets_dict, video, person_id=person_id, dim=dim_asym, joint='Hip', smoothing=False)
        print(asymmetry)
        output_dict["walking_type"].append(walking_type)
        output_dict["video_id"].append(video_id)
        output_dict["person_id"].append(person_id)
        output_dict["camera_type"].append(camera_type)
        output_dict["steps_length"].append(steps_length)
        output_dict["avg_step_length"].append(avg_step_length)
        output_dict["speed"].append(speed)
        output_dict["time"].append(time)
        output_dict["distance"].append(distance)
        output_dict["asymmetry"].append(asymmetry)
        
    df = pd.DataFrame(output_dict)
    
    return df
