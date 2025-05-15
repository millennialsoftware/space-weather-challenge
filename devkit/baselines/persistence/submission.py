import torch
from pathlib import Path
import dill
import time
import os
import pandas as pd
from datetime import datetime
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.utils import Constants
import json
import glob
from propagator import prop_orbit
from atm import * 


def generate_ground_truth(): 

    BASELINE_MODEL_DIR = Path('trained_model')
    TEST_DATA_DIR = Path('../../../data/train') # Modify to test once data is uploaded
    TEST_PREDS_FP = Path('../../../submission/ground_truth.json')
    
    # Paths
    initial_states_file = os.path.join(TEST_DATA_DIR, "initial_states/00000_to_02284-initial_states.csv")
    omni2_path = os.path.join(TEST_DATA_DIR, "omni2")
    sat_density_path = os.path.join(TEST_DATA_DIR, "sat_density")
    
    initial_states = pd.read_csv(initial_states_file)
    
    # Load initial states
    initial_states = pd.read_csv(initial_states_file, usecols=['File ID', 'Timestamp', 'Semi-major Axis (km)', 'Eccentricity', 'Inclination (deg)','RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)'])
    print(initial_states.columns)
    initial_states = initial_states[initial_states['File ID'] == 2200]
    initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'])
    
    # Process each row of the initial states
    model = torch.load(f"{BASELINE_MODEL_DIR}/persistence_model.pkl", pickle_module=dill)
    
    predictions = {}
    count = 0
    for _, row in initial_states.iterrows():

        if count >= 1:
            break
            
        file_id = row['File ID']
        # Load corresponding OMNI2 data
        omni2_file = os.path.join(omni2_path, f"omni2-{file_id:05}.csv")
        sat_density_file_pattern = os.path.join(sat_density_path, f"*-{file_id:05d}.csv")
        matching_files = glob.glob(sat_density_file_pattern)
                
        if matching_files:
            sat_density_file = matching_files[0]  # Use the first matching file
        else:
            sat_density_file = ""

        
        if not os.path.exists(omni2_file):
            print(f"OMNI2 file {omni2_file} not found! Skipping...")
            continue

        if not os.path.exists(sat_density_file):
            print(f"Sat Density file {sat_density_file} not found! Skipping...")
            continue
        
        omni2_data = pd.read_csv(omni2_file, usecols=['Timestamp', 'f10.7_index', 'ap_index_nT'])
        sat_density_data = pd.read_csv(sat_density_file)
        omni2_data['Timestamp'] = pd.to_datetime(omni2_data['Timestamp'])
        omni2_data = omni2_data.ffill()  
    
        initial_state = row.drop("File ID")
    
        result = model(omni2_data, initial_state.to_dict())
    
        predictions[file_id] = {
            "Timestamp": sat_density_data["Timestamp"]
                .apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%dT%H:%M:%S'))
                .tolist(),
            "Orbit Mean Density (kg/m^3)": sat_density_data['Orbit Mean Density (kg/m^3)'].tolist(),
            "MSIS Orbit Mean Density (kg/m^3)": result["Density (kg/m3)"].tolist()
        }


        print(f"Model execution for {file_id} Finished")
    
        count += 1
    
    with open(TEST_PREDS_FP, "w") as outfile: 
        json.dump(predictions, outfile)
    
    print("Saved predictions to: {}".format(TEST_PREDS_FP))
    # time.sleep(360) # EVALAI BUG FIX

generate_ground_truth()