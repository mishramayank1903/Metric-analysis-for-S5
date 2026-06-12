import os
import yaml
import torch
import torchaudio

def get_sources_paths(output_dir):
    mix_source_1_paths = {}
    mix_source_2_paths = {}
    mix_source_3_paths = {}
    mix_paths = {}
    mix_yamls = {}

    for i in range(1, 501):
        idx = str(i).zfill(3)
        mix_source_1_paths[i] = os.path.join(output_dir, f'foa/audio_mix_metu_2_{idx}/wet_source_1.wav')
        mix_source_2_paths[i] = os.path.join(output_dir, f'foa/audio_mix_metu_2_{idx}/wet_source_2.wav')
        mix_source_3_paths[i] = os.path.join(output_dir, f'foa/audio_mix_metu_2_{idx}/wet_source_3.wav')
        mix_paths[i] = os.path.join(output_dir, f'foa/audio_mix_metu_2_{idx}/audio_mix_metu_2_{idx}.wav')
        mix_yamls[i] = os.path.join(output_dir, f'yaml/audio_mix_metu_2_{idx}.yaml')


    return mix_source_1_paths, mix_source_2_paths, mix_source_3_paths, mix_paths, mix_yamls    



def add_background_noise(wet_source, start_idx, end_idx, noise, snr_db):
    
    # Calculate scaling factor for desired SNR
    signal_power = torch.mean(wet_source[start_idx :end_idx]**2)
    noise_power = torch.mean(noise[start_idx :end_idx]**2)
    required_noise_power = signal_power / (10**(snr_db / 10))
    scaling_factor = torch.sqrt(required_noise_power/noise_power)
    scaled_noise = scaling_factor * noise

    # Add scaled noise to wet source
    noisy_audio = wet_source.clone()
    noisy_audio[start_idx :end_idx] += scaled_noise[start_idx :end_idx]

    return noisy_audio 

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.unsafe_load(file)
            return data
        
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None



def get_audio_sources(mix_source_paths, mix_path, audiomix_metadata):
    """
    Args:
        mix_source_paths (list of str): Paths to source audio files (e.g., [path1, path3]).
        mix_path (str): Path to the mixed audio file.
        audiomix_metadata (str): Path to metadata YAML.
    """
    data = read_yaml_file(audiomix_metadata)
    wet_sources = []
    start_indices = []
    end_indices = []
    
    for i, source_path in enumerate(mix_source_paths):
        # Load source audio
        wet_source, sr = torchaudio.load(source_path)
        wet_source = wet_source[0, :]  # Take first channel
        wet_sources.append(wet_source)
        
        # Extract event metadata for the i-th source
        event = data['foreground_events'][i]
        onset = event['event_time']
        offset = onset + event['event_duration']
        
        # Calculate indices
        start_idx = int(onset * sr)
        end_idx = int(offset * sr)
        start_indices.append(start_idx)
        end_indices.append(end_idx)
    
    # Load mix audio
    mix_audio, sr_mix = torchaudio.load(mix_path)
    mix_audio = mix_audio[0, :]  # Take first channel
    
    return wet_sources, mix_audio, start_indices, end_indices


def create_labelling_errors(data, choice):
    
    est_lbl = []  # Initialize with empty list as default
    
    if choice == int(1):
        est_lbl = [data['foreground_events'][0]['label'], 
                  data['foreground_events'][1]['label'], 
                  data['foreground_events'][2]['label']]
    elif choice == int(2):  
        est_lbl = [data['foreground_events'][0]['label'], 
                  data['foreground_events'][1]['label']]
    elif choice == int(3): 
        est_lbl = [data['foreground_events'][0]['label']]
    elif choice == int(4): 
        est_lbl = [data['foreground_events'][0]['label'], 
                  data['foreground_events'][2]['label'], 
                  data['foreground_events'][1]['label']]
    elif choice == int(5):     
        est_lbl = [data['foreground_events'][2]['label'], 
                  data['foreground_events'][0]['label'], 
                  data['foreground_events'][1]['label']]
    elif choice == int(6):    
        est_lbl = ['cartoon', 
                  data['foreground_events'][1]['label'], 
                  data['foreground_events'][2]['label']]
    else:
        raise ValueError(f"Invalid choice: {choice}. Expected 1-6 as strings")

    return est_lbl


