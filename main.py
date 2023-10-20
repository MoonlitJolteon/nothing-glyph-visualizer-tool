import numpy as np
import argparse
import pandas as pd
from scipy.io import wavfile
from ffmpeg import FFmpeg, Progress

args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Takes a wav file and outputs a labels file that can be used with SebiAi's custom glyph lighting scripts alongside an ogg audio file.")
args_parser.add_argument("FILE", help="The wav file to be processed.")
args_parser.add_argument("--compatibility-mode", "-c", action="store_true", help="Use this flag if you are using phone 1. Otherwise, leave it out.")
args_parser.add_argument("--output", "-o", help="The output file name. If not specified, the output will match the name of the input.")

args = args_parser.parse_args()

# Parameters
segment_duration_seconds = 0.25
min_smooth_frames = 2
threshold_fraction = 0.5 # Adjust this value to control lights-on percentage, 0.5 = lights on when the frequency is above about half the maximum amplitude
phone_one_compatibility = False # Set to True if using phone 1, False if using phone 2

# Load audio data
sample_rate, audio_data = wavfile.read(args.FILE)

# Frequency ranges for analysis (in Hz)
# Define frequency ranges  and associated zone lights for analysis (in Hz)
phone1_frequency_ranges = [(20.0, 345.0), (345.0, 670.0), (670.0, 995.0), (995.0, 1320.0), (1320.0, 1645.0)]
phone1_zone_lights = [1, 2, 3, 4, 5]

phone2_frequency_ranges = [(20.0, 182.5), (182.5, 345.0), (345.0, 507.5), (507.5, 670.0), (670.0, 832.5), (832.5, 995.0), 
(995.0, 1157.5), (1157.5, 1320.0), (1320.0, 1482.5), (1482.5, 1645.0)]
phone2_zone_lights = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]

# phone2_frequency_ranges = [(20.0, 60.625), (60.625, 101.25), (101.25, 141.875), (141.875, 182.5), (182.5, 223.125), (223.125, 263.75), (263.75, 304.375), (304.375, 345.0), (345.0, 385.625), (385.625, 426.25), (426.25, 466.875), (466.875, 507.5), (507.5, 548.125), (548.125, 588.75), (588.75, 629.375), (629.375, 670.0), (670.0, 710.625), (710.625, 751.25), (751.25, 791.875), (791.875, 832.5), (832.5, 873.125), (873.125, 913.75), (913.75, 954.375), (954.375, 995.0), (995.0, 1035.625), (1035.625, 1076.25), (1076.25, 1116.875), (1116.875, 1157.5), (1157.5, 1198.125), (1198.125, 1238.75), (1238.75, 1279.375), (1279.375, 1320.0), (1320.0, 1360.625)]
# phone2_zone_lights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# Calculate segment duration in samples
segment_duration_samples = int(segment_duration_seconds * sample_rate)

# Compute FFT for the entire audio data
audio_spectrum = np.abs(np.fft.fft(audio_data))
audio_frequencies = np.fft.fftfreq(len(audio_data), 1 / sample_rate)

# Initialize thresholds for each frequency range
frequency_ranges = phone1_frequency_ranges if phone_one_compatibility else phone2_frequency_ranges
zone_lights = phone1_zone_lights if phone_one_compatibility else phone2_zone_lights
thresholds = []
for start_freq, end_freq in frequency_ranges:
    start_index = int(start_freq / sample_rate * len(audio_data))
    end_index = int(end_freq / sample_rate * len(audio_data))
    max_amplitude = np.max(audio_spectrum[start_index:end_index])
    thresholds.append(threshold_fraction * max_amplitude)


# Function to apply smoothing to a list of booleans
def smooth_bool_list(bool_list, min_frames):
    return np.convolve(bool_list, np.ones(min_frames), mode="same") >= min_frames

output = ""


# Analyze the audio in segments
num_segments = len(audio_data) // segment_duration_samples

# Map the audio waveform to the visualizer light IDs
audio_waveform = audio_data[:, 0]
normalized_waveform = audio_waveform / np.max(np.abs(audio_waveform))
start = 4
end = 19
width = end - start
mapped_amplitude = np.round((normalized_waveform - normalized_waveform.min()) / np.ptp(normalized_waveform) * width + start)

for segment_idx in range(num_segments):
    start_sample = segment_idx * segment_duration_samples
    end_sample = start_sample + segment_duration_samples
    segment = audio_data[start_sample:end_sample]
    segment_mapped_amplitude = mapped_amplitude[start_sample:end_sample]

    # Compute FFT for the segment
    segment_spectrum = np.abs(np.fft.fft(segment))
    segment_frequencies = np.fft.fftfreq(len(segment), 1 / sample_rate)

    # Initialize lights_status for this segment
    lights_status = [False] * len(frequency_ranges)

    # Analyze amplitude for each frequency range
    for idx, (start_freq, end_freq) in enumerate(frequency_ranges):
        indices = (segment_frequencies >= start_freq) & (segment_frequencies <= end_freq)
        amplitude_in_range = np.mean(segment_spectrum[indices])

        # Check if the amplitude exceeds the threshold
        lights_status[idx] = amplitude_in_range > thresholds[idx]

    # Apply smoothing to the lights_status
    lights_status = smooth_bool_list(lights_status, min_smooth_frames)

    print(segment_idx)
    for i in range(0, len(segment_mapped_amplitude), 600):
        for j in range(4, 20, 1):
            if segment_mapped_amplitude[i] > j:
                output += f"{segment_idx * segment_duration_seconds}\t{(segment_idx + 1) * segment_duration_seconds}\t#{j}-100-25\n"

    for i, light_status in enumerate(lights_status):
        if light_status:
            output += f"{segment_idx * segment_duration_seconds}\t{(segment_idx + 1) * segment_duration_seconds}\t{zone_lights[i]}-100-25\n"

# Write the output to a file
total_length = len(audio_data) / sample_rate
output += f"{total_length}\t{total_length}\tEND"

with open(f'{".".join(args.FILE.split(".")[:-1])}.txt', "w") as file:
    file.write(output)


# Cleanup the labels file by merging itentical labels that are directly connected to each other
df = pd.read_csv(f'{".".join(args.FILE.split(".")[:-1])}.txt', sep='\t', names=['start', 'end', 'value'])
merged_rows = []
df = df.sort_values(by=['value', 'start'])
current_row = df.iloc[0].copy() 
for i in range(1, len(df)):
    next_row = df.iloc[i]
    
    if current_row['value'] == next_row['value'] and current_row['end'] >= next_row['start']:
        current_row['end'] = max(current_row['end'], next_row['end'])
    else:
        merged_rows.append(current_row)
        current_row = next_row.copy()
merged_rows.append(current_row)
merged_df = pd.DataFrame(merged_rows, columns=['start', 'end', 'value'])
merged_df = merged_df.sort_values(by=['start'])
merged_df.to_csv(f'{".".join(args.FILE.split(".")[:-1])}.txt', sep='\t', header=False, index=False)


# Convert the wav file to opus formatted ogg
(
    FFmpeg()
        .input(args.FILE)
        .output(
            f'{".".join(args.FILE.split(".")[:-1])}.ogg',
            acodec='libopus'
        )
).execute()