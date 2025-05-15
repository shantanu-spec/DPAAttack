import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Function to calculate the Hamming weight
def hamming_weight(n):
    """Return the Hamming weight of an integer n (number of set bits)."""
    return bin(n).count("1")

# Function to plot traces
def plot_traces(traces, title, save_as):
    plt.figure(figsize=(10, 6))
    for i in range(min(10, traces.shape[0])):  # Plot first 10 traces
        plt.plot(traces[i, :], label=f'Trace {i + 1}')
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.savefig(save_as)
    plt.close()

# Start the timer
start_time = time.time()

# Load the power traces data
traces = pd.read_csv('traces.csv', header=None).values.astype(float)  # Load and convert data to float
plaintext = pd.read_csv('input_plaintext.txt', header=None, delim_whitespace=True).values
ciphertext = pd.read_csv('output_ciphertext.txt', header=None, delim_whitespace=True).values

# Step 1: Align the power traces (baseline adjustment)
num_traces = traces.shape[0]
num_samples = traces.shape[1]

# Calculate the baseline for each trace using the first N samples
baselines_all_traces = np.mean(traces[:, :num_samples], axis=1)

# Shift all traces by subtracting their respective baselines
mean_baseline_all_traces = np.mean(baselines_all_traces)
for i in range(traces.shape[0]):
    shift_amount = baselines_all_traces[i] - mean_baseline_all_traces
    traces[i, :] -= shift_amount

# Step 1: Plot the first power trace
plot_traces(traces, 'First Ten Power Traces After Alignment', 'first_trace.png')

# Initialize key bytes list
key_bytes = []
threshold = 0.05  # Define a threshold for correlation difference

# Loop through all 16 bytes
for byte_index in range(16):
    key_guesses = np.zeros(256)
    correlations = np.zeros((256, num_samples))

    # Pre-compute hypothetical power for all key guesses for current byte
    hypothetical_power = np.zeros((256, num_traces))

    for key_guess in range(256):
        # Calculate hypothetical power for all traces at once
        hypothetical_power[key_guess, :] = np.array([
            hamming_weight(int(plaintext[t, 0][byte_index*2:byte_index*2+2], 16) ^ key_guess)
            for t in range(num_traces)
        ])

        # Calculate correlation for all time samples
        for time_sample in range(num_samples):
            actual_power = traces[:, time_sample]
            if np.std(hypothetical_power[key_guess]) == 0 or np.std(actual_power) == 0:
                corr = 0
            else:
                corr = np.corrcoef(hypothetical_power[key_guess], actual_power)[0, 1]
            correlations[key_guess, time_sample] = 0 if np.isnan(corr) else corr

    # Determine the best and second-best guesses
    max_correlation = np.max(np.abs(correlations), axis=1)
    best_two_guesses = np.argsort(max_correlation)[-2:]  # Best two guesses
    
    # Choose best guess only if it's significantly better than the second best guess
    if max_correlation[best_two_guesses[0]] - max_correlation[best_two_guesses[1]] > threshold:
        key_bytes.append(best_two_guesses[0])  # Use best guess
    else:
        key_bytes.append(best_two_guesses[0])  # For now, default to best guess if difference is small

    # Plotting the two best guesses for each byte
    plt.figure(figsize=(10, 6))
    plt.plot(correlations[best_two_guesses[0], :], label=f'Best Guess: {best_two_guesses[0]}')
    plt.plot(correlations[best_two_guesses[1], :], label=f'Second Best Guess: {best_two_guesses[1]}')
    plt.title(f'Correlation for Two Best Key Guesses (Byte {byte_index})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.savefig(f'two_best_guesses_byte_{byte_index}.png')
    plt.close()

# Print the guessed key bytes in hexadecimal
guessed_key = ''.join(f'{key_byte:02x}' for key_byte in key_bytes)
print(f"The 128-bit AES key is: {guessed_key}")

# Step 4: Optimize Evolution Computation
best_time_sample = np.argmax(np.max(np.abs(correlations), axis=0))
key_evolution = np.zeros((256, num_traces))

# Optimize by limiting trace evolution for visualization
trace_limit = min(1000, num_traces)

# Loop through the first N traces (reduce computational load)
for trace_count in range(2, trace_limit):
    actual_power = traces[:trace_count, best_time_sample]  # Use only the best time sample

    for key_guess in range(256):
        if np.std(hypothetical_power[key_guess, :trace_count]) == 0 or np.std(actual_power) == 0:
            corr = 0
        else:
            corr = np.corrcoef(hypothetical_power[key_guess, :trace_count], actual_power)[0, 1]
        key_evolution[key_guess, trace_count - 1] = 0 if np.isnan(corr) else corr

# Plot key evolution
plt.figure(figsize=(12, 8))
for key_guess in range(256):
    plt.plot(key_evolution[key_guess, :], alpha=0.6)
plt.title(f'Key Evolution Over Time at Time Sample {best_time_sample}')
plt.xlabel('Number of Measurements [0-{trace_limit}]')
plt.ylabel('Correlation Coefficient[-1:1]')
plt.grid(True)
plt.savefig('key_evolution.png')

# Step 5: Calculate Mean Time to Disclosure (MTD)
stabilization_points = []
for byte_index in range(16):
    max_time_sample = np.argmax(np.max(np.abs(correlations), axis=0))
    best_key_guess = np.argmax(np.abs(correlations[:, max_time_sample]))
    for t in range(num_traces):
        if np.argmax(np.abs(correlations[:, max_time_sample][:t + 1])) == best_key_guess:
            stabilization_points.append(t)
            break
    else:
        stabilization_points.append(num_traces)

mean_time_to_disclosure = np.mean(stabilization_points)
print(f"Mean Time to Disclosure: {mean_time_to_disclosure}")

# Step 6: End the timer and print the time taken
end_time = time.time()
print(f"Time taken for all operations: {end_time - start_time} seconds")