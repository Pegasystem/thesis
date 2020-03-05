import shutil
import subprocess
from multiprocessing import Pool

import matplotlib.pyplot as plot
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import stft, istft
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from parameters import *
from thesis import AutoEncoder, AmplitudeDatasetDynamic, SingularAmplitudeDataset


# ============================================== I/O ==============================================


def read_file(filename):
    # Read the specified wav file and prints a warning if the sample rate is not fitting.
    # Returns a list of samples.
    rate, samples = wavfile.read(filename)
    if rate != SAMPLE_RATE:
        print("Sample rate of " + filename + " is not " + str(SAMPLE_RATE) + " kHz, results will not be consistent.")
    return samples


def write_file(stereo, filename):
    # Rounds the signal so that it can be properly written and writes it to disk with the appropriate sample rate.
    round_stereo = np.rint(stereo).astype(np.int16)
    wavfile.write(filename, SAMPLE_RATE, round_stereo)


def convert_files(filenames):
    pool = Pool(NUM_WORKERS)
    pool.map(stft_to_file, filenames)
    pool.close()


def stft_to_file(filename):
    file = read_file(filename)
    left, right = split_stereo(file)

    stft_left = np.transpose(stft(left, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])
    stft_right = np.transpose(stft(right, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])

    to_write = to_amplitude(np.concatenate((stft_left, stft_right)))

    np.savez(filename, to_write)
    print("Saved STFTs of " + filename + " to " + filename + ".npz.")


def stft_from_file(filename):
    return np.load(filename + ".npz")["arr_0"]


def read_file_lists():
    train_list = []
    with open(TRAIN_LIST) as to_read:
        for line in to_read:
            line = line[:-1]
            train_list.append(line)

    validation_list = []
    with open(VALIDATION_LIST) as to_read:
        for line in to_read:
            line = line[:-1]
            validation_list.append(line)

    test_list = []
    with open(TEST_LIST) as to_read:
        for line in to_read:
            line = line[:-1]
            test_list.append(line)

    return train_list, validation_list, test_list


def move_stuff(new_best=False):
    # Moves all the model's generated files to a directory of the model's parameters.
    new_directory = os.path.join(DIRECTORY, PARAMS)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    files = os.listdir(DIRECTORY)
    for file in files:
        if file.startswith("epoch"):
            shutil.move(file, new_directory)
        if file.endswith("output.npz"):
            shutil.move(file, new_directory)
        if file.endswith(".png"):
            shutil.move(file, new_directory)
        if new_best:
            if file == "best-model":
                shutil.copy(file, new_directory)


def batch_process(filenames, to_replace):
    tuples = []
    for filename in filenames:
        tuples.append((filename, filename.replace("orig", to_replace)))

    for pair in tuples:
        process_file(*pair)


def batch_peaq(filenames, to_replace):
    tuples = []
    for filename in filenames:
        tuples.append((filename, filename.replace("orig", to_replace)))
    pool = Pool(NUM_WORKERS)
    results = pool.starmap(peaq, tuples)
    pool.close()

    print("Writing results to file...")
    np.savez("peaq-" + to_replace + ".npz", results=results)


# ============================================== AUDIO STUFF ==============================================


def split_stereo(stereo_samples):
    # Splits an array of stereo audio into it's left channel and it's right channel.
    # Returns a tuple of the left channel and the right channel.
    left_channel = stereo_samples[:, 0]
    right_channel = stereo_samples[:, 1]

    return left_channel, right_channel


def merge_stereo(left_channel, right_channel):
    # Merges a given left and right channel back into stereo audio.
    # Returns a np array with stereo audio which can be written with scipy's wavfile library.

    return np.stack([left_channel, right_channel], axis=1)


def to_amplitude(stft_in):
    # Converts the given STFT data to the appropriate amplitudes.
    # Returns the same type of array, with amplitudes instead of complex numbers.

    return np.sqrt(np.imag(stft_in) ** 2 + np.real(stft_in) ** 2)


def to_phase(stft_in):
    # Converts the given STFT data to the appropriate phases.
    # Returns the same type of array, with phases instead of complex numbers.

    return np.arctan2(np.imag(stft_in), np.real(stft_in))


def to_signal(amplitudes, phases):
    # Merges the given set of amplitudes and phases back into an audio signal.
    # Returns an array of complex numbers ready to be fed to the iSTFT function.

    return np.transpose(amplitudes * np.exp(1j * phases))


def peaq(original, to_compare):
    cmd = ["peaq", "--advanced", original, to_compare]
    print("Comparing " + original + " to " + to_compare)
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=ENVIRONMENT).communicate()[0]
    output = output.decode("utf-8")
    output = output.split('\n')
    output = output[0].split()
    return float(output[3])


# ============================================== PLOTTING ==============================================


def plot_peaq(show):
    peaq_128 = np.load("peaq-128.npz")["results"]
    peaq_192 = np.load("peaq-192.npz")["results"]
    peaq_320 = np.load("peaq-320.npz")["results"]
    peaq_orig = np.load("peaq-orig-output.npz")["results"]
    peaq_128nn = np.load("peaq-128-output.npz")["results"]

    data = [peaq_128, peaq_192, peaq_320, peaq_orig, peaq_128nn]
    figure, axis = plot.subplots()
    axis.set_title("PEAQ results - " + PARAMS)
    axis.boxplot(data)
    plot.xticks([1, 2, 3, 4, 5], ["128kbps", "192kbps", "320kbps", "NN original", "NN 128kbps"])

    if show:
        plot.show()
    else:
        plot.savefig("peaq.png", dpi=500)


def plot_loss(filename, show):
    loss = torch.load(filename)
    train_loss = loss["train-loss"]
    val_loss = loss["val-loss"]
    epochs = list(range(0, loss["epoch"] + 1))

    plot.plot(epochs, train_loss)
    plot.plot(epochs, val_loss)

    if show:
        plot.show()
    else:
        plot.savefig("loss-" + str(loss["epoch"] + 1) + ".png", dpi=500)


# ============================================== AI ==============================================


def find_lr(training_files, logstart, logend, smooth=False):
    # noinspection PyGlobalUndefined
    global model
    # noinspection PyUnresolvedReferences
    model = AutoEncoder().cuda()

    train_dataset = AmplitudeDatasetDynamic(training_files)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=WEIGHT_DECAY)

    learning_rates = np.logspace(logstart, logend, num=len(train_dataloader))
    losses = []

    print("Finding best learning rate...")
    for lr, data in zip(learning_rates, train_dataloader):
        for parameter in optimizer.param_groups:
            parameter['lr'] = lr
        # noinspection PyArgumentList
        data = Variable(data).cuda()
        output = model(data.float())
        loss = criterion(output, data.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plot.xscale("log")
    if smooth:
        poly = np.polyfit(learning_rates, losses, 10)
        smooth = np.poly1d(poly)(learning_rates)
        plot.plot(learning_rates, smooth)
        plot.show()
    else:
        plot.plot(learning_rates, losses)
        plot.show()


def calculate(amplitudes):
    # Calculate the new amplitudes using the model - does not support CUDA.
    global model

    temp_dataset = SingularAmplitudeDataset(amplitudes)
    temp_dataloader = DataLoader(temp_dataset)

    model = AutoEncoder()
    # noinspection PyUnresolvedReferences
    model.load_state_dict(torch.load("best-model", map_location=torch.device("cpu"))["state-dict"])

    new_amps = []
    # noinspection PyUnresolvedReferences
    model.eval()
    with torch.no_grad():
        for temp_data in temp_dataloader:
            # noinspection PyArgumentList
            temp_data = Variable(temp_data)
            temp_output = model(temp_data.float())
            new_amps.append(temp_output.detach().numpy()[0])

    return np.array(new_amps)


def process_file(filename, write_to):
    # Runs a file through the neural network.
    print("Running file " + filename + " through the neural network.")
    file = read_file(filename)
    left, right = split_stereo(file)

    stft_left = np.transpose(stft(left, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])
    stft_right = np.transpose(stft(right, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])

    amp_left = to_amplitude(stft_left)
    amp_right = to_amplitude(stft_right)

    if SMALL:
        amp_left = np.log(amp_left, where=amp_left != 0)
        amp_right = np.log(amp_right, where=amp_right != 0)

    phase_left = to_phase(stft_left)
    phase_right = to_phase(stft_right)

    new_left = calculate(amp_left)
    new_right = calculate(amp_right)

    if SMALL:
        new_left = np.exp(new_left, where=new_left != 0)
        new_right = np.exp(new_right, where=new_right != 0)

    new_left = np.maximum(new_left, amp_left)
    new_right = np.maximum(new_right, amp_right)

    signal_left = to_signal(new_left, phase_left)
    signal_right = to_signal(new_right, phase_right)

    istft_left = istft(signal_left, fs=SAMPLE_RATE, nperseg=NPERSEG)[1]
    istft_right = istft(signal_right, fs=SAMPLE_RATE, nperseg=NPERSEG)[1]

    to_write = merge_stereo(istft_left, istft_right)
    write_file(to_write, write_to)
    print("File successfully processed, written to " + write_to + ".")
