from math import sqrt, atan2
from cmath import exp
import numpy
from scipy.io import wavfile
from scipy.signal import stft, istft
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 44100     # sample rate of all files, has to be the same
SEG_LENGTH = 1024       # amount of samples in a single segment

BATCH_SIZE = 256        # amount of pieces the DataLoader will put together
EPOCHS = 25
SHUFFLE = False         # don't order samples randomly
NUM_WORKERS = 4         # for multithreading
LEARNING_RATE = 0.001

FIRST_LAYER_SIZE = 512
SECOND_LAYER_SIZE = 256
THIRD_LAYER_SIZE = 128


def read_file(filename):
    # Read the specified wav file and prints a warning if the sample rate is not fitting.
    # Returns a list of samples.
    rate, samples = wavfile.read(filename)
    if rate != SAMPLE_RATE:
        print("Sample rate of " + filename + " is not " + str(SAMPLE_RATE) + " kHz, results will not be consistent.")
    return samples


def write_file(stereo, filename):
    # Rounds the signal so that it can be properly written and writes it to disk with the appropriate sample rate.
    round_stereo = numpy.rint(stereo).astype(numpy.int16)
    wavfile.write(filename, SAMPLE_RATE, round_stereo)


def split_stereo(stereo_samples):
    # Splits an array of stereo audio into it's left channel and it's right channel.
    # Returns a tuple of the left channel and the right channel.
    left_channel = stereo_samples[:, 0]
    right_channel = stereo_samples[:, 1]

    return left_channel, right_channel


def merge_stereo(left_channel, right_channel):
    # Merges a given left and right channel back into stereo audio.
    # Returns a numpy array with stereo audio which can be written with scipy's wavfile library.
    stereo = numpy.stack([left_channel, right_channel], axis=1)

    return stereo


def zero_pad(signal, modulo):
    # Pads a given signal so that it is divisible by the given modulo so that the input of the NN is the same dimension.
    # Returns the signal with the appropriate amount of zeros appended.
    n, remainder = divmod(len(signal), modulo)
    n += bool(remainder)
    new_signal = numpy.zeros(n * modulo)
    new_signal[:len(signal)] = signal

    return new_signal


def to_amplitude(stft_in):
    # Converts the given STFT data to the appropriate amplitudes.
    # Returns the same type of array, with amplitudes instead of complex numbers.
    amplitudes = []

    for window in stft_in:
        temp = []
        for sample in window:
            amplitude = sqrt(sample.real ** 2 + sample.imag ** 2)
            temp.append(amplitude)
        del temp[-1]        # to make sure it's of SEG_LENGTH
        amplitudes.append(temp)

    return numpy.array(amplitudes)


def to_phase(stft_in):
    # Converts the given STFT data to the appropriate phases.
    # Returns the same type of array, with phases instead of complex numbers.
    phases = []

    for window in stft_in:
        temp = []
        for sample in window:
            phase = atan2(sample.imag, sample.real)
            temp.append(phase)
        del temp[-1]  # to make sure it's of SEG_LENGTH
        phases.append(temp)

    return numpy.array(phases)


def to_signal(amplitudes, phases):
    # Merges the given set of amplitudes and phases back into an audio signal.
    # Returns an array of complex numbers ready to be fed to the iSTFT function.
    signal = []

    for window_amp, window_phase in zip(amplitudes, phases):
        temp = []
        for sample_amp, sample_phase in zip(window_amp, window_phase):
            temp.append(sample_amp * exp(1j * sample_phase))
        signal.append(temp)

    return numpy.array(signal)


def calculate(amplitudes):
    # Calculate the new amplitudes using the model.
    temp_dataset = SingularAmplitudeDataset(amplitudes)
    temp_dataloader = DataLoader(temp_dataset, batch_size=1, shuffle=SHUFFLE, num_workers=NUM_WORKERS)

    new_amps = []
    for temp_data in temp_dataloader:
        temp_data = Variable(temp_data).cuda()
        temp_output = model(temp_data.float())
        new_amps.append(temp_output.cpu().detach().numpy()[0])

    return numpy.array(new_amps)


def process_file(filename):
    # Runs a file through the neural network.
    print("\nRunning file " + filename + " through the neural network.")
    file = read_file(filename)
    left, right = split_stereo(file)

    left = zero_pad(left, SEG_LENGTH)
    right = zero_pad(right, SEG_LENGTH)

    nperseg = int(len(left) / SEG_LENGTH * 2)  # to make sure all segments are of SEG_LENGTH

    stft_left = stft(left, fs=SAMPLE_RATE, nperseg=nperseg)[2]
    stft_right = stft(right, fs=SAMPLE_RATE, nperseg=nperseg)[2]

    amp_left = to_amplitude(stft_left)
    amp_right = to_amplitude(stft_right)

    phase_left = to_phase(stft_left)
    phase_right = to_phase(stft_right)

    new_left = calculate(amp_left)
    new_right = calculate(amp_right)

    signal_left = to_signal(new_left, phase_left)
    signal_right = to_signal(new_right, phase_right)

    istft_left = istft(signal_left, fs=SAMPLE_RATE, nperseg=nperseg)[1]
    istft_right = istft(signal_right, fs=SAMPLE_RATE, nperseg=nperseg)[1]

    to_write = merge_stereo(istft_left, istft_right)
    write_file(to_write, "output.wav")
    print("File successfully processed, written to output.wav.")


class AmplitudeDataset(Dataset):
    def __init__(self, filenames):
        # filenames = array with filenames
        # self.amps = array with amplitudes
        # Reads all files and processes them.
        temp = []

        for filename in filenames:
            print("Reading file " + filename + "...")
            file = read_file(filename)
            left, right = split_stereo(file)

            left = zero_pad(left, SEG_LENGTH)
            right = zero_pad(right, SEG_LENGTH)

            nperseg = int(len(left) / SEG_LENGTH * 2)

            stft_left = stft(left, fs=SAMPLE_RATE, nperseg=nperseg)[2]
            stft_right = stft(right, fs=SAMPLE_RATE, nperseg=nperseg)[2]

            temp.append(to_amplitude(stft_left))
            temp.append(to_amplitude(stft_right))
        self.amps = numpy.concatenate(temp)
        print("\nFinished reading all files.\n")

    def __len__(self):
        return len(self.amps)

    def __getitem__(self, idx):
        return self.amps[idx]


class SingularAmplitudeDataset(Dataset):
    def __init__(self, amplitudes):
        self.amps = amplitudes

    def __len__(self):
        return len(self.amps)

    def __getitem__(self, idx):
        return self.amps[idx]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(SEG_LENGTH, FIRST_LAYER_SIZE),
            # nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE),
            # nn.Linear(SECOND_LAYER_SIZE, THIRD_LAYER_SIZE)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(THIRD_LAYER_SIZE, SECOND_LAYER_SIZE),
            # nn.Linear(SECOND_LAYER_SIZE, FIRST_LAYER_SIZE),
            nn.Linear(FIRST_LAYER_SIZE, SEG_LENGTH)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


file_list = ["oneandonly.wav", "doot.wav"]

# NN setup
dataset = AmplitudeDataset(file_list)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# NN training
for epoch in range(EPOCHS):
    for data in dataloader:
        data = Variable(data).cuda()

        output = model(data.float())
        loss = criterion(output, data.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, loss.data.item()))

process_file("oneandonly.wav")
