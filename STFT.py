import datetime
from multiprocessing import Manager, Pool
import numpy
from scipy.io import wavfile
from scipy.signal import stft, istft
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# File lists should contain a filename (or path to a file) on every line.
TRAIN_LIST = "files.txt"
VALIDATION_LIST = "validation.txt"

SAMPLE_RATE = 44100     # sample rate of all files, has to be the same
NPERSEG = 2048          # amount of samples in a single segment

BATCH_SIZE = 256        # amount of pieces the DataLoader will put together
EPOCHS = 25
NUM_WORKERS = 8         # for multithreading/multiprocessing
LEARNING_RATE = 0.0002

INPUT_LAYER_SIZE = int(NPERSEG / 2 + 1)     # because the output of the STFT is of length NPERSEG / 2 + 1
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

    return numpy.sqrt(numpy.imag(stft_in) ** 2 + numpy.real(stft_in) ** 2)


def to_phase(stft_in):
    # Converts the given STFT data to the appropriate phases.
    # Returns the same type of array, with phases instead of complex numbers.

    return numpy.arctan2(numpy.imag(stft_in), numpy.real(stft_in))


def to_signal(amplitudes, phases):
    # Merges the given set of amplitudes and phases back into an audio signal.
    # Returns an array of complex numbers ready to be fed to the iSTFT function.

    return numpy.transpose(amplitudes * numpy.exp(1j * phases))


def calculate(amplitudes):
    # Calculate the new amplitudes using the model.
    temp_dataset = SingularAmplitudeDataset(amplitudes)
    temp_dataloader = DataLoader(temp_dataset, num_workers=NUM_WORKERS)

    new_amps = []
    model.eval()
    with torch.no_grad():
        for temp_data in temp_dataloader:
            temp_data = Variable(temp_data).cuda()
            temp_output = model(temp_data.float())
            new_amps.append(temp_output.cpu().detach().numpy()[0])

    return numpy.array(new_amps)


def convert_files(filenames):
    pool = Pool(NUM_WORKERS)
    pool.map(stft_to_file, filenames)
    pool.close()


def stft_to_file(filename):
    file = read_file(filename)
    left, right = split_stereo(file)

    stft_left = numpy.transpose(stft(left, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])
    stft_right = numpy.transpose(stft(right, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])

    to_write = to_amplitude(numpy.concatenate((stft_left, stft_right)))

    numpy.savez(filename, to_write, stft=to_write)
    print("Saved STFTs of " + filename + " to " + filename + ".npz.")


def stft_from_file(filename):

    return numpy.load(filename + ".npz")["stft"]


def process_file(filename):
    # Runs a file through the neural network.
    print("\nRunning file " + filename + " through the neural network.")
    file = read_file(filename)
    left, right = split_stereo(file)

    stft_left = numpy.transpose(stft(left, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])
    stft_right = numpy.transpose(stft(right, fs=SAMPLE_RATE, nperseg=NPERSEG)[2])

    amp_left = to_amplitude(stft_left)
    amp_right = to_amplitude(stft_right)

    phase_left = to_phase(stft_left)
    phase_right = to_phase(stft_right)

    new_left = calculate(amp_left)
    new_right = calculate(amp_right)

    signal_left = to_signal(new_left, phase_left)
    signal_right = to_signal(new_right, phase_right)

    istft_left = istft(signal_left, fs=SAMPLE_RATE, nperseg=NPERSEG)[1]
    istft_right = istft(signal_right, fs=SAMPLE_RATE, nperseg=NPERSEG)[1]

    to_write = merge_stereo(istft_left, istft_right)
    write_file(to_write, "output.wav")
    print("File successfully processed, written to output.wav.")


def train(training_files, validation_files):
    train_dataset = AmplitudeDatasetDynamic(training_files)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_dataset = AmplitudeDatasetDynamic(validation_files)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print("Starting training...")
    total_time = 0
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        for data in train_dataloader:
            data = Variable(data).cuda()
            output = model(data.float())
            train_loss = criterion(output, data.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for data in validation_dataloader:
                data = Variable(data).cuda()
                output = model(data.float())
                validation_loss = criterion(output, data.float())
        end = time.time() - start
        total_time += end
        estimate = str(datetime.timedelta(seconds=round((total_time / (epoch + 1)) * (EPOCHS - epoch + 1))))
        print("epoch [{}/{}], loss: {:.4f}, val. loss: {:.4f}, time: {:.2f}s, est. time: {}"
              .format(epoch + 1, EPOCHS, train_loss.item(), validation_loss.item(), end, estimate))


class AmplitudeDatasetDynamic(Dataset):
    def calculate_length(self, filename):
        self.temp.append(len(stft_from_file(filename)))

    def get_stft(self, filename):
        self.temp.append(stft_from_file(filename))

    def __init__(self, filenames):
        # filenames = array with filenames
        # self.length = dataset length
        # self.filenames = array with list of filenames to be read
        # self.amps = array that contains the currently read file
        # Calculates dataset length.
        print("Calculating dataset length - this might take a while...")
        manager = Manager()
        self.temp = manager.list()
        self.filenames = filenames
        self.filenames_temp = self.filenames.copy()
        self.amps = []
        self.stft = []

        pool = Pool(NUM_WORKERS)
        pool.map(self.calculate_length, filenames)
        pool.close()

        self.length = sum(self.temp)
        print("Finished calculating dataset length, amount of STFTs: " + str(self.length) + ".\n")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if len(self.amps) == 0:
            # Read the next file.
            if len(self.filenames_temp) == 0:
                # Start over from the beginning.
                self.filenames_temp.extend(self.filenames)
            if len(self.stft) == 0:
                manager = Manager()
                self.temp = manager.list()
                temp = []
                while len(temp) != NUM_WORKERS and len(self.filenames_temp) != 0:
                    temp.append(self.filenames_temp.pop())

                pool = Pool(NUM_WORKERS)
                pool.map(self.get_stft, temp)
                pool.close()

                self.stft = self.temp[:]
                current_stft = self.stft.pop()
            else:
                current_stft = self.stft.pop()

            self.amps.extend(current_stft)
            self.amps_reversed = self.amps[::-1]
        # Returns a sample from the currently read file.
        self.amps.pop()
        return self.amps_reversed[len(self.amps_reversed) - len(self.amps) - 1]


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
            nn.Linear(INPUT_LAYER_SIZE, FIRST_LAYER_SIZE),
            # nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE),
            # nn.Linear(SECOND_LAYER_SIZE, THIRD_LAYER_SIZE)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(THIRD_LAYER_SIZE, SECOND_LAYER_SIZE),
            # nn.Linear(SECOND_LAYER_SIZE, FIRST_LAYER_SIZE),
            nn.Linear(FIRST_LAYER_SIZE, INPUT_LAYER_SIZE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

model = AutoEncoder().cuda()
train(train_list, validation_list)

# model.load_state_dict(torch.load("model-120"))

torch.save(model.state_dict(), "model-120-bs256-lr2e-4-25epochs")

process_file("candybits_192.wav")

# convert_files(validation_list)
