import datetime
import matplotlib.pyplot as plot
from multiprocessing import Manager, Pool
import numpy
import os
from scipy.io import wavfile
from scipy.signal import stft, istft
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import subprocess

DATASET = "83"

# File lists should contain a filename (or path to a file) on every line.
TRAIN_LIST = "files.txt." + str(DATASET)
VALIDATION_LIST = "validation.txt." + str(DATASET)
TEST_LIST = "test.txt." + str(DATASET)

SAMPLE_RATE = 44100     # sample rate of all files, has to be the same
NPERSEG = 2048          # amount of samples in a single segment - requires converting files again if changed

model = None

BATCH_SIZE = 1024
EPOCHS = 50
NUM_WORKERS = 8         # amount of available CPU threads
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5

INPUT_LAYER_SIZE = int(NPERSEG / 2 + 1)     # because the output of the STFT is of length NPERSEG / 2 + 1
FIRST_LAYER_SIZE = int(INPUT_LAYER_SIZE / 2)
SECOND_LAYER_SIZE = 256
THIRD_LAYER_SIZE = 128

PARAMS = str(DATASET) + "-bs" + str(BATCH_SIZE) + "-lr" + str(LEARNING_RATE) + "-epochs" + str(EPOCHS) + \
         "-nperseg" + str(NPERSEG)

ENVIRONMENT = os.environ.copy()
ENVIRONMENT["GST_PLUGIN_PATH"] = "/usr/local/lib/gstreamer-1.0"


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
    # Calculate the new amplitudes using the model - does not support CUDA.
    temp_dataset = SingularAmplitudeDataset(amplitudes)
    temp_dataloader = DataLoader(temp_dataset)

    new_amps = []
    model.eval()
    with torch.no_grad():
        for temp_data in temp_dataloader:
            temp_data = Variable(temp_data)
            temp_output = model(temp_data.float())
            new_amps.append(temp_output.detach().numpy()[0])

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

    numpy.savez(filename, to_write)
    print("Saved STFTs of " + filename + " to " + filename + ".npz.")


def stft_from_file(filename):

    return numpy.load(filename + ".npz")["arr_0"]


def batch_peaq(filenames, to_replace):
    tuples = []
    for filename in filenames:
        tuples.append((filename, filename.replace("orig", to_replace)))
    pool = Pool(NUM_WORKERS)
    results = pool.starmap(peaq, tuples)
    pool.close()

    print("Writing results to file...")
    numpy.savez("peaq-" + to_replace + ".npz", results=results)


def peaq(original, to_compare):
    cmd = ["peaq", "--advanced", original, to_compare]
    print("Comparing " + original + " to " + to_compare)
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=ENVIRONMENT).communicate()[0]
    output = output.decode("utf-8")
    output = output.split('\n')
    output = output[0].split()
    return float(output[3])


def batch_process(filenames, to_replace):
    tuples = []
    for filename in filenames:
        tuples.append((filename, filename.replace("orig", to_replace)))
    pool = Pool(NUM_WORKERS)
    pool.starmap(process_file, tuples)
    pool.close()


def process_file(filename, write_to):
    # Runs a file through the neural network.
    print("Running file " + filename + " through the neural network.")
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
    write_file(to_write, write_to)
    print("File successfully processed, written to " + write_to + ".")


def plot_peaq(show):
    peaq_128 = numpy.load("peaq-128.npz")["results"]
    peaq_192 = numpy.load("peaq-192.npz")["results"]
    peaq_320 = numpy.load("peaq-320.npz")["results"]
    peaq_orig = numpy.load("peaq-orig-output.npz")["results"]
    peaq_128nn = numpy.load("peaq-128-output.npz")["results"]

    data = [peaq_128, peaq_192, peaq_320, peaq_orig, peaq_128nn]
    figure, axis = plot.subplots()
    axis.set_title("PEAQ results - " + PARAMS)
    axis.boxplot(data)
    plot.xticks([1, 2, 3, 4, 5], ["128kbps", "192kbps", "320kbps", "NN original", "NN 128kbps"])

    if show:
        plot.show()
    else:
        plot.savefig("peaq-" + PARAMS + ".png")


def plot_loss(filename, epochs, show):
    loss = numpy.load(filename)
    train_loss = loss["train_loss"]
    val_loss = loss["val_loss"]
    epochs = list(range(1, epochs + 1))

    plot.plot(epochs, train_loss)
    plot.plot(epochs, val_loss)

    if show:
        plot.show()
    else:
        plot.savefig("loss-" + PARAMS + ".png")


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


def train_model(train_list, validation_list):
    global model

    model = AutoEncoder().cuda()
    train(train_list, validation_list)
    torch.save(model.state_dict(), "model-" + PARAMS)
    plot_loss("loss-" + PARAMS + ".npz", EPOCHS, False)


def test_model(test_list):
    global model

    device = torch.device("cpu")
    model = AutoEncoder()
    model.load_state_dict(torch.load("model-" + PARAMS, map_location=device))

    batch_process(test_list, "orig-output")
    batch_process(test_list, "128-output")
    batch_peaq(test_list, "orig-output")
    batch_peaq(test_list, "128-output")
    plot_peaq(False)


def train(training_files, validation_files):
    train_dataset = AmplitudeDatasetDynamic(training_files)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_dataset = AmplitudeDatasetDynamic(validation_files)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Starting training...")
    total_time = 0
    epoch_counter = 0
    train_losses = []
    validation_losses = []
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        train_loss = 0
        validation_loss = 0
        for data in train_dataloader:
            data = Variable(data).cuda()
            output = model(data.float())
            loss = criterion(output, data.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            for data in validation_dataloader:
                data = Variable(data).cuda()
                output = model(data.float())
                loss = criterion(output, data.float())

                validation_loss += loss.item()
        end = time.time() - start
        total_time += end
        train_loss = train_loss / len(train_dataloader)
        validation_loss = validation_loss / len(validation_dataloader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        estimate = str(datetime.timedelta(seconds=round((total_time / (epoch + 1)) * (EPOCHS - epoch + 1))))
        print("epoch [{}/{}], loss: {:.4f}, val. loss: {:.4f}, time: {:.2f}s, est. time: {}"
              .format(epoch + 1, EPOCHS, train_loss, validation_loss, end, estimate))
        epoch_counter += 1
        if epoch_counter % 25 == 0:
            epoch_counter = 0
            torch.save(model.state_dict(), "epoch" + str(epoch + 1) + "-model")
    print("Writing losses to file...\n")
    numpy.savez("loss-" + PARAMS, train_loss=train_losses, val_loss=validation_losses)


class AmplitudeDatasetDynamic(Dataset):
    def calculate_length(self, filename):
        self.temp.append(len(stft_from_file(filename)))

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

            self.amps.extend(stft_from_file(self.filenames_temp.pop()))
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
            nn.Linear(INPUT_LAYER_SIZE, FIRST_LAYER_SIZE), nn.ReLU(True)
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


print("Parameters: " + PARAMS)
training, validation, test = read_file_lists()
train_model(training, validation)
test_model(test)
