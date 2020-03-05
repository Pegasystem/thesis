import os

DATASET = "83"

# File lists should contain a filename (or path to a file) on every line.
TRAIN_LIST = "files.txt." + str(DATASET)
VALIDATION_LIST = "validation.txt." + str(DATASET)
TEST_LIST = "test.txt." + str(DATASET)

SAMPLE_RATE = 44100  # sample rate of all files, has to be the same
NPERSEG = 2048  # amount of samples in a single segment - requires converting files again if changed

model = None

BATCH_SIZE = 512
EPOCHS = 25
NUM_WORKERS = 8  # amount of available CPU threads
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

SMALL = False  # runs all data through log function & exp as reverse to make it smaller

INPUT_LAYER_SIZE = int(NPERSEG / 2 + 1)  # because the output of the STFT is of length NPERSEG / 2 + 1
FIRST_LAYER_SIZE = int(INPUT_LAYER_SIZE * 0.75)
SECOND_LAYER_SIZE = int(INPUT_LAYER_SIZE * 0.5)
THIRD_LAYER_SIZE = 128

PARAMS = str(DATASET) + "-bs" + str(BATCH_SIZE) + "-lr" + str(LEARNING_RATE) + "-epochs" + str(EPOCHS) + \
         "-nperseg" + str(NPERSEG) + "-firstlayer" + str(FIRST_LAYER_SIZE) + "-secondlayer" + str(SECOND_LAYER_SIZE)

ENVIRONMENT = os.environ.copy()
ENVIRONMENT["GST_PLUGIN_PATH"] = "/usr/local/lib/gstreamer-1.0"

DIRECTORY = os.getcwd()
