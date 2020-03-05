import datetime
import time

from torch.utils.data import Dataset

from helpers import *
from parameters import *


def train_model(train_list, validation_list):
    # noinspection PyGlobalUndefined
    global model

    # noinspection PyUnresolvedReferences
    model = AutoEncoder().cuda()
    train(train_list, validation_list)
    plot_loss("epoch" + str(EPOCHS) + "-model", False)


def test_model(test_list):
    global model

    best_model = torch.load("best-model")

    model = AutoEncoder()
    for key, value in best_model["state-dict"]:
        best_model["state-dict"][key] = value.cpu()
    # noinspection PyUnresolvedReferences
    model.load_state_dict(best_model["state-dict"])

    batch_process(test_list, "orig-output")
    batch_process(test_list, "128-output")
    batch_peaq(test_list, "orig-output")
    batch_peaq(test_list, "128-output")
    move_stuff()
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
    best = {"loss": 1e1000}

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        train_loss = 0
        validation_loss = 0
        for data in train_dataloader:
            # noinspection PyArgumentList
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
                # noinspection PyArgumentList
                data = Variable(data).cuda()
                output = model(data.float())
                loss = criterion(output, data.float())

                validation_loss += loss.item()

        end = time.time() - start
        total_time += end
        estimate = str(datetime.timedelta(seconds=round((total_time / (epoch + 1)) * (EPOCHS - epoch + 1))))

        train_loss = train_loss / len(train_dataloader)
        validation_loss = validation_loss / len(validation_dataloader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print("epoch [{}/{}], loss: {:.8f}, val. loss: {:.8f}, time: {:.2f}s, est. time: {}"
              .format(epoch + 1, EPOCHS, train_loss, validation_loss, end, estimate))

        if best["loss"] > validation_loss:
            best = {"loss": validation_loss, "epochs": epoch, "state_dict": model.state_dict()}
            state = {"epoch": epoch, "state-dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "train-loss": train_losses, "val-loss": validation_losses, "best": best}
            torch.save(state, "best-model")

        epoch_counter += 1
        if epoch_counter % 10 == 0:
            epoch_counter = 0
            state = {"epoch": epoch, "state-dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "train-loss": train_losses, "val-loss": validation_losses, "best": best}
            torch.save(state, "epoch" + str(epoch + 1) + "-model")

        move_stuff()


class AmplitudeDatasetDynamic(Dataset):

    def __init__(self, filenames):
        # filenames = array with filenames
        # self.length = dataset length
        # self.filenames = array with list of filenames to be read
        # self.amps = array that contains the currently read file
        # Calculates dataset length.
        print("Calculating dataset length - this might take a while...")
        self.filenames = filenames
        self.filenames_temp = self.filenames[:]
        self.amps = []
        self.length = 0

        for filename in filenames:
            current_file = stft_from_file(filename)
            self.length += len(current_file)

        print("Finished calculating dataset length, amount of STFTs: " + str(self.length) + ".\n")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if len(self.amps) == 0:
            # Read the next file.
            if len(self.filenames_temp) == 0:
                # Start over from the beginning.
                self.filenames_temp.extend(self.filenames)

            current_file = stft_from_file(self.filenames_temp.pop())
            if SMALL:
                self.amps = np.log(current_file, where=current_file != 0)
            else:
                self.amps = current_file
            self.amps_reversed = self.amps[::-1]
        # Returns a sample from the currently read file.
        self.amps = self.amps[:-1]
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
            nn.Linear(INPUT_LAYER_SIZE, FIRST_LAYER_SIZE), nn.ReLU(True),
            nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE), nn.ReLU(True)
            # nn.Linear(SECOND_LAYER_SIZE, THIRD_LAYER_SIZE)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(THIRD_LAYER_SIZE, SECOND_LAYER_SIZE),
            nn.Linear(SECOND_LAYER_SIZE, FIRST_LAYER_SIZE), nn.ReLU(True),
            nn.Linear(FIRST_LAYER_SIZE, INPUT_LAYER_SIZE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    print("Parameters: " + PARAMS)
    training, validation, test = read_file_lists()
    train_model(training, validation)
    test_model(test)

    # find_lr(training, -7, -2)
    # find_lr(training, -7, -2, True)

    # convert_files(training)
    # convert_files(validation)

    # plot_loss("loss-" + PARAMS + ".npz", EPOCHS, True)


if __name__ == '__main__':
    main()
