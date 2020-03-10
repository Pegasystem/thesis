import datetime
import time

from torch.utils.data import Dataset

from helpers import *
from parameters import *


def find_lr(training_input, training_output, logstart, logend, smooth=False):
    # noinspection PyGlobalUndefined
    global model
    # noinspection PyUnresolvedReferences
    model = AutoEncoder().cuda()

    training_input_dataset = AmplitudeDatasetDynamic(training_input)
    training_input_dataloader = DataLoader(training_input_dataset, batch_size=BATCH_SIZE)
    training_output_dataset = AmplitudeDatasetDynamic(training_output)
    training_output_dataloader = DataLoader(training_output_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=WEIGHT_DECAY)

    learning_rates = np.logspace(logstart, logend, num=len(training_output_dataloader))
    losses = []

    print("Finding best learning rate...")
    for lr, inp, outp in zip(learning_rates, training_input_dataloader, training_output_dataloader):
        for parameter in optimizer.param_groups:
            parameter['lr'] = lr
        # noinspection PyArgumentList
        inp = Variable(inp).cuda()
        # noinspection PyArgumentList
        outp = Variable(outp).cuda()

        model_output = model(inp.float())
        loss = criterion(model_output, outp.float())

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


def train_model(train_input, train_output, validation_input, validation_output):
    # noinspection PyGlobalUndefined
    global model

    # noinspection PyUnresolvedReferences
    model = AutoEncoder().cuda()
    train(train_input, train_output, validation_input, validation_output)
    plot_loss(os.path.join(DIRECTORY, PARAMS, "epoch" + str(EPOCHS) + "-model"), False)


def test_model(test_orig, test_128):
    batch_process(test_orig, "orig-output")
    batch_process(test_128, "128-output")
    batch_peaq(test_orig, "orig-output")
    batch_peaq(test_orig, "128-output")
    plot_peaq(False)
    move_stuff()


def train(training_input, training_output, validation_input, validation_output, checkpoint=False):
    training_input_dataset = AmplitudeDatasetDynamic(training_input)
    training_input_dataloader = DataLoader(training_input_dataset, batch_size=BATCH_SIZE)
    training_output_dataset = AmplitudeDatasetDynamic(training_output)
    training_output_dataloader = DataLoader(training_output_dataset, batch_size=BATCH_SIZE)
    validation_input_dataset = AmplitudeDatasetDynamic(validation_input)
    validation_input_dataloader = DataLoader(validation_input_dataset, batch_size=BATCH_SIZE)
    validation_output_dataset = AmplitudeDatasetDynamic(validation_output)
    validation_output_dataloader = DataLoader(validation_output_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Starting training...")

    total_time = 0
    epoch_counter = 0
    start = 0
    train_losses = []
    validation_losses = []
    best = {"loss": 1e1000}

    if checkpoint:
        to_load = os.path.join(DIRECTORY, PARAMS, "best-model")
        checkpoint_state = torch.load(to_load)
        start = checkpoint_state["epoch"]
        print("Starting from checkpoint at epoch " + str(start) + "...")
        model.load_state_dict(checkpoint_state["state-dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        train_losses = checkpoint_state["train-loss"]
        validation_losses = checkpoint_state["val-loss"]
        best = checkpoint_state["best"]
        print("Checkpoint successfully loaded.\n")

    for epoch in range(start=start, stop=EPOCHS):
        start = time.time()
        train_loss = 0
        validation_loss = 0
        new_best = False

        model.train()
        for inp, outp in zip(training_input_dataloader, training_output_dataloader):
            # noinspection PyArgumentList
            inp = Variable(inp).cuda()
            # noinspection PyArgumentList
            outp = Variable(outp).cuda()

            model_output = model(inp.float())
            loss = criterion(model_output, outp.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for inp, outp in zip(validation_input_dataloader, validation_output_dataloader):
                # noinspection PyArgumentList
                inp = Variable(inp).cuda()
                # noinspection PyArgumentList
                outp = Variable(outp).cuda()

                model_output = model(inp.float())
                loss = criterion(model_output, outp.float())

                validation_loss += loss.item()

        end = time.time() - start
        total_time += end
        estimate = str(datetime.timedelta(seconds=round((total_time / (epoch + 1)) * (EPOCHS - epoch + 1))))

        train_loss = train_loss / len(training_output_dataloader)
        validation_loss = validation_loss / len(validation_output_dataloader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print("epoch [{}/{}], loss: {:.8f}, val. loss: {:.8f}, time: {:.2f}s, est. time: {}"
              .format(epoch + 1, EPOCHS, train_loss, validation_loss, end, estimate))

        if best["loss"] > validation_loss:
            best = {"loss": validation_loss, "epochs": epoch, "state_dict": model.state_dict()}
            state = {"epoch": epoch, "state-dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "train-loss": train_losses, "val-loss": validation_losses, "best": best}
            torch.save(state, "best-model")
            new_best = True

        epoch_counter += 1
        if epoch_counter % 25 == 0 or epoch + 1 == EPOCHS:
            epoch_counter = 0
            state = {"epoch": epoch, "state-dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "train-loss": train_losses, "val-loss": validation_losses, "best": best}
            torch.save(state, "epoch" + str(epoch + 1) + "-model")

        move_stuff(new_best)


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
                self.amps = np.log(current_file + 1)
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
            nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE), nn.ReLU(True),
            # nn.Linear(SECOND_LAYER_SIZE, THIRD_LAYER_SIZE), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(THIRD_LAYER_SIZE, SECOND_LAYER_SIZE), nn.ReLU(True),
            nn.Linear(SECOND_LAYER_SIZE, FIRST_LAYER_SIZE), nn.ReLU(True),
            nn.Linear(FIRST_LAYER_SIZE, INPUT_LAYER_SIZE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    print("Parameters: " + PARAMS)
    train_input, train_output, validation_input, validation_output, test_orig, test_128 = read_file_lists()
    train_model(train_input, train_output, validation_input, validation_output)
    test_model(test_orig, test_128)

    # find_lr(train_input, train_output, -7, -2)
    # find_lr(train_input, train_output, -7, -2, True)

    # convert_files(train_input)
    # convert_files(train_output)
    # convert_files(validation_input)
    # convert_files(validation_output)

    # plot_loss(os.path.join(DIRECTORY, PARAMS, "epoch" + str(EPOCHS) + "-model"), True)


if __name__ == '__main__':
    main()
