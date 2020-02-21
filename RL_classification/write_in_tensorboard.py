from tensorboardX import SummaryWriter
import torch

import layer
import sl_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path):
    model = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def convert_distrbution_to_frequency(distribution):
    # * Make float to integer
    distribution *= 1000
    concatenated_freq = []
    for freq in distribution:
        temp_freq = torch.zeros(0)
        for idx, val in enumerate(freq):
            temp = torch.zeros(round(val.item()))
            temp += int(idx)
            temp_freq = torch.cat([temp_freq, temp], dim=0)

        temp_freq = temp_freq.tolist()
        concatenated_freq += [temp_freq]

    return concatenated_freq


def write_frequency_histogram(model_path):
    model = load_model(model_path)
    sl = sl_train.SL()
    distribution, _, target = sl.get_distribution(model)

    writer = SummaryWriter(logdir='histogram/accuracy-90' + str())
    frequency = convert_distrbution_to_frequency(distribution)
    frequency_checker = {}
    for freq, tar in zip(frequency, target):
        step = 0
        str_tar = str(tar.item())

        if str_tar in frequency_checker:
            step = frequency_checker[str_tar]
        else:
            frequency_checker[str_tar] = 0

        writer.add_histogram(tag=str_tar, values=freq, global_step=step)
        frequency_checker[str_tar] += 1


if __name__ == "__main__":
    write_frequency_histogram('./model/accuracy90.pth')
    print("Done")
