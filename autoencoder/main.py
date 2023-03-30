import torch
from torch import Tensor, nn, FloatTensor
import numpy as np
from tqdm import tqdm
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('on device: ', device)


class MoodPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_factor = 4
        self.downsampling = nn.Conv1d(1, 1, kernel_size=self.down_factor, stride=self.down_factor, padding='valid')
        self.downsampling.weight = nn.Parameter(
            torch.full(self.downsampling.weight.size(), 1 / self.downsampling.weight.size()[-1]), requires_grad=False)

        self.requires_grad_(False)

    def forward(self, music):
        #print(music.size())
        return self.downsampling(music)


class MoodEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=128, stride=16, padding='valid')
        self.conv2 = nn.Conv1d(3, 12, kernel_size=128, stride=8, padding='valid')
        self.conv3 = nn.Conv1d(12, 24, kernel_size=64, stride=8, padding='valid')
        self.dense1 = nn.Linear(23 * 24, 400)

    def forward(self, music):
        #print(music.size())
        music = torch.tanh(self.conv1(music))
        #print(music.size())
        music = torch.tanh(self.conv2(music))
        #print(music.size())
        music = torch.tanh(self.conv3(music))
        #print(music.size())
        music = music.view(-1, 23 * 24)
        music = torch.tanh(self.dense1(music))
        #print(music.size())
        return music


class MoodDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(24, 12, kernel_size=64, stride=8, output_padding=2)
        self.conv2 = nn.ConvTranspose1d(12, 3, kernel_size=128, stride=8, output_padding=4)
        self.conv3 = nn.ConvTranspose1d(3, 1, kernel_size=128, stride=16, output_padding=3)
        self.dense1 = nn.Linear(400, 23 * 24)

    def forward(self, music):
        music = torch.tanh(self.dense1(music))
        music = music.view(-1, 24, 23)
        #print(music.size())
        music = torch.tanh(self.conv1(music))
        #print(music.size())
        music = torch.tanh(self.conv2(music))
        #print(music.size())
        music = torch.tanh(self.conv3(music))
        #print(music.size())
        return music


class Mood(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = MoodPreprocess()

        self.encoder = MoodEncoder()
        self.decoder = MoodDecoder()

        self.to(device)

    def forward(self, music):
        music = self.encoder.forward(music)
        music = self.decoder.forward(music)
        return music

    def mood(self, music):
        music = self.preprocess(music)
        return self.encoder.forward(music)

    def batch_loop(self, train_music, optimizer, loss_func, e_num=''):

        train_loss = 0
        for sample in tqdm(train_music, desc=f'Epoch: {e_num}', unit='batch'):
            sample = sample.to(device)
            truth = self.preprocess(sample)
            pred = self(truth)

            loss = loss_func(pred, truth)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.train(False)
        test_loss = 0
        for sample in train_music.test():
            sample = sample.to(device)
            truth = self.preprocess(sample)
            pred = self(truth)

            test_loss += float(loss_func(pred, truth))
        self.train(True)
        return train_loss / train_music.train_len(), test_loss / train_music.test_len()

    def epoch_loop(self, train_music, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-5)
        loss_func = nn.MSELoss()
        ep_manager = EpochManager(epochs, model=self, detailed=50)

        for epoch in ep_manager:
            ep_manager += self.batch_loop(train_music, optimizer, loss_func, e_num=epoch)
            ep_manager(optimizer)

        return ep_manager


class EpochManager:
    def __init__(self, epochs, detailed=100, lr_decay_limit=0.01, lr_decay_speed=50, model=None):
        self.epochs = epochs
        self.early_stopping = False
        self.train_loss = []
        self.test_loss = []
        self.detailed = detailed
        self.last_changed_opti = 0
        self.lr_decay_limit = lr_decay_limit
        self.lr_decay_speed = lr_decay_speed
        self.model = model

        self.minimum = None

    def __call__(self, optimizer):
        if len(self.test_loss) - self.last_changed_opti > self.lr_decay_speed and (self.train_loss[-2] - self.train_loss[-1]) / self.test_loss[-2] < self.lr_decay_limit:
            self.last_changed_opti = len(self.test_loss)
            #print('Update Optimizer', len(self.test_loss))
            #for g in optimizer.param_groups:
            #    g['lr'] /= 2
        if self.minimum is None:
            self.minimum = self.test_loss[-1]
        elif len(self.test_loss) % self.detailed:
            if min(self.test_loss[len(self.test_loss) - self.detailed:]) < self.minimum:
                torch.save(self.model, 'checkpoint')
                self.minimum = min(self.test_loss[len(self.test_loss) - self.detailed:])

    def __iter__(self):
        for epoch in range(self.epochs):
            if epoch % self.detailed == 0:
                self.show('Models/train.png')
            if not self.early_stopping:
                yield epoch
        self.show('Models/train.png')

    def __add__(self, other):
        train, test = other
        train, test = float(train), float(test)
        self.train_loss.append(train)
        self.test_loss.append(test)
        print(f'Epoch {len(self.train_loss)}:   train: {train}, test: {test}')
        return self

    def show(self, path=None):
        import matplotlib.pyplot as plt
        plt.plot(self.train_loss, marker='x', label='train')
        plt.plot(self.test_loss, marker='x', label='test')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

    def __exit__(self):
        if self.model is not None:
            torch.save(self.model, 'model_exit')
            pickle.dump([self.train_loss, self.test_loss], open('histroy.pkl', 'rb'))


class SongGenerator:
    def __init__(self, song, flag):
        # input in the NN in which format/normalization (has an effect on the output)
        self.song = song
        self.flag = flag

    def __iter__(self):
        for i in range(264000, len(self.flag)):  # 44 kHz * 6 sec
            if self.flag[i]:
                yield Tensor(self.song[i - 264000: i].reshape(1, 1, -1)).to(device)

    def __len__(self):
        return int(np.sum(self.flag[264000:]))


class SnippetGenerator:
    def __init__(self, music, beat_timings, batchsize=1):
        # test and train
        self.music = music
        mask = np.zeros(len(beat_timings), dtype=bool)
        select = np.random.choice(np.arange(len(beat_timings)), size=int(len(beat_timings) * 0.2))
        print(select)
        mask[select] = True
        self.beats = beat_timings[~mask]
        self.test_beats = beat_timings[mask]
        self.batch = batchsize
        self.timing = 132300  # 6s * 22 050 kHz

    def __getitem__(self, item):
        return self.music[item - self.timing: item]

    def __len__(self):
        return len(self.beats) // self.batch

    def __iter__(self):
        batches = np.random.choice(self.beats, size=(len(self.beats) // self.batch, self.batch), replace=False)
        for batch in batches:
            yield Tensor(np.concatenate([self.music[snip - self.timing: snip].reshape(1, 1, -1) for snip in batch], axis=0))

    def test(self):
        batches = np.random.choice(self.test_beats, size=(len(self.test_beats) // self.batch, self.batch), replace=False)
        for batch in batches:
            yield Tensor(
                np.concatenate([self.music[snip - self.timing: snip].reshape(1, 1, -1) for snip in batch], axis=0))

    def test_len(self):
        return int(len(self.test_beats) // self.batch * self.batch)

    def train_len(self):
        return int(len(self.beats) // self.batch * self.batch)


if __name__ == '__main__':
    x = torch.load('checkpoint')
    #print(x)

    m = Mood()
    print(m)
    #music = np.load('audio.npy')
    #print(music, np.max(music), np.min(music), music.dtype)
    #flags = np.load('flags.npy') * 22050

    print('loading data')
    music = np.load('G:/ProjectEuler/techno_scraper/GerberAI/training_data/audio.npy')
    flags = np.load('G:/ProjectEuler/techno_scraper/GerberAI/training_data/flags.npy')
    print(len(flags))
    print('data loaded')
    #flags = np.array(flags[np.searchsorted(flags, 132300):], dtype=int)
    #print((flags))
    gen = SnippetGenerator(music, flags, 2048)

    #[print(x.size()) for x in gen]
    #print(flags, np.sum(flags))
    #print(len(music))
    #t = Tensor(music[0:264000].reshape(1, 1, -1))
    #t = torch.rand((1, 1, 44*10**3 * 6))
    #x.forward(t)
    #m.forward(t)
    #t.to(device)
    #print(t.device)
    #gen = SongGenerator(np.load('audio.npy'), np.load('flags.npy'))
    print(len(gen))
    #for item in gen:
    #    item = item.to(device)
    #    m.forward(m.preprocess(item))
    #    exit()
    #m = x
    res = m.epoch_loop(gen, 10)
    res.show()
    #print(m.downsampling.weight)
    #print([x.is_cuda for x in m.parameters()])
    #print(m.encoder.conv2.weight.get_device())
    torch.save(m, 'Models/first_big_data')



