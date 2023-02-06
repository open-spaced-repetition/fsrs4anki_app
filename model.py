import numpy as np
import torch
from torch import nn

init_w = [1, 1, 5, -0.5, -0.5, 0.2, 1.4, -0.02, 0.8, 2, -0.2, 0.5, 1]


class FSRS(nn.Module):
    def __init__(self, w):
        super(FSRS, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(w))
        self.zero = torch.FloatTensor([0.0])

    def forward(self, x, s, d):
        '''
        :param x: [review interval, review response]
        :param s: stability
        :param d: difficulty
        :return:
        '''
        if torch.equal(s, self.zero):
            # first learn, init memory states
            new_s = self.w[0] + self.w[1] * (x[1] - 1)
            new_d = self.w[2] + self.w[3] * (x[1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = torch.exp(np.log(0.9) * x[0] / s)
            new_d = d + self.w[4] * (x[1] - 3)
            new_d = self.mean_reversion(self.w[2], new_d)
            new_d = new_d.clamp(1, 10)
            # recall
            if x[1] > 1:
                new_s = s * (1 + torch.exp(self.w[6]) *
                             (11 - new_d) *
                             torch.pow(s, self.w[7]) *
                             (torch.exp((1 - r) * self.w[8]) - 1))
            # forget
            else:
                new_s = self.w[9] * torch.pow(new_d, self.w[10]) * torch.pow(
                    s, self.w[11]) * torch.exp((1 - r) * self.w[12])
        return new_s, new_d

    def loss(self, s, t, r):
        return - (r * np.log(0.9) * t / s + (1 - r) * torch.log(1 - torch.exp(np.log(0.9) * t / s)))

    def mean_reversion(self, init, current):
        return self.w[5] * init + (1-self.w[5]) * current


class WeightClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'w'):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)  # initStability
            w[1] = w[1].clamp(0.1, 5)  # initStabilityRatingFactor
            w[2] = w[2].clamp(1, 10)  # initDifficulty
            w[3] = w[3].clamp(-5, -0.1)  # initDifficultyRatingFactor
            w[4] = w[4].clamp(-5, -0.1)  # updateDifficultyRatingFactor
            w[5] = w[5].clamp(0, 0.5)  # difficultyMeanReversionFactor
            w[6] = w[6].clamp(0, 2)  # recallFactor
            w[7] = w[7].clamp(-0.2, -0.01)  # recallStabilityDecay
            w[8] = w[8].clamp(0.01, 1.5)  # recallRetrievabilityFactor
            w[9] = w[9].clamp(0.5, 5)  # forgetFactor
            w[10] = w[10].clamp(-2, -0.01)  # forgetDifficultyDecay
            w[11] = w[11].clamp(0.01, 0.9)  # forgetStabilityDecay
            w[12] = w[12].clamp(0.01, 2)  # forgetRetrievabilityFactor
            module.w.data = w


def lineToTensor(line):
    ivl = line[0].split(',')
    response = line[1].split(',')
    tensor = torch.zeros(len(response), 2)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][1] = int(response)
    return tensor


class Collection:
    def __init__(self, w):
        self.model = FSRS(w)

    def states(self, t_history, r_history):
        with torch.no_grad():
            line_tensor = lineToTensor(list(zip([t_history], [r_history]))[0])
            output_t = [(self.model.zero, self.model.zero)]
            for input_t in line_tensor:
                output_t.append(self.model(input_t, *output_t[-1]))
            return output_t[-1]
