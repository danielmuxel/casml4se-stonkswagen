import numpy as np
import matplotlib.pyplot as plt

from .transform import reindex

def implot(data, title=None, scale=False, cmap='viridis', label=None, minVal=None, maxVal=None):
    dayOfWeek = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
    if scale:
        maxVal = data.max()
        minVal = data.min()

        absMax = max(abs(maxVal), abs(minVal))
        maxVal = absMax
        minVal = -absMax
    else:
        if minVal is None:
            minVal = data.min()
        if maxVal is None:
            maxVal = data.max()

    data, stepsPerWeek, completeTimestamps = reindex(data)

    x = np.arange(0, stepsPerWeek, stepsPerWeek // 14)
    y = np.arange(0, len(completeTimestamps) // stepsPerWeek, 4)

    xLabels = []

    for i in x:
        day = completeTimestamps[i].dayofweek
        hour = completeTimestamps[i].hour
        xLabels.append(f'{dayOfWeek[day]}-{hour:02d}')

    yLabels = []
    for i in y:
        weekOfYear = completeTimestamps[i*stepsPerWeek].weekofyear
        year = completeTimestamps[i*stepsPerWeek].year
        yLabels.append(f'{year}-{weekOfYear:02d}')

    meterMatrix = data.values.reshape(-1, stepsPerWeek)

    fig, ax = plt.subplots(figsize=(15, 10))

    im = ax.imshow(meterMatrix, aspect='auto', interpolation='none', cmap=cmap, vmin=minVal, vmax=maxVal)
    ax.set_yticks(y, labels=yLabels)
    ax.set_xticks(x, labels=xLabels)
    if title is not None:
        ax.set_title(title)
    cbar = fig.colorbar(im)

    if label is not None:
        cbar.set_label(label)

    return fig