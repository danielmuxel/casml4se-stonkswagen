import pandas as pd

def reindex(df):
    timestamps = df.index
    freq = timestamps[1] - timestamps[0]

    start = timestamps[0]
    while start.dayofweek != 0:
        start = start - pd.Timedelta(days=1)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)

    end = timestamps[-1]
    while end.dayofweek != 6:
        end = end + pd.Timedelta(days=1)
    end = end.replace(hour=23, minute=59, second=59, microsecond=0)

    completeTimestamps = pd.date_range(start=start, end=end, freq=freq)

    df = df.reindex(completeTimestamps, method=None)

    if freq == pd.Timedelta(hours=1):
        stepsPerWeek = 7*24
    elif freq == pd.Timedelta(minutes=5):
        stepsPerWeek = 7*24*12
    else:
        print("Frequency not supported.")
        exit(-1)

    return df, stepsPerWeek, completeTimestamps