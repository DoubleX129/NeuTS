import random
import numpy as np
import pickle as cPickle
import random
import os

x_range = [0, 1]
y_range = [0, 1]
z_range = [0, 1]


class PreprocesserTS(object):
    def __init__(self, delta, dim_ranges):
        self.delta = delta
        self.dim_ranges = dim_ranges
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        self.grid_hash = []
        for dim, range in enumerate(self.dim_ranges):
            dmax, dmin = range[1], range[0]
            self.grid_hash.append(self._frange(dmax, dmin, self.delta[dim]))

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def get_grid_index(self, tuple):
        grid_tuple = []
        for i, dim_value in enumerate(tuple):
            grid_tuple.append(
                int((dim_value - self.dim_ranges[i][0]) / self.delta[i]))

        index = 0
        if len(grid_tuple) == 1:
            index = grid_tuple[0]
        elif len(grid_tuple) == 2:
            index = grid_tuple[0] + grid_tuple[1] * len(self.dim_ranges[0])
        elif len(grid_tuple) == 3:
            index = grid_tuple[0] + grid_tuple[1] * len(self.dim_ranges[0]) + grid_tuple[2] * len(
                self.dim_ranges[0]) * len(self.dim_ranges[1])

        return grid_tuple, index


def norm_data(ts_all):
    max_v, min_v = -1000, 1000
    for ts in ts_all:
        if max_v < max(ts):
            max_v = max(ts)
        if min_v > min(ts):
            min_v = min(ts)
    norm_ts = [[[(i - min_v) / (max_v - min_v)] for i in ts] for ts in ts_all]
    return norm_ts


def norm_data_3D(ts_all):
    max_v, min_v = [-1000, -1000, -1000], [1000, 1000, 1000]
    for ts in ts_all:
        if max_v[0] < max([i[0] for i in ts]):
            max_v[0] = max([i[0] for i in ts])
        if min_v[0] > min([i[0] for i in ts]):
            min_v[0] = min([i[0] for i in ts])
        if max_v[1] < max([i[1] for i in ts]):
            max_v[1] = max([i[1] for i in ts])
        if min_v[1] > min([i[1] for i in ts]):
            min_v[1] = min([i[1] for i in ts])
        if max_v[2] < max([i[2] for i in ts]):
            max_v[2] = max([i[2] for i in ts])
        if min_v[2] > min([i[2] for i in ts]):
            min_v[2] = min([i[2] for i in ts])
    norm_ts = [[[(i[0] - min_v[0]) / (max_v[0] - min_v[0]),
                 (i[1] - min_v[1]) / (max_v[1] - min_v[1]),
                 (i[2] - min_v[2]) / (max_v[2] - min_v[2])] for i in ts] for ts in ts_all]
    return norm_ts


def feature_generation(path='./data/FordA/', ts_number=1000000):
    fname = path.split('/')[-2]
    print(fname)
    ts_train = open(path + fname + '_TRAIN').readlines()
    ts_test = open(path + fname + '_TEST').readlines()
    ts_all = []
    ts_all_labels = []
    for l in ts_train + ts_test:
        ts = l.replace('\n', '').split(',')
        ts = [float(i) for i in ts]
        ts_all_labels.append(ts[0])
        ts = ts[1:]
        if fname == 'ItalyPowerDemand':
            ts_all.append(ts)
        else:
            ts_all.append([ts[i] for i in range(0, 96, 4)])
    ts_all = norm_data(ts_all)
    ts_all = ts_all[:ts_number]
    print(len(ts_all))
    print(ts_all[0])
    print(ts_all[1])
    print(len(ts_all[0]))
    print(ts_all_labels)

    preprocessor = PreprocesserTS(delta=[0.001], dim_ranges=[x_range])

    ts_all_grid = [
        [[preprocessor.get_grid_index(tuple=i)[1]] for i in ts] for ts in ts_all]
    print(ts_all_grid[0])

    length = len(ts_all[0])
    ts_index = {}
    for i, ts in enumerate(ts_all):
        ts_index[i] = ts
    cPickle.dump(ts_index, open(
        './features/{}_all_ts_index'.format(fname), 'wb'))
    cPickle.dump((ts_all, [], length), open(
        './features/{}_all_ts_value'.format(fname), 'wb'))
    cPickle.dump((ts_all_grid, [], length), open(
        './features/{}_all_ts_grid'.format(fname), 'wb'))
    cPickle.dump((ts_all_labels, [], length), open(
        './features/{}_all_ts_label'.format(fname), 'wb'))

    return './features/{}_all_ts_value'.format(fname), fname


def feature_generation_3D(path='./data/UWaveGestureLibraryAll/', ts_number=5000):
    fname = path.split('/')[-2]
    print(fname)
    ts_train = open(path + fname + '_TRAIN').readlines()
    ts_test = open(path + fname + '_TEST').readlines()
    ts_all = []
    ts_all_labels = []
    for l in ts_train + ts_test:
        ts = l.replace('\n', '').split(',')
        ts = [float(i) for i in ts]
        ts_all_labels.append(ts[0])
        ts = ts[1:]
        ts_all.append([[ts[i], ts[315 + i], ts[315 * 2 + i]]
                       for i in range(0, 315, 15)])
        # ts_all.append(ts[80:])

    ts_all = norm_data_3D(ts_all)
    ts_all = ts_all[:ts_number]
    print(len(ts_all))
    print(ts_all[0])
    print(ts_all[1])
    print(len(ts_all[0]))

    preprocessor = PreprocesserTS(delta=[0.02, 0.02, 0.02], dim_ranges=[
        x_range, y_range, z_range])

    ts_all_grid = [[preprocessor.get_grid_index(
        tuple=i)[0] for i in ts] for ts in ts_all]
    print(ts_all_grid[0])

    length = len(ts_all[0])
    ts_index = {}
    for i, ts in enumerate(ts_all):
        ts_index[i] = ts
    cPickle.dump(ts_index, open(
        './features/{}_ts_index'.format(fname), 'wb'))
    cPickle.dump((ts_all, [], length), open(
        './features/{}_ts_value'.format(fname), 'wb'))
    cPickle.dump((ts_all_grid, [], length), open(
        './features/{}_ts_grid'.format(fname), 'wb'))
    cPickle.dump((ts_all_labels, [], length), open(
        './features/{}_ts_label'.format(fname), 'wb'))

    return './features/{}_ts_value'.format(fname), fname


def get_data_path(path):
    labels = set()
    user_list = os.listdir(path)
    data_num = 0
    data_path = []
    for user in user_list:
        if os.path.exists(path + '/' + user + '/labels.txt'):
            with open(path + '/' + user + '/labels.txt') as f:
                for line in f.readlines()[1:]:
                    file_name = path + '/' + user + '/Trajectory/' + line.split()[0].replace('/', '') + line.split()[
                        1].replace(':', '') + '.plt'
                    if os.path.exists(file_name):
                        labels.add(line.split()[-1])
                        data_path.append([file_name, line.split()[-1]])
                        data_num += 1
    print(f"Toral Trajectory {data_num}")
    labels = list(labels)
    labels.sort()
    print(f"labels: {labels}")
    return data_path, labels


def get_trajectory(path):
    trajectory = []
    with open(path) as f:
        for line in f.readlines()[6:]:
            latitude, longitude, _, altitude = line.split(',')[:4]
            trajectory.append([eval(latitude), eval(longitude), eval(altitude)])
    return trajectory


def feature_generation_geolife(path='/home/guanxinyan2022/NeuTS/Geolife Trajectories 1.3/Data', ts_number=5000):
    data_path, labels = get_data_path(path)
    labels2id = {labels[i]: i + 1 for i in range(len(labels))}
    print(labels2id)
    random.shuffle(data_path)
    data_path = data_path[:ts_number]
    fname = 'geolife'

    ts_all = []
    ts_all_labels = []
    for data in data_path:
        ts_all.append(get_trajectory(data[0]))
        ts_all_labels.append(labels2id[data[1]])

    ts_all = norm_data_3D(ts_all)
    ts_all = ts_all[:ts_number]
    print(len(ts_all))
    print(ts_all[0])
    print(ts_all[1])
    print(len(ts_all[0]))

    preprocessor = PreprocesserTS(delta=[0.02, 0.02, 0.02], dim_ranges=[
        x_range, y_range, z_range])

    ts_all_grid = [[preprocessor.get_grid_index(
        tuple=i)[0] for i in ts] for ts in ts_all]
    print(ts_all_grid[0])

    length = len(ts_all[0])
    ts_index = {}
    for i, ts in enumerate(ts_all):
        ts_index[i] = ts
    cPickle.dump(ts_index, open(
        './features/{}_ts_index'.format(fname), 'wb'))
    cPickle.dump((ts_all, [], length), open(
        './features/{}_ts_value'.format(fname), 'wb'))
    cPickle.dump((ts_all_grid, [], length), open(
        './features/{}_ts_grid'.format(fname), 'wb'))
    cPickle.dump((ts_all_labels, [], length), open(
        './features/{}_ts_label'.format(fname), 'wb'))

    return './features/{}_ts_value'.format(fname), fname
