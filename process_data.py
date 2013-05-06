import os
import matplotlib.pyplot as plt

import util

DATA_DIRECTORY = util.DATA_DIRECTORY
DATA_SOURCE = util.DATA_SOURCE

def get_seed_coords(key):
    coord_list = []
    with open(os.path.join(DATA_SOURCE, 'disambiguations.latlng.ntnt')) as f:
        lines = f.read().split('\n')
        for line in lines:
            if key in line:
                toks = line.split(' ')
                coord_list.append((float(toks[4][1:]), float(toks[5][:toks[5].index('"')])))
    return coord_list

def get_cluster_coords(key, location_type):
    coord_list = []
    with open(os.path.join(DATA_DIRECTORY, os.path.join(location_type, key + '.clusters'))) as f:
        for line in f:
            coord_list.append([float(c) for c in line.strip().split()])
    return coord_list

def get_data_coords(key):
    coord_list = []
    with open(os.path.join(DATA_SOURCE, 'clusterSourcesByID.tsv')) as f:
        lines = f.read().split('\n')
        for line in lines:
            if key in line:
                toks = line.split('\t')
                coord_list.append((float(toks[2]), float(toks[3])))
    return coord_list

def get_seed_keys(sort=False):
    seeds = {}
    with open(os.path.join(DATA_SOURCE, 'disambiguations.latlng.ntnt')) as f:
        lines = f.read().split('\n')
        for line in lines:
            if '<http://dbpedia.org/resource/' in line:
                # len('<http://dbpedia.org/resource/') == 29
                key = line[29:line.index('>')]
                if key not in seeds:
                    seeds[key] = 0
                seeds[key] += 1
    return seeds

def get_seed_keys_sorted():
    import operator
    return sorted(get_seed_keys().iteritems(), key=operator.itemgetter(1), reverse=True)

def write_seeds():
    seeds = get_seed_keys_sorted()
    with open(os.path.join(DATA_DIRECTORY('seed_keys.txt', 'w+'))) as f:
        for cluster in seeds:
            f.write(str(cluster) + '\n')

def get_data_keys(sort=False):
    data = {}
    with open(os.path.join(DATA_SOURCE, 'clusterSourcesByID.tsv')) as f:
        lines = f.read().split('\n')
        for line in lines:
            if '<http://dbpedia.org/resource/' in line:
                key = line[29:line.index('>')]
                if key not in data:
                    data[key] = 0
                data[key] += 1
    return data

def get_data_keys_sorted():
    import operator
    return sorted(get_data_keys().iteritems(), key=operator.itemgetter(1), reverse=True)

def write_data():
    data = get_data_keys_sorted()
    with open(os.path.join(DATA_DIRECTORY('data_keys.txt', 'w+'))) as f:
        for cluster in data:
            f.write(str(cluster) + '\n')

def plot_data(key):
    util.plot_points(util.rearrange_data(get_data_coords(key)))

def plot_seeds(key):
    util.plot_points(util.rearrange_data(get_seed_coords(key)))

def plot_clusters(key, location_type):
    util.plot_points(util.rearrange_data(get_cluster_coords(key, location_type)))

def plot_data_seeds(key):
    util.plot_data_seeds(util.rearrange_data(get_data_coords(key)),
        util.rearrange_data(get_seed_coords(key)), title=key)

def plot_data_clusters(key, location_type):
    util.plot_data_seeds(util.rearrange_data(get_data_coords(key)),
        util.rearrange_data(get_cluster_coords(key, location_type)), title=location_type + ': ' + key)

def add_to_test_data(key, location_type):
    data_coords = util.rearrange_data(get_data_coords(key))
    data_filename = key + '.data'
    if not os.path.exists(os.path.join(DATA_DIRECTORY, location_type)):
        os.makedirs(os.path.join(DATA_DIRECTORY, location_type))
    data_filepath = os.path.join(DATA_DIRECTORY, os.path.join(location_type, data_filename))
    with open(data_filepath, 'w+') as f:
        f.write(str(data_coords).replace('], [', '\n').replace('[[', '').replace(']]', '').replace(', ', ' '))
    seed_coords = util.rearrange_data(util.rearrange_data(get_seed_coords(key)))
    seed_filename = key + '.seeds'
    if not os.path.exists(os.path.join(DATA_DIRECTORY, location_type)):
        os.makedirs(os.path.join(DATA_DIRECTORY, location_type))
    seed_filepath = os.path.join(DATA_DIRECTORY, os.path.join(location_type, seed_filename))
    with open(seed_filepath, 'w+') as f:
        f.write(str(seed_coords).replace('], [', '\n').replace('[[', '').replace(']]', '').replace(', ', ' '))

def write_with_stats(locations, filename):
    seeds = get_seed_keys() # dict
    data = get_data_keys() # dict
    clusters = []
    for key, t in locations:
        clusters.append((key, t, seeds[key], data[key]))
    with open(os.path.join('meta', filename), 'w+') as f:
        for c in clusters:
            f.write(c[0] + ' (' + c[1] + '): ' + str(c[2]) + ' seeds, ' + str(c[3]) + ' data points\n')

def main():
    pass

if __name__ == "__main__":
    main()