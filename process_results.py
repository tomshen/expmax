import os
import numpy as np

import util
import process_data

RESULTS_DIRECTORY = util.RESULTS_DIRECTORY

def load_results_file(location_type, location_name):
    with open(os.path.join(RESULTS_DIRECTORY, os.path.join(location_type, location_name + '.results'))) as f:
        # at this point, we assume we have properly formatted data
        def clean_line(line):
            return line.strip().replace('[', '').replace(']', '')
        def to_float_array(line):
            return map(float, clean_line(line).replace(', ', ' ').split(' '))
        means = []
        covs = []
        lines = f.readlines()
        means_raw = lines[1:lines.index('Covariances:\n')]
        covs_raw = lines[lines.index('Covariances:\n')+1:len(lines)]
        means_temp = [mean_raw.strip().split(', ') for mean_raw in means_raw]
        for mean in means_temp:
            if len(mean) == 2:
                means.append([float(mean[0].replace('[', '')), float(mean[1].replace(']', ''))])
        assert(len(covs_raw) % 2 == 0)
        i = 0
        while i < len(covs_raw):
            covs.append([to_float_array(covs_raw[i]), to_float_array(covs_raw[i+1])])
            i += 2
        assert(len(covs) == len(means))
        return means, covs

def main():
    location_type = 'city'
    for city in open(os.path.join(util.DATA_DIRECTORY, location_type + '.txt')):
        location = city.strip()
        model_means, model_covs = load_results_file(location_type, location)
        data = util.rearrange_data(process_data.get_data_coords(location))
        filepath = os.path.join(RESULTS_DIRECTORY, os.path.join(location_type, location + '.png'))
        util.plot_data_model(location, data, model_means, model_covs, False, filepath)

if __name__ == '__main__':
    main()