# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
from datetime import datetime

from dynast.utils import log


# TODO(macsz) Not used anywhere. Drop?
class ResultsManager:
    '''
    Search results are stored in a Pymoo result object. This class manages the retrieval
    of the results.

    csv_path - output csv file name
    manager - ParameterManager object
    search_output - pymoo results object from the SearchAlgoManager
    '''

    def __init__(self, csv_path, manager, search_output):

        self.csv_path = csv_path
        self.manager = manager
        self.search_output = search_output

    def front_to_csv(self, filepath=None, overwrite=True):

        if filepath is None:
            filepath = self.csv_path[:-4] + '_front.csv'

        if overwrite:
            with open(filepath, 'w') as f:
                writer = csv.writer(f)

        # Note: move to DataFrame management in the future
        final_pop_params = self.search_output.pop.get('X')
        final_pop_objectives = self.search_output.pop.get('F')
        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            for i in range(len(final_pop_params)):
                obj_x, obj_y = final_pop_objectives[i][0], final_pop_objectives[i][1]
                sample = self.manager.translate2param(final_pop_params[i])
                date = str(datetime.now())
                writer.writerow([sample, date, obj_x, -obj_y])
        log.info('Final search population saved to: {}'.format(filepath))

        return None

    def history_to_csv(self, filepath=None, overwrite=True):

        if filepath is None:
            filepath = self.csv_path

        if overwrite:
            with open(filepath, 'w') as f:
                writer = csv.writer(f)

        # Note: move to DataFrame management in the future
        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            for iter in range(len(self.search_output.history)):
                hist_pop_params = self.search_output.history[iter].result().pop.get('X')
                hist_pop_objectives = self.search_output.history[iter].result().pop.get('F')
                for i in range(len(hist_pop_params)):
                    obj_x, obj_y = hist_pop_objectives[i][0], hist_pop_objectives[i][1]
                    sample = self.manager.translate2param(hist_pop_params[i])
                    date = str(datetime.now())
                    writer.writerow([sample, date, obj_x, -obj_y])
        log.info('Full search history saved to: {}'.format(filepath))

        return None
