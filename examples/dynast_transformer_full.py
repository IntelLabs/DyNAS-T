"""
DyNAS-T Reference Search

SuperNetwork: HAT Transformer-Base

Search Tactic: Full Search (predictor training then search)

Description: Multi-Objective genetic algorithm search.
Assumes that the user has:
    * a latency predictor to predict the network, and the metadata
      with information on the hardware config
    * an AccuracyPredictor neural network that predicts the acc
The results can be saved to the `--csv_path` file.
"""
# Imports
import argparse
import copy
import csv
import pickle
from datetime import datetime

import numpy as np

from dynast.analytics_module.results import ResultsManager
from dynast.evaluation_module.predictor import (TransformerBleuPredictor,
                                                TransformerLatencyPredictor)  # TODO(Maciej) Change to `Predictor`
# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.search_module.search import (ProblemMultiObjective,
                                         SearchAlgoManager)


class HATRunner:
    '''
    The OFARunner is responsible for 'running' the subnetwork evaluation.
    '''
    def __init__(self, supernet, model_dir, lat_predictor, acc_predictor,unique_value_path=None):

        self.supernet = supernet
        self.model_dir = model_dir
        self.acc_predictor = acc_predictor
        self.latency_predictor = lat_predictor
        self.onehot_unique = unique_value_path


    def estimate_accuracy_bleu(self, subnet_cfg):

        # Ridge Predictor - 121 vector
        bleu = self.acc_predictor.predict_single(subnet_cfg)
        print(bleu)
        return bleu

    def estimate_accuracy_bleuq(self, subnet_cfg):

        # Ridge Predictor - 121 vector
        bleuq = self.acc_predictor.predict_single(subnet_cfg)
        return bleuq


    def estimate_latency(self, subnet_cfg):

        # Latency Predictor
        latency = self.latency_predictor.predict_single(subnet_cfg)
        return latency


    def onehot_custom(self, subnet_cfg):

        features = []

        features.extend(subnet_cfg['encoder_embed_dim'])

        encoder_layer_num = subnet_cfg['encoder_layer_num']
        encode_layer_num_int = encoder_layer_num[0]
        features.extend(encoder_layer_num)

        #Encoder FFN Embed Dim
        encoder_ffn_embed_dim = subnet_cfg['encoder_ffn_embed_dim']

        if encode_layer_num_int < 6:
            encoder_ffn_embed_dim.extend([0]*(6-encode_layer_num_int))
        features.extend(encoder_ffn_embed_dim)

        #Encoder Self-Attn Heads

        encoder_self_attention_heads = subnet_cfg['encoder_self_attention_heads'][:encode_layer_num_int]

        if encode_layer_num_int < 6:
            encoder_self_attention_heads.extend([0]*(6-encode_layer_num_int))
        features.extend(encoder_self_attention_heads)


        features.extend(subnet_cfg['decoder_embed_dim'])

        decoder_layer_num = subnet_cfg['decoder_layer_num']
        decoder_layer_num_int = decoder_layer_num[0]
        features.extend(decoder_layer_num)

        #Decoder FFN Embed Dim
        decoder_ffn_embed_dim = subnet_cfg['decoder_ffn_embed_dim'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_ffn_embed_dim.extend([0]*(6-decoder_layer_num_int))
        features.extend(decoder_ffn_embed_dim)


        #Decoder Attn Heads
        decoder_self_attention_heads = subnet_cfg['decoder_self_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
                    decoder_self_attention_heads.extend([0]*(6-decoder_layer_num_int))
        features.extend(decoder_self_attention_heads)

        #Decoder ENDE HEADS

        decoder_ende_attention_heads = subnet_cfg['decoder_ende_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
                    decoder_ende_attention_heads.extend([0]*(6-decoder_layer_num_int))

        features.extend(decoder_ende_attention_heads)

        arbitrary_ende_attn_trans = []
        for i in range(decoder_layer_num_int):
            if subnet_cfg['decoder_arbitrary_ende_attn'][i] == -1:
                arbitrary_ende_attn_trans.append(1)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 1:
                arbitrary_ende_attn_trans.append(2)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 2:
                arbitrary_ende_attn_trans.append(3)

        if decoder_layer_num_int < 6:
                    arbitrary_ende_attn_trans.extend([0]*(6-decoder_layer_num_int))
        features.extend(arbitrary_ende_attn_trans)

        examples = np.array([features])
        one_hot_count = 0
        unique_values = []
        with open(self.onehot_unique,'rb') as f:
            load_unique_values = pickle.load(f)
            unique_values = load_unique_values.tolist()
        for unique in unique_values:
            one_hot_count += len(unique.tolist())


        one_hot_examples = np.zeros((examples.shape[0], one_hot_count))
        for e, example in enumerate(examples):
            offset = 0
            for f in range(len(example)):
                index = np.where(unique_values[f] == example[f])[0] + offset
                one_hot_examples[e, index] = 1.0
                offset += len(unique_values[f])


        return one_hot_examples


class UserEvaluationInterface:
    '''
    The interface class update is required to be updated for each unique SuperNetwork
    framework as it controls how evaluation calls are made from DyNAS-T

    Parameters
    ----------
    evaluator : class
        The 'runner' that performs the validation or prediction
    manager : class
        The DyNAS-T manager that translates between PyMoo and the parameter dict
    csv_path : string
        (Optional) The csv file that get written to during the subnetwork search
    '''

    def __init__(self, evaluator, manager, csv_path=None):
        self.evaluator = evaluator
        self.manager = manager
        self.csv_path = csv_path

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        one_hot_feature = self.evaluator.onehot_custom(param_dict)


        sample = {
            'encoder_embed_dim': param_dict["encoder_embed_dim"],
            "decoder_embed_dim": param_dict["decoder_embed_dim"],
            "encoder_ffn_embed_dim":  param_dict["encoder_ffn_embed_dim"],
            "decoder_ffn_embed_dim":  param_dict["decoder_ffn_embed_dim"]
            }
        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        latency = self.evaluator.estimate_latency(one_hot_feature)
        bleu = self.evaluator.estimate_accuracy_bleuq(one_hot_feature)

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, latency, bleu]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, latency, -bleu



def main(args):

    # --------------------------------
    # OFA <-> DyNAS-T Interface Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager

    supernet_parameters = {"encoder_embed_dim": {'count':1,'vars':[640, 512]},
                           "decoder_embed_dim": {'count':1, 'vars': [640, 512]},
                           "encoder_ffn_embed_dim": {'count':6, 'vars':[3072, 2048, 1024]},
                           "decoder_ffn_embed_dim" : {'count':6,'vars': [3072, 2048, 1024]},
                           "encoder_layer_num": {'count':1,'vars':[6]},
                           "decoder_layer_num": {'count':1,'vars':[6, 5, 4, 3, 2, 1]},
                           "encoder_self_attention_heads": {'count':6, 'vars':[8, 4]},
                           "decoder_self_attention_heads": {'count':6, 'vars':[8, 4]},
                           "decoder_ende_attention_heads": {'count':6, 'vars':[8, 4]},
                           "decoder_arbitrary_ende_attn": {'count':6, 'vars':[-1, 1, 2]}}




    supernet_manager = ParameterManager(param_dict=supernet_parameters,
                                        seed=args.seed)
    # Objective 1 (Latency or MACs)
    supernet = args.supernet

    # Objective 2 (Accuracy)
    if args.accpred_path:
        print('[Info] Loading pre-trained accuracy predictor.')
        acc_pred = TransformerBleuPredictor()
        acc_pred.load(args.accpred_path)
    else:
        print('[Info] Building Accuracy Predictor')
        df = supernet_manager.import_csv(args.input_csv, config='config', objective='top1')
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = TransformerBleuPredictor()
        acc_pred.train(features, labels)

    if args.latpred_path:
       print('[Info] Loading pre-trained latency predictor.')
       lat_pred = TransformerLatencyPredictor()
       lat_pred.load(args.latpred_path)
    else:
       print('[Info] Building Latency Predictor')
       df = supernet_manager.import_csv(args.input_csv, config='config', objective='top1')
       features, labels = supernet_manager.create_training_set(df)
       lat_pred = TransformerLatencyPredictor()
       lat_pred.train(features, labels)

    # Instatiate objective 'runner'
    runner = HATRunner(supernet=supernet, model_dir=args.model_dir,lat_predictor=lat_pred,
                       acc_predictor=acc_pred,unique_value_path=args.unique_value_path)

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(evaluator=runner,
                                                   manager=supernet_manager,
                                                   csv_path=args.csv_path)

    # Instantiate Multi-Objective Problem Class
    problem = ProblemMultiObjective(evaluation_interface=evaluation_interface,
                                    param_count=supernet_manager.param_count,
                                    param_upperbound=supernet_manager.param_upperbound)

    # Instantiate Search Manager
    search_manager = SearchAlgoManager(algorithm=args.algorithm,
                                       seed=args.seed)
    search_manager.configure_nsga2(population=args.population,
                                   num_evals=args.num_evals)

    # Run the search!
    output = search_manager.run_search(problem)

    # Process results
    results = ResultsManager(csv_path=args.csv_path,
                             manager=supernet_manager,
                             search_output=output)
    results.history_to_csv()
    results.front_to_csv()



if __name__ == '__main__':
    # SuperNetwork Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/store/.torch/ofa_nets')
    parser.add_argument('--supernet', default='ofa_mbv3_d234_e346_k357_w1.0',
                        choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2'])
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--latpred_path', help='path to latency predictor')
    parser.add_argument('--accpred_path', help='path to pre-trained acc predictor')
    parser.add_argument('--unique_value_path', help='path to unique one-hot values')

    parser.add_argument('--input_csv', help='path to pre-trained acc predictor')

    # DyNAS-T Arguments
    parser.add_argument('--algorithm', default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='Search algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals', default=100000, type=int, help='number of TOTAL acc_pred, LUT samples')
    parser.add_argument('--csv_path', required=True, default=None, help='location to save results.')
    parser.add_argument('--population', default=50, type=int, help='population size for each generation')
    args = parser.parse_args()

    print('\n'+'-'*40)
    print('DyNAS-T Multi-Objective Search Starting')
    print('-'*40)

    main(args)
