import csv

from dynast.supernetwork.supernetwork_registry import SUPERNET_ENCODING, SUPERNET_METRICS, SUPERNET_PARAMETERS

if __name__ == '__main__':
    supernet: str = 'ofa_resnet50'
    results_path: str = '/nfs/site/home/mszankin/store/nosnap/results/dynast/dynast_ofaresnet50_random_a100.csv'
    save_to_path: str = (
        '/nfs/site/home/mszankin/store/code/handi/pretrained/predictors/datasets/resnet50_latency_a100_imagenet.csv'
    )
    seed: int = 42
    config_name: str = 'subnet'
    objective_name: str = 'latency'
    names = ['subnet', 'date'] + SUPERNET_METRICS[supernet]

    supernet_manager = SUPERNET_ENCODING[supernet](param_dict=SUPERNET_PARAMETERS[supernet], seed=seed)

    df = supernet_manager.import_csv(results_path, config=config_name, objective=objective_name, column_names=names)
    print(f'Rows: {len(df)}')

    with open(save_to_path, 'w', newline='', encoding='utf-8') as file:
        mywriter = csv.writer(file, delimiter=',', lineterminator='\n')
        for index, row in df.iterrows():
            subnet = row['subnet']
            features = subnet['d'] + subnet['w'] + subnet['e']
            predictor_row = features + [row[objective_name]]
            mywriter.writerow(predictor_row)
    print(f'Saved to: {save_to_path}')
