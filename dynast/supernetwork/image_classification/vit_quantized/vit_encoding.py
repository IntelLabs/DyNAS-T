import numpy as np

from dynast.supernetwork.image_classification.vit.vit_encoding import ViTEncoding


class ViTQuantizedEncoding(ViTEncoding):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)

    def onehot_custom(self, subnet_cfg, provide_onehot=True, max_layers=12):
        features = []
        num_layers = subnet_cfg['num_layers'][0]

        attn_head_list = subnet_cfg['num_attention_heads'][:num_layers] + [0] * (max_layers - num_layers)
        intermediate_size_list = subnet_cfg['vit_intermediate_sizes'][:num_layers] + [0] * (max_layers - num_layers)
        features = [num_layers] + attn_head_list + intermediate_size_list

        qbit_list = (
            [subnet_cfg['q_bits'][0]]
            + subnet_cfg['q_bits'][1 : 6 * num_layers + 1]
            + [0] * (max_layers - num_layers) * 6
            + [subnet_cfg['q_bits'][-1]]
        )
        features = features + qbit_list

        if provide_onehot == True:
            examples = np.array([features])
            one_hot_count = 0
            # TODO(macsz,sharathns93) `self.unique_values` might be uninitialized
            # if `create_training_set` hasn't been called prior to this
            unique_values = self.unique_values

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

        else:
            return features
