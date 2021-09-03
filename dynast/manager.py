





class SuperNetManager:
    """
    This class manages the supernet architecture (i.e. elastic parameters)
    """
        
    def __init__(self, parameter_dict, verbose=True):
        self.parameter_dict = parameter_dict
        self.verbose = verbose
    
    def map_to_onehot(self):
        pass
        
    def map_from_onehot(self):
        pass
    
    def describe(self):
        print('[Info] Parameter Dict:\n{}'.format(self.parameter_dict))
    
    def __len__(self):
        return len(self.parameter_dict)


class CustomImageClassifierManager(SuperNetManager):
    
    def __init__(self, parameter_dict=dict(), verbose=True):
        super().__init__(parameter_dict, verbose)
        self.block_counter = None
        
    def add_categorical(self, category, value):
        self.parameter_dict[category] = value
        
    def remove_categorical(self, category):
        if category in self.parameter_dict:
            self.parameter_dict.pop(category, None)
            print('[Info] removed category key {}'.format(category))
        else:
            print('[Warning] invalid category key {}'.format(category))
        
    def add_single_block(self, block_arch):
        if self.block_counter == None:
            self.block_counter = 0
        else:
            self.block_counter += 1
        
        if block_arch:
            self.parameter_dict['block_{}'.format(self.block_counter)] = block_arch
            print('[Info] Added block {} with parameters {}'.format(self.block_counter, block_arch))
        if self.verbose:
            print('[Info] Parameter Dict:\n{}'.format(self.parameter_dict))
    
    def replace_single_block(self, block_name, block_arch):
        if block_name not in self.parameter_dict:
            print('[Warning] block name not found in full parameter dictionary')
        else:
            self.parameter_dict[block_name] = block_arch
        if self.verbose:
            print('[Info] Parameter Dict:\n{}'.format(self.parameter_dict))

    def delete_block(self, block_name):
        if block_name in self.parameter_dict:
            self.parameter_dict.pop('key', None)
            print('[Info] removed block_name {}'.format(block_name))
        else:
            print('[Warning] invalid block_name {}'.format(block_name))
    

class HANDImageClfManager(SuperNetManager):
    
    def __init__(self, parameter_dict=dict(), verbose=True):
        super().__init__(parameter_dict, verbose)
        
    def build_mobilenetv3(self, resolution=224, width=None):
        self.parameter_dict = dict()
        sub_dict = dict()
        
        self.parameter_dict['r'] = resolution
        self.parameter_dict['wid'] = width
        for layer in range(4):
            sub_dict[f'layer_{layer}'] = {'kernel_size' : [3,5,7], 
                                          'expansion_ratio' : [3,4,6]}
        sub_dict[f'depth_options'] = [2,3,4]
        for block in range(5):
            self.parameter_dict[f'block_{block}'] = sub_dict
        if self.verbose:
            print('[Info] MobileNetV3 SuperNet parameter dictionary created:\
                  \n{}'.format(self.parameter_dict))
        
    def build_resnet50(self, resolution=224, width=None):
        raise NotImplementedError

    def pymoo_map(self):
        block_depth_map = list()
        kernel_size_map = list()
        expansion_ratio_map = list()

        for key, value in self.parameter_dict.items():
            if 'block' in key:
                for layer, operation in self.parameter_dict[key].items():
                    if 'layer' in layer:
                        if 'kernel_size' in operation:
                            kernel_size_map.append(self.parameter_dict[key][layer]['kernel_size']) 
                        if 'expansion_ratio' in operation:
                            expansion_ratio_map.append(self.parameter_dict[key][layer]['expansion_ratio'])
                    elif 'depth_options' in layer:
                        block_depth_map.append(self.parameter_dict[key]['depth_options'])

        return block_depth_map, kernel_size_map, expansion_ratio_map 
        
        

class BootstrapNASManager(SuperNetManager):
    
    def __init__(self, parameter_dict=dict()):
        super().__init__(parameter_dict)
        self.block_counter = None