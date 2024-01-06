import argparse
import os

import yaml

'''
get hyperparamters from a yaml file
yaml file e.g.:
train:
    train_epoch: 100
    
'''


class AnaArgParser:
    def __init__(self, cfg_path=None, exp_dir=None, exp_datetime=None, exp_description=None):
        parser = argparse.ArgumentParser(description=exp_description)
        parser.add_argument('--config', type=str, default=cfg_path, help='configs file')
        parser.add_argument('--datetime', type=str, default=exp_datetime, help='datetime')
        parser.add_argument('--exp_dir', type=str, default=exp_dir, help='exp directory')
        parser.add_argument('--local_rank', type=int, default=-1, help='distributed training')
        args = parser.parse_args()
        assert args.config is not None
        self.cfg = self.load_cfg_from_cfg_file(args.config)
        self.cfg.__setattr__('exp_dir', exp_dir if exp_dir is not None else args.exp_dir)
        self.cfg.__setattr__('datetime', exp_datetime if exp_datetime is not None else args.datetime)
        self.cfg.__setattr__('local_rank', args.local_rank)

    @staticmethod
    def load_cfg_from_cfg_file(file):
        cfg = {}
        assert os.path.isfile(file) and file.endswith('.yaml'), \
            '{} is not a yaml file'.format(file)

        with open(file, 'r') as f:
            cfg_from_file = yaml.safe_load(f)

        for key in cfg_from_file:
            for k, v in cfg_from_file[key].items():
                cfg[k] = v

        cfg = CfgNode(cfg)
        return cfg


class CfgNode(dict):
    def __init__(self, init_dict=None, key_list=None):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
