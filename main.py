import argparse
import numpy as np
from train import train_and_eval
from configure import get_default_config


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yaleB', ##############################
                        help='the name of dataset [default: cub')
parser.add_argument('--missing-rate', type=float, default=0.5,
                        help='view missing rate [default: 0]')
args = parser.parse_args()
print(args.dataset)

metric_names = ['acc', 'macro_P', 'micro_P', 'macro_R', 'micro_F', 'macro_F']
accumulated_metrics = {n:[] for n in metric_names}

for seed in range(5):################################################################
    config = get_default_config(args.dataset)
    config['training']['missing_rate'] = args.missing_rate
    #config['training']['batch_size'] = 10###
    scores = train_and_eval(config, seed)
    for k, v in accumulated_metrics.items():
        v.append(scores[k])
    print(scores)

for k, v in accumulated_metrics.items():
    print(k,str(round(100*np.mean(v),5))+chr(177)+str(round(100*np.std(v),5)))

print(args)
for k, v in accumulated_metrics.items():
    print(k,str(round(100*np.mean(v),2))+chr(177)+str(round(100*np.std(v),2)))