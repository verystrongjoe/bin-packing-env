from environment import PalleteWorld
import pickle
from sample_agent.reinforce_agg import *

fn = 'env_data.npy'
# env = PalleteWorld(env_id=0, n_random_fixed=5)
# env.seed(1)
# env.save_dataset(fn)
#
fn = 'env_data.npy'
l = []
with open(fn, 'rb') as f:
    l = pickle.load(f)

for i, items in enumerate(l):
    env = PalleteWorld(datasets=[items],env_id=i)
    main(env)