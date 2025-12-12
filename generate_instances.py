from generator.Generator import *

import os
import json
import re

def save_instances(cfg, instances, type):
    save_dir = f"./data/{type}/{cfg.width}x{cfg.height}-{cfg.n_agents}"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, instance in enumerate(instances):
        payload = {
            "height": cfg.height,
            "width": cfg.width,
            "n_agents": cfg.n_agents,
            "grid": instance.grid.tolist(),
            "starts": instance.starts.tolist(),
            "goals": instance.goals.tolist(),
        }

        json_str = json.dumps(payload, indent=4, ensure_ascii=False)
        json_str = re.sub(
            r'\[\s*((?:-?\d+\s*,\s*)*-?\d+\s*)\]',
            lambda m: '[' + ', '.join(x.strip() for x in m.group(1).replace('\n', ' ').split(',')) + ']',
            json_str
        )

        filename = os.path.join(
            save_dir,                    
            f"instance_{idx}.json"
        )

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_str)

if __name__ == "__main__":
    width = 10
    height = 10
    n_agents = 10
    obstacle_ratio = 0.2
    n_train_instances = 1000
    n_test_instances = 100
    seed = 42

    cfg = MAPFGeneratorConfig(
        width=width,
        height=height,
        n_agents=n_agents,
        obstacle_ratio=obstacle_ratio,
        n_samples=n_train_instances,  
    )

    # * Train set
    generator = MAPFInstanceGenerator(cfg, seed=seed)
    instances = generator.sample_instances()
    save_instances(cfg, instances, 'train')

    # * Test set
    generator.set_n_instances(n_test_instances)
    instances = generator.sample_instances()
    save_instances(cfg, instances, 'test')