import yaml
# modify here to specify configure file you want to use
default_cfg = "src/config/config.yaml"

def getConfig(filename=default_cfg):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

args = getConfig()
