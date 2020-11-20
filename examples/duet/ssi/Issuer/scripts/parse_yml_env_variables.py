# To run this:
# export DB_PASS=very_secret_and_complex
# python use_env_variables_in_config_example.py -c /path/to/yaml
# do stuff with conf, e.g. access the database password like this: conf['database']['DB_PASS']
# YAML FILE
# genesis-url: !ENV ${GENESIS_URL}
import argparse

import yaml
from yml_env_variables import parse_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAML environment variables substitution')
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    print(args)
    conf = parse_config(path=args.conf_file)
    print(conf)
    with open(r'/tmp/agent_conf.yml', 'w') as file:
        documents = yaml.dump(conf, file)