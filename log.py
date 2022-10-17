import logging
import os
import environ

if not os.path.exists('log'):
    os.makedirs('log')

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

env = environ.Env()
environ.Env.read_env()

ENVIRONMENT = env('ENVIRONMENT', default='LOCAL')

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = logging.FileHandler(f'log/{ENVIRONMENT}.txt', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def log(message):
    logger.info(message)

