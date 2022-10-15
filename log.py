import logging
import os

if not os.path.exists('log'):
    os.makedirs('log')

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = logging.FileHandler('log/mapreducetree.log', mode='w', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def log(message):
    logger.info(message)

