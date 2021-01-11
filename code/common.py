import os
import logging

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)

filehandler = logging.FileHandler("log.txt", mode='a') # mode='w'
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s', '%Y-%m-%d %H:%M:%S')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s', '%H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)


def listfile(dirpath):
    allfile = []
    for filename in os.listdir(dirpath):
        child = dirpath + '/' + filename
        allfile.append(child)
    return allfile
