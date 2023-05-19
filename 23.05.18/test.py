import yaml
import sys
import fire
import os
import inspect
import configargparse
from configargparse import YAMLConfigFileParser

data = "test"
def current():
    """
    return : the current dir path
    """
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    head, tail = os.path.split(current)
    print(head, tail)
    print("\n")
    return head

def main(file="test.yaml"):
    default = os.path.join(current(),data+'.yaml')
    print(default)

if __name__ == "__main__":
    sys.exit(fire.Fire(main))
