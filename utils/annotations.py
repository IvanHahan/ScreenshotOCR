import json
import os


def parse_annotation(path):
    with open(path, 'r') as f:
        annotation = json.loads(f.read())
        return annotation
