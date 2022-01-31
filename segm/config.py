import json


class Config:
    """Class to handle config.json."""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get(self, key):
        return self.config[key]

    def get_classes(self):
        return self.config['classes']

    def get_train(self, key):
        return self.config['train'][key]

    def get_val(self, key):
        return self.config['val'][key]

    def get_test(self, key):
        return self.config['test'][key]

    def get_image(self, key):
        return self.config['image'][key]
