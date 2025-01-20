import yaml

class ConfigLoader:
    _instance = None

    def __new__(cls, config_file='configs.yaml'):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance.config_file = config_file
            cls._instance.params = cls._instance.load_params()
        return cls._instance

    def load_params(self):
        try:
            with open(self.config_file, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except Exception as e:
            raise Exception(f"Failed to load params file: {str(e)}")

    def get_yolo_weight(self):
        return self.params['yolo']['weight']

    def get_arcface_weight(self):
        return self.params['arcface']['weight']

    def get_faiss_index(self):
        return self.params['faiss']['index']
