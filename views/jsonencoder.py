import json
import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert NumPy types to standard Python types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
