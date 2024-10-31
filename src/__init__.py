import glob
import os

# Get the current directory
module_dir = os.path.dirname(__file__)

# Import all .py files in the current directory (excluding __init__.py)
for module_file in glob.glob(os.path.join(module_dir, '*.py')):
    module_name = os.path.basename(module_file)[:-3]  # Strip the .py extension
    if module_name != '__init__':
        exec(f'from .{module_name} import *')
