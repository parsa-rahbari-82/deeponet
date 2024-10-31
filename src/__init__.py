
import os
import pkgutil

# Automatically import all modules in the current package
__all__ = []

# Iterate through all modules in the package
for _, module_name, _ in pkgutil.iter_modules(__path__):
    __all__.append(module_name)
    # Dynamically import the module
    __import__(f'.{module_name}', locals(), globals())