from setuptools import setup, find_packages

setup(
    name='model_sat',
    version='0.1.0',
    author='Felicio Cassalho',
    description='Satellite data download, crop, and collocation with model output',
    packages=find_packages(),
    install_requires=[
        #'ocsmesh',  # Optional if you want users to install it
        'numpy',
        'xarray',
        'requests',
        'scipy',
        'tqdm',
    ],
    python_requires='>=3.8',
)
