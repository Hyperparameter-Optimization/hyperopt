from setuptools import setup, find_packages

setup(
    name='hyperOpt',
    version='0.0.1',
    description='Evolutionary algorithms for hyperparameter optimization',
    url='',
    author=['Laurits Tani'],
    author_email='laurits.tani@cern.ch',
    license='MIT',
    packages=find_packages(),
    package_data={
        'hyperOpt': [
            'tests/*',
            'examples/*/*',
            'visualization/*',
            'settings/*']
    },
    install_requires=[
        'docopt',
        'scipy',
        'scikit-learn',
        'pandas',
        'numpy', 
        'xgboost'
    ],
)