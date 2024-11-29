from setuptools import setup, find_packages

setup(
    name='MDRMF',
    version='0.0.12',
    packages=find_packages(),
    description='Multidrug Resistance Machine Fishing',
    author='Jacob Molin Nielsen',
    author_email='jacob.molin@me.com',
    url='https://github.com/MolinDiscovery/MDRMF',  # use the URL to the github repo
    download_url='https://github.com/MolinDiscovery/MDRMF/archive/v0.0.12.tar.gz',
    keywords=['machine fishing', 'drug discovery', 'machine learning', 'pool based active learning'],
    classifiers=[],
    include_package_data=True,
    package_data={
        'MDRMF': ['schemas/*.yaml'],
    },    
)