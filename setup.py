from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements.
    '''
    with open(file_path) as file_obj:
        # A more efficient way to read and clean requirements
        requirements = [req.strip() for req in file_obj.readlines()]
    
    # Remove the '-e .' if it exists
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='Chicken-Disease-Classification',
    version='0.0.1',
    author='wuraola mathew',
    author_email='arena6663@gmail.com',
    
    # These are the two missing/incorrect lines
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    
    install_requires=get_requirements('requirements.txt')
)