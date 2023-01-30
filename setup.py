from setuptools import setup, find_packages

setup(
    name='monai-tutorial',
    version='0.1.0',
    description='MONAI tutorial',
    author='MONAI team / Julien Cohen-Adad',
    author_email='',
    url='',
    packages=find_packages(include=['exampleproject', 'exampleproject.*']),
    install_requires=[
        'monai',
    ],
    extras_require={},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    entry_points={},
    package_data={}
)

# setup.py template from: https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/