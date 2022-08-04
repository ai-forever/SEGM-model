from setuptools import setup


with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name='segmmodel',
    packages=['segm'],
    install_requires=packages
)
