from setuptools import setup, find_packages


setup(
    name='learn_images',
    version='0.0.2',
    description='Module containing my experiments',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=['scripts']),
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas', "torch"],
    author='Osman Bayram',
    author_email="osmanfbayram@gmail.com",
)
