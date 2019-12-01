try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'rsfmodel',
    'author': 'John R. Leeman, Ryan May',
    'url': 'http://github.com/jrleeman/rsfmodel',
    'download_url': 'http://github.com/jrleeman/rsfmodel',
    'author_email': 'kd5wxb@gmail.com',
    'version': '0.2',
    'install_requires': ['nose'],
    'packages': ['rsfmodel'],
    'scripts': [],
    'name': 'rsfmodel'
}

setup(**config)
