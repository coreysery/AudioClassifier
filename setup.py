try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='Deep DSP',
    version='0.1dev',
    packages=['DSP', 'deepdsp'],
    license='MIT',
    scripts=[],
    install_requires=[
        "tensorflow >= 0.11.0",
    ],
)