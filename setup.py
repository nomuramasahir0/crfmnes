# from distutils.core import setup
from setuptools import setup

# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="crfmnes",
      long_description=long_description,
      version="0.0.3",
      description="CR-FM-NES " +
                  "for numerical optimization in Python",
      author="Masahiro Nomura",
      author_email="masahironomura5325@gmail.com",
      maintainer="Masahiro Nomura",
      maintainer_email="masahironomura5325@gmail.com",
      url="https://github.com/nmasahiro/crfmnes",
      license="MIT",
      classifiers = [
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
      ],
      keywords=["optimization", "CR-FM-NES"],
      packages=["crfmnes"],
      requires=["numpy"],
      package_data={'': ['LICENSE']},
      )