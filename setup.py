from setuptools import setup

setup(name='smallsat_sim',
      version='1.0.0',
      install_requires=[
          'casadi',
          'numpy',
          'pyyaml',
          'mujoco',
          'mujoco-mjx'
      ],
      )