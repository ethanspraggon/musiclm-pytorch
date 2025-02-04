from setuptools import setup, find_packages

setup(
  name = 'musiclm-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.2',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text to music',
    'contrastive learning'
  ],
  install_requires=[
    'accelerate',
    'audiolm-pytorch>=0.17.0',
    'beartype',
    'einops>=0.6',
    'lion-pytorch',
    'vector-quantize-pytorch>=1.0.0',
    'x-clip',
    'torch>=1.12',
    'torchaudio',
    'fairseq',
    'transformers'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
