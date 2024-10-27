from setuptools import setup, find_packages

setup(
    name="speaksense",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "transformers>=4.46.0",
        "mediapipe>=0.10.14",
        "numpy>=2.1.2",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "librosa>=0.10.0",
        "sounddevice>=0.5.1"
    ],
    author="Salo Soja Edwin",
    author_email="salosoja@gmail.com",
    description="A multimodal deep learning system for detecting VA-directed speech",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Dawn-Of-Justice/SpeakSense",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)