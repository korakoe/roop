from setuptools import setup, find_packages

setup(
    name="Roop",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="AGPL-3.0",
    description="Roop - Libified",
    author="Korakoe",
    long_description_content_type="text/markdown",
    url="https://github.com/korakoe/roop/tree/main",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformers",
        "attention mechanism",
        "text-to-image",
    ],
    install_requires=[
        "numpy==1.23.5",
        "opencv-python==4.7.0.72",
        "onnx==1.14.0",
        "insightface==0.7.3",
        "psutil==5.9.5",
        "tk==0.1.0",
        "pillow==9.5.0",
        "torch==2.0.1",
        "onnxruntime==1.15.0",
        "onnxruntime-gpu==1.15.0",
        "tensorflow==2.13.0rc1",
        "tensorflow==2.12.0",
        "protobuf==4.23.2",
        "tqdm==4.65.0",
        "moviepy==1.0.3",
        "codeformer-pip @ git+https://github.com/korakoe/codeformer-pip.git"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: AGPL-3.0 License",
        "Programming Language :: Python :: 3.9",
    ],
)
