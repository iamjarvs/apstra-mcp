from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="streamlit-app",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Streamlit application template",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repo-name",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "streamlit-app=app.main:main",
        ],
    },
)