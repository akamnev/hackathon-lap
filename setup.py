import setuptools


INSTALL_REQUIRES = []
EXTRAS_REQUIRE = {
    'dev': [],
}

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='hack-lap',
    version='1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.6',
)
