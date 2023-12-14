from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', encoding='utf-16') as req:
        return req.read().splitlines()

setup(
    name='FetchRewardsDashboard',
    version='1.0.1',
    description='A dashboard for Fetch Rewards data analysis',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
        'fetch_dashboard=Fetch_Dash.main:run'
        ]
    }
)