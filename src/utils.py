from pathlib import Path

def create_directories():
    folders=[
        'outputs',
        'outputs/figures',
        'outputs/reports',
        'outputs/submissions',
        'models',
        'data/processed'
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True,exist_ok=True)
