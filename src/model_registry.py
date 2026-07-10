from pathlib import Path

class ModelRegistry:
    def __init__(self,root="models"):
        self.root=Path(root)
        self.root.mkdir(exist_ok=True)

    def path(self,name):
        return self.root/f"{name}.pkl"
