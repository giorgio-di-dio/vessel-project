import kagglehub
import os

data_dir = "data"

if not os.path.exists(data_dir):
    print("Download del dataset in corso...")
    path = kagglehub.dataset_download("nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation")
    print(f"Dataset scaricato in: {path}")
else:
    print("Dataset già presente, salto il download.")