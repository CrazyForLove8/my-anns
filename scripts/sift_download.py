import os 
import tarfile
from urllib.request import urlretrieve

url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
path = os.path.join("datasets", "siftsmall.tar.gz")
if os.path.exists(path):
    print(f"{path} already exists")
else:
    print(f"Downloading {url} to {path}")
    os.makedirs("datasets", exist_ok=True)
    urlretrieve(url, path)
with tarfile.open(path) as f:
    f.extractall("datasets")
os.remove(path)