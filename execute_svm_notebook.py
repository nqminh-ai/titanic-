import nbformat
from nbclient import NotebookClient
import time

print("Starting execution of members/svm/03_svm.ipynb...", flush=True)
t0 = time.time()
with open('members/svm/03_svm.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open('members/svm/03_svm.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook executed and saved successfully in {time.time()-t0:.2f} seconds!", flush=True)
