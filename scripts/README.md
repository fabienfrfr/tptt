# Titanesque Scripts

```python
import requests, zipfile, io, shutil

repo_url = "https://github.com/fabienfrfr/tptt/archive/refs/heads/dev.zip"
resp = requests.get(repo_url)
z = zipfile.ZipFile(io.BytesIO(resp.content))
z.extractall("temp_repo")  # extrait tout dans temp_repo
shutil.move("temp_repo/tptt-dev/scripts", "./scripts") # download les scripts déjà fait
shutil.rmtree("temp_repo")

```