import json
import subprocess
import sys
from pathlib import Path

import requests

assert len(sys.argv) == 2, "Uploader needs heibox api token"
heibox_token = sys.argv[1]

url = "https://heibox.uni-heidelberg.de/api/v2.1/via-repo-token/upload-link/?path=%2F&from=api&replace=1"

headers = {"accept": "application/json", "authorization": f"Bearer {heibox_token}"}
response = requests.get(url, headers=headers)
upload_url = response.text.strip('"')
assert len(upload_url) > 50, f"Failed to get upload url: {upload_url}"
print("Obtained upload url")

to_upload = None
for to_upload in Path("installer").glob("PlantSeg*"):
    print("Uploading ", to_upload)
    with open(to_upload, "rb") as fp:
        files = {"file": (to_upload.name, fp, "application/octet-stream")}
        payload = {"parent_dir": "/", "replace": "1"}
        headers = {"accept": "application/json", "authorization": "Bearer asdf"}

        response = requests.post(upload_url, data=payload, files=files, headers=headers)

print("Uploaded installer, ", response.text)

# attach to release
if to_upload:
    output = subprocess.run(
        ["gh", "release", "list", "-L", "1", "--json", "tagName"],
        capture_output=True,
        text=True,
    ).stdout
    tag = json.loads(output)[0]["tagName"]
    up_out = subprocess.run(
        ["gh", "release", "upload", tag, f"installer/{to_upload.name}"],
        capture_output=True,
        text=True,
    )
    print(
        f"Attached installer {to_upload.name} to Release {tag}!\n"
        f"Output: {up_out.stdout}, {up_out.stderr}"
    )
else:
    print("Could not upload anything!")
    print(list(Path("installer").glob("PlantSeg*")))
