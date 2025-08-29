import os
import subprocess
import shutil

# 1️⃣ Make folder for docs
docs_dir = "./k8s_docs"
os.makedirs(docs_dir, exist_ok=True)

# 2️⃣ Clone the GitHub repo (sparse checkout for docs)
repo_url = "https://github.com/kubernetes/website.git"
clone_dir = "./website"

# Clone with depth=1 and sparse
subprocess.run([
    "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, clone_dir
], check=True)

# Set sparse-checkout to content/en/docs
subprocess.run(["git", "-C", clone_dir, "sparse-checkout", "set", "content/en/docs"], check=True)

# 3️⃣ Move docs to ./k8s_docs
source_docs = os.path.join(clone_dir, "content", "en", "docs")
for item in os.listdir(source_docs):
    src_path = os.path.join(source_docs, item)
    dst_path = os.path.join(docs_dir, item)
    if os.path.isdir(src_path):
        shutil.move(src_path, dst_path)
    else:
        shutil.move(src_path, dst_path)

# 4️⃣ Clean up
shutil.rmtree(clone_dir)

print(f"Kubernetes docs downloaded to {docs_dir}")
