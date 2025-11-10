# alfa
```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
pip install flashrag-dev[full]

cd ..
pip install "sentence-transformers" "bm25s"

cd /home
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

cd /project_name
# CPU-only version
#conda install -c pytorch faiss-cpu=1.8.0
# GPU(+CPU) version
#conda install -c pytorch -c nvidia faiss-gpu=1.8.0

conda create -n rag312 -c conda-forge -c pytorch python=3.12 faiss-cpu=1.8.0
conda activate rag312
```