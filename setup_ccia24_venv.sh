### create venv
conda create -n ccia24 -c conda-forge python=3.10 -y
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate ccia24
echo $(which pip) # this should be /home/ubuntu/anaconda3/envs/ccia24
pip install --no-cache-dir -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch_fid==0.3.0
pip install accelerate==0.26.1

# to run inference_app.py:
pip install flask

conda deactivate
echo "ccia24 conda env created !"
#conda remove -n ccia24 --all

export PATH="~/.conda/envs/ccia24/bin:$PATH"
