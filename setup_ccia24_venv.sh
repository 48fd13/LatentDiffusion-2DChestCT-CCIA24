### create venv
conda create -n ccia24 -c conda-forge python=3.9
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate ccia24
echo $(which pip) # this should be /home/ubuntu/anaconda3/envs/ccia24
pip install --no-cache-dir -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchmetrics==1.3.0.post0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_fid==0.3.0
pip install accelerate==0.26.1
conda deactivate
echo "ccia24 conda env created !"
#conda remove -n ccia24 --all
