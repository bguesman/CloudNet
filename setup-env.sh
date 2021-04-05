# To be safe, remove existing environment
rm -rf cloud-net-env

# Create environment and use it
python3 -m venv cloud-net-env
source cloud-net-env/bin/activate

# Install requirements list
pip install -r requirements.txt
