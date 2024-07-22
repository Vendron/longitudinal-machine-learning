
# Install Python 3.9 and virtual environment
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev

# Create virtual environment
python3.9 -m venv venv

source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Run 'source venv/bin/activate' to activate the virtual environment."