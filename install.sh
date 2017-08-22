# Take out any existing huge amounts of data before re-installing
mv DataEngineering/dataset/downloadedDataset ..
sudo pip install -r requirements.txt
sudo pip install . --upgrade
# Put back any existing downloaded data
mv ../downloadedDataset DataEngineering/dataset/.
python tests/testImport.py
