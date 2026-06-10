table S1, SMILES Naked 
table S3 sheet column smiles
table S4 sheet column smiles!

conda env create -f 10062026/environment.yml
conda activate chempa-convert
conda run -n chempa-convert python 10062026/convert_to_smiles.py
conda run -n chempa-convert python 10062026/convert_to_smiles.py --output-dir 10062026/my_outputs
conda run -n chempa-convert python 10062026/match_tables_to_amines.py
