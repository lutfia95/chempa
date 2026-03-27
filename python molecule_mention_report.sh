python molecule_mention_report.py --input primary_aliphatic_amines.tsv --out-prefix ./primary_scan/primary_aliphatic_amines --workers 30 
python molecule_mention_report.py --input secondary_aliphatic_amines.tsv --out-prefix ./secondary_aliphatic_amines/secondary_aliphatic_amines --workers 30
python molecule_mention_report.py --input aromatic_amines.tsv --out-prefix ./aromatic_amines/aromatic_amines --workers 30
python molecule_mention_report.py --input others.tsv --out-prefix ./others/others --workers 30