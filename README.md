
# BioUnlearn



**BioUnlearn: Domain-Adaptive Unlearning for Clinical LLMs**  

Targeting Nature Communications



## Structure



bio-unlearning/

├── configs/            # Master config (API keys, hyperparams)

├── stage_a_umls/       # UMLS graph traversal → per-concept JSON files

├── stage_b_instances/  # LLM oracle instance generation (PAC / RGU / IFE)

├── stage_c_validation/ # Human spot-check tooling and IAA scoring

├── data/

│   ├── raw/            # Silver-labeled instances before filtering

│   ├── filtered/       # Instances that passed all 4 automated filters

│   ├── validated/      # 400-instance human-validated subset

│   └── splits/         # Final train/val/test splits (frozen)

└── logs/               # Run logs



## Branch



Active development: `saisab`

