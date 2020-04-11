# Convert event_schema.json to get label map
python ner_crf/utils/data_process.py schema_role_process ./data/ori_datasets/event_schema.json ./data/datasets/vocab_roles_label_map.txt

# Create train/dev set from origin datasets
python ner_crf/utils/data_process.py origin_events_process ./data/ori_datasets/train.json ./data/datasets/train.json
python ner_crf/utils/data_process.py origin_events_process ./data/ori_datasets/dev.json ./data/datasets/dev.json