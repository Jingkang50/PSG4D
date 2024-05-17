# Training Relation Head
The relation head follows the implementation in PVSG. When the feature tubes are available, we turn to OpenPVSG for relation modeling.

```
# Train IPS
sh scripts/train/train_ips.sh
# Tracking and save query features
sh scripts/utils/prepare_qf_ips.sh
# Prepare for relation modeling
sh scripts/utils/prepare_rel_set.sh
# Train relation models
sh scripts/train/train_relation.sh
# Test
sh scripts/test/test_relation_full.sh
```
