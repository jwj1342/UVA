# Data

| file | content |
|---|---|
| `swesmith_train_ids.txt` | 499 SWE-smith instances: the training-task pool (collection, replay, training) |
| `swesmith_heldout_ids.txt` | 200 SWE-smith instances, instance-disjoint from the training pool (development) |
| `verified_500_order.txt` | the 500 SWE-bench Verified instances in the fixed order used for slicing |
| `verified_rescue_pool_ids.txt` | the 127 Verified instances the base student failed on its first solo attempt |

`python -m uva.data.make_pool` turns an id list into the JSON Lines records the harness reads
(`instance_id, repo, image_name, problem_statement, FAIL_TO_PASS, PASS_TO_PASS, base_commit`). The
SWE-smith split is instance-disjoint; repositories can appear in both halves. The consultation pairs
are released separately.
