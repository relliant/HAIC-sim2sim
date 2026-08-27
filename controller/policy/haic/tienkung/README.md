# TienKung policy

Place the ONNX policy exported by HAIC here as `box_1.onnx`, or pass another
path with `python scripts/deploy_haic.py --task tienkung_box --policy ...`.
The sim2sim repository includes the matching 20-DoF model and retargeted
motion, but does not duplicate the trained HAIC checkpoint/export.
