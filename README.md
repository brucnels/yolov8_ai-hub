Attempt to quantize a custom yolov8s pt weight file

Trained with the train.py script
- Generates trained pt file through ultralytics train method
- output best.pt is already generated, but can be regenerated through this script

quantize_detector.py
- modified for default to yolov8 
- added ckpt_name to call the trained best.pt file

You can run through
```
python3 quantize_detector.py
```

Full error
```
Traceback (most recent call last):
  File "/local/mnt/workspace/brucnels/AIMET/yolov8_ai-hub/quantize_detector.py", line 74, in <module>
    model = model_cls.from_pretrained(ckpt_name=__location__+"/best.pt", aimet_encodings=None, include_postprocessing=True)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/qai_hub_models/models/yolov8_det_quantized/model.py", line 68, in from_pretrained
    equalize_model(model, input_shape)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/cross_layer_equalization.py", line 859, in equalize_model
    folded_pairs = fold_all_batch_norms(model, input_shapes, dummy_input)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/batch_norm_fold.py", line 496, in fold_all_batch_norms_to_weight
    connected_graph = ConnectedGraph(model, inp_tensor_list)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/meta/connectedgraph.py", line 158, in __init__
    self._construct_graph(model, model_input)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/meta/connectedgraph.py", line 299, in _construct_graph
    module_tensor_shapes_map = ConnectedGraph._generate_module_tensor_shapes_lookup_table(model, model_input)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/meta/connectedgraph.py", line 288, in _generate_module_tensor_shapes_lookup_table
    run_hook_for_layers_with_given_input(model, model_input, forward_hook, leaf_node_only=False)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/utils.py", line 343, in run_hook_for_layers_with_given_input
    _ = model(*input_tensor)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 658, in call_wrapped
    return self._wrapped_call(self, *args, **kwargs)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 277, in __call__
    raise e
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 267, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1212, in _call_impl
    result = forward_call(*input, **kwargs)
  File "<eval_with_key>.1", line 257, in forward
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1212, in _call_impl
    result = forward_call(*input, **kwargs)
  File "/local/mnt/workspace/brucnels/AIMET/.env/lib/python3.10/site-packages/aimet_torch/elementwise_ops.py", line 61, in forward
    return functional(*args, **kwargs)
RuntimeError: The size of tensor a (9261) must match the size of tensor b (8400) at non-singleton dimension 2
```