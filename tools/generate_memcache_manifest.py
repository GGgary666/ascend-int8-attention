#!/usr/bin/env python3
"""Freeze a previously hashed model and numerical scales for the paired patch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_sparse_attn_sharedkv_int8.memcache_manifest import make_manifest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-content-manifest',type=Path,required=True)
parser.add_argument('--connector-contract',type=Path,required=True)
parser.add_argument('--scales',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('cache_compatibility',args.connector_contract)
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
manifest=make_manifest(json.loads(args.model_content_manifest.read_text()),
                       torch.load(args.scales,map_location='cpu',weights_only=True)['scales'],contract)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ('weight_manifest','metadata_files')}))
