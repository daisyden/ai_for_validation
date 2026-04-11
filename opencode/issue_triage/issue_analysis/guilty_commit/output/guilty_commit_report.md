# Guilty Commit Analysis Report

Generated: 2026-04-10 08:34:07

## Test Information:
  - test_file: test_ops_xpu.py
  - test_name: test_compare_cpu_addcmul
  - test_cases: [{'test_path': 'test_ops_xpu.py', 'test_name': 'test_compare_cpu_addcmul'}]
  - error_message: AssertionError: Tensor-likes are not close
  - submit_time: 2026-04-10 08:34:06.801791
  - test_type: eager

## Test Type Analysis:
  - Detected test type: eager
  - Note: Excluding purely inductor/dynamo related commits
  - Focus on: core operator implementations, test definitions, OpInfo

## Commit Range: HEAD~50 to HEAD

## pytorch/pytorch Commits
  - 546d1c5db5c: [inductor] Add tensor value/alpha samples to inductor opinfo tests (#176874)
  - 46ba12aa808: [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition (#176871)
  - c21a3a53fd3: [inductor] Add singletensor capturable optimizers to bitwise tests (#176807)
  - d98da60ba27: [dynamo] Route addcdiv_ decomposition through add_ with alpha (#176806)
  - 079d7f050fb: [dynamo] Route addcmul_ decomposition through add_ with alpha (#176805)
  - be7dbd8a340: [inductor] Add lerp decompositions for bitwise parity with eager (#176804)
  - c38518f47ed: [inductor] Make addcmul/addcdiv decomp skip unconditional and add another decomp (#175839)
  - b87b807f4fd: [inductor] Add _foreach_addcdiv lowering to match _foreach_addcmul (#176237)
  - 0c157745f23: [inductor] Skip addcmul decomposition to enable FMA lowering (#175309)
  - 7bf8db3d905: use fma in addcmul when possible (#172750)

## intel/torch-xpu-ops Commits
  - 02e5eaf7: addcmul | Early exit if inputs are on different devices (CPU and XPU) (#2985)
  - 610b6de7: Add aten::addcdiv.Tensor and aten::addcmul.Tensor (#1073)
  - 45c2d8f0: Add aten::addcmul (#418)
  - 617bba99: Add aten::_foreach_addcmul/_foreach_addcdiv and their variants (#376)
  - 682d0e4f: Add aten::addcdiv and its variants (#486)
  - e635bee8: Extended `test_ops_xpu`: update tolerances (#3222)
  - ea43d64f: Add large tensor test decorator (#3228)
  - ba7f025c: [SYCLTLA] Upgrade SYCLTLA to v0.8 (#3207)
  - 8b9259ea: Explicitly skip cudnn_rnn/miopen_rnn tests (#3113)
  - 370b0852: Add ProcessGroupXCCL functionality to pass test_c10d_xccl.py (#3171)

## Detailed Analysis

### Commit 46ba12aa808
Date: 2026-03-10 05:18:53 +0000 Michael Lazos
Message: [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition (#176871)
Files changed: commit 46ba12aa808c11b583db06dec1e0b8331c1911b9, Author: Michael Lazos <mlazos@meta.com>, Date:   Mon Mar 9 13:49:40 2026 -0700,     [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition (#176871),     
  - core_operator_related: True
  - test_related: False
  - opinfo_related: False
  - inductor_related: True
  - dynamo_related: True

### Commit d98da60ba27
Date: 2026-03-10 05:18:53 +0000 Michael Lazos
Message: [dynamo] Route addcdiv_ decomposition through add_ with alpha (#176806)
Files changed: commit d98da60ba27b0822a625d48d8bb61e10bac153ef, Author: Michael Lazos <mlazos@meta.com>, Date:   Mon Mar 9 13:49:39 2026 -0700,     [dynamo] Route addcdiv_ decomposition through add_ with alpha (#176806),     
  - core_operator_related: True
  - test_related: False
  - opinfo_related: False
  - inductor_related: True
  - dynamo_related: True

### Commit be7dbd8a340
Date: 2026-03-10 05:18:53 +0000 Michael Lazos
Message: [inductor] Add lerp decompositions for bitwise parity with eager (#176804)
Files changed: commit be7dbd8a34053fb307194b3504c73be5b59f001a, Author: Michael Lazos <mlazos@meta.com>, Date:   Mon Mar 9 13:49:37 2026 -0700,     [inductor] Add lerp decompositions for bitwise parity with eager (#176804),     
  - core_operator_related: True
  - test_related: False
  - opinfo_related: False
  - inductor_related: True
  - dynamo_related: True

### Commit 7bf8db3d905
Date: 2026-01-24 00:46:08 +0000 Natalia Gimelshein
Message: use fma in addcmul when possible (#172750)
Files changed: commit 7bf8db3d9059d09140d32ceb0301efe28ad4885d, Author: Natalia Gimelshein <ngimel@meta.com>, Date:   Sat Jan 24 00:46:08 2026 +0000,     use fma in addcmul when possible (#172750),     
  - core_operator_related: True
  - test_related: True
  - opinfo_related: False
  - inductor_related: True
  - dynamo_related: True