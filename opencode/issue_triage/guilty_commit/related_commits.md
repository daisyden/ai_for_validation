# Related addcmul Commits

## pytorch/pytorch Repository

| # | Commit Hash | Description |
|---|-------------|-------------|
| 1 | da267220647 | Update torch-xpu-ops commit pin (#177238) |
| 2 | 546d1c5db5c | [inductor] Add tensor value/alpha samples to inductor opinfo tests (#176874) |
| 3 | 46ba12aa808 | [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition (#176871) |
| 4 | c21a3a53fd3 | [inductor] Add singletensor capturable optimizers to bitwise tests (#176807) |
| 5 | d98da60ba27 | [dynamo] Route addcdiv_ decomposition through add_ with alpha (#176806) |
| 6 | 079d7f050fb | [dynamo] Route addcmul_ decomposition through add_ with alpha (#176805) |
| 7 | be7dbd8a340 | [inductor] Add lerp decompositions for bitwise parity with eager (#176804) |
| 8 | e5d66cc9284 | [inductor] Add tensor value/alpha samples to inductor opinfo tests |
| 9 | c13d2bdd5d9 | [inductor] Add singletensor capturable optimizers to bitwise tests |
| 10 | 151ce41e41c | [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition |
| 11 | 18d2acb1759 | [dynamo] Route addcdiv_ decomposition through add_ with alpha |
| 12 | 20b7f4cfec1 | [dynamo] Route addcmul_ decomposition through add_ with alpha |
| 13 | 02149ede676 | [inductor] Add lerp decompositions for bitwise parity with eager |
| 14 | db8ce2a22cf | [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition |
| 15 | 89c0dd34477 | [inductor] Add singletensor capturable optimizers to bitwise tests |
| 16 | 08fd3ac4a4b | [dynamo] Route addcdiv_ decomposition through add_ with alpha |
| 17 | 3a8babecc5a | [dynamo] Route addcmul_ decomposition through add_ with alpha |
| 18 | 9c6d5157f72 | [inductor] Add lerp decompositions for bitwise parity with eager |
| 19 | 150e63fd394 | [dynamo] Use inductor_prims.fma for addcmul_ value=1 decomposition |
| 20 | befdfda26cc | [inductor] Add singletensor capturable optimizers to bitwise tests |
| 21 | 771b6b9b3a8 | [dynamo] Route addcdiv_ decomposition through add_ with alpha |
| 22 | b31bb05c27c | [dynamo] Route addcmul_ decomposition through add_ with alpha |
| 23 | 6a1b5d5d8b4 | [inductor] Add lerp decompositions for bitwise parity with eager |
| 24 | 69afb56f596 | [dynamo] Use inductor_prims.fma for tensor alpha in method_add_ |
| 25 | 7288fd5d332 | [inductor] Add singletensor capturable optimizers to bitwise tests |
| 26 | 2726a29ad6e | [dynamo] Route addcdiv_ decomposition through add_ with alpha |
| 27 | 33fba59e0b3 | [dynamo] Route addcmul_ decomposition through add_ with alpha |
| 28 | 3e28fce313a | [inductor] Add lerp decompositions for bitwise parity with eager |
| 29 | 34eacc61f22 | [inductor] Replace lerp_scalar lowering with decomposition into add(alpha=weight) |
| 30 | aa95629cc26 | [dynamo] Fix add_ and addcdiv_ decompositions for bitwise parity |
| 31 | f57abf78553 | [dynamo] Route addcmul_/addcdiv_ decompositions through add_ with alpha |
| 32 | 13e2970907d | [inductor] Add lerp lowering and fix addcmul_ decomposition for bitwise parity |
| 33 | a2c6e4207b1 | [inductor] Add singletensor capturable optimizers to bitwise tests |
| 34 | c38518f47ed | [inductor] Make addcmul/addcdiv decomp skip unconditional and add another decomp (#175839) |
| 35 | b87b807f4fd | [inductor] Add _foreach_addcdiv lowering to match _foreach_addcmul (#176237) |
| 36 | ff6784b064a | [inductor] Make addcmul/addcdiv decomp skip unconditional |
| 37 | ac1c9d0fa32 | [inductor] Add _foreach_addcdiv lowering to match _foreach_addcmul |

## intel/torch-xpu-ops Repository

| # | Commit Hash | Description |
|---|-------------|-------------|
| 1 | a83ca3dc | Use std::fma in addcmul and foreach pointwise ops for FMA parity with CUDA |
| 2 | 02e5eaf7 | addcmul \| Early exit if inputs are on different devices (CPU and XPU) (#2985) |
| 3 | a0c260f7 | Add aten::addcdiv.Tensor and aten::addcmul.Tensor (#1073) |
| 4 | 610b6de7 | Add aten::addcdiv.Tensor and aten::addcmul.Tensor (#1073) |
| 5 | 45c2d8f0 | Add aten::addcmul (#418) |
| 6 | 617bba99 | Add aten::_foreach_addcmul/_foreach_addcdiv and their variants (#376) |

## Summary

- **pytorch/pytorch**: 37 addcmul-related commits since early 2026
- **intel/torch-xpu-ops**: 6 addcmul-related commits

## Most Relevant Commits for test_compare_cpu_addcmul_xpu failures

The following commits in pytorch/pytorch are the most likely candidates for causing the test failures:

1. **46ba12aa808** - [dynamo] Use inductor_prims.fma for addcmul_/addcdiv_ tensor value decomposition (#176871)
2. **079d7f050fb** - [dynamo] Route addcmul_ decomposition through add_ with alpha (#176805)
3. **546d1c5db5c** - [inductor] Add tensor value/alpha samples to inductor opinfo tests (#176874)

In intel/torch-xpu-ops:

1. **a83ca3dc** - Use std::fma in addcmul and foreach pointwise ops for FMA parity with CUDA
2. **02e5eaf7** - addcmul | Early exit if inputs are on different devices (CPU and XPU) (#2985)