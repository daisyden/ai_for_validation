# Benchmark Models Reference

This file contains the list of benchmark models used in torch-xpu-ops CI testing.

## Source
- https://github.com/intel/torch-xpu-ops/blob/main/.ci/benchmarks/

## HuggingFace Models (65 models)
```
AlbertForMaskedLM, AlbertForQuestionAnswering, AllenaiLongformerBase,
BartForCausalLM, BartForConditionalGeneration, BartForSmallForCausalLM,
BartForSmallForConditionalGeneration, BlenderbotForCausalLM,
BlenderbotForConditionalGeneration, BlenderbotSmallForCausalLM,
BlenderbotSmallForConditionalGeneration, CamemBert, DebertaV2ForMaskedLM,
DebertaV2ForQuestionAnswering, DistilBertForMaskedLM,
DistilBertForQuestionAnswering, DistillGPT2, ElectraForCausalLM,
ElectraForQuestionAnswering, GoogleFnet, google/gemma-2-2b, google/gemma-3-4b-it,
GPT2ForSequenceClassification, GPTJForCausalLM, GPTJForQuestionAnswering,
GPTNeoForCausalLM, GPTNeoForSequenceClassification, LayoutLMForMaskedLM,
LayoutLMForSequenceClassification, M2M100ForConditionalGeneration,
MBartForCausalLM, MBartForConditionalGeneration, MegatronBertForCausalLM,
MegatronBertForQuestionAnswering, meta-llama/Llama-3.2-1B,
mistralai/Mistral-7B-Instruct-v0.3, MobileBertForMaskedLM,
MobileBertForQuestionAnswering, MT5ForConditionalGeneration,
openai/gpt-oss-20b, openai/whisper-tiny, OPTForCausalLM, PegasusForCausalLM,
PegasusForConditionalGeneration, PLBartForCausalLM,
PLBartForConditionalGeneration, Qwen/Qwen3-0.6B, RobertaForCausalLM,
RobertaForQuestionAnswering, T5ForConditionalGeneration, T5Small,
TrOCRForCausalLM, XGLMForCausalLM, XLNetLMHeadModel, YituTechConvBert
```

## Timm Models (88 models)
```
adv_inception_v3, beit_base_patch16_224, botnet26t_256, cait_m36_384,
coat_lite_mini, convit_base, convmixer_768_32, convnext_base,
convnextv2_nano.fcmae_ft_in22k_in1k, crossvit_9_240, cspdarknet53,
deit_base_distilled_patch16_224, deit_tiny_patch16_224, dla102,
dm_nfnet_f0, dpn107, eca_botnext26ts_256, eca_halonext26ts,
ese_vovnet19b_dw, fbnetc_100, fbnetv3_b, gernet_l, ghostnet_100,
gluon_inception_v3, gmixer_24_224, gmlp_s16_224, hrnet_w18,
inception_v3, jx_nest_base, lcnet_050, levit_128, mixer_b16_224,
mixnet_l, mnasnet_100, mobilenetv2_100, mobilenetv3_large_100,
mobilevit_s, nfnet_l0, pit_b_224, pnasnet5large, poolformer_m36,
regnety_002, repvgg_a2, res2net101_26w_4s, res2net50_14w_8s, resnext50,
resmlp_12_224, resnest101e, rexnet_100, sebotnet33ts_256,
selecsls42b, spnasnet_100, swin_base_patch4_window7_224,
swsl_resnext101_32x16d, tf_efficientnet_b0, tf_mixnet_l, tinynet_a,
tnt_s_patch16_224, twins_pcpvt_base, visformer_small,
vit_base_patch14_dinov2.lvd142m, vit_base_patch16_224,
vit_base_patch16_siglip_256, volo_d1_224, xcit_large_24_p8_224
```

## TorchBench Models (18 models)
```
BERT_pytorch, Background_Matting, LearningToPaint, alexnet,
dcgan, densenet121, mnasnet1_0, mobilenet_v2, mobilenet_v3_large,
nvidia_deeprecommender, pytorch_unet, resnet18, resnet50,
resnext50_32x4d, shufflenet_v2_x1_0, squeezenet1_1, vgg16
```

## Model Pattern Matchers

### HuggingFace Pattern
```python
import re
HF_PATTERNS = [
    r"Albert[A-Z]\w*", r"Bart[A-Z]\w*", r"Bert[A-Z]\w*",
    r"Blenderbot[A-Z]\w*", r"CamemBert[A-Z]\w*", r"Deberta[A-Z]\w*",
    r"Distil[A-Z]\w*", r"Distill[A-Z]\w*", r"Electra[A-Z]\w*",
    r"GPT2[A-Z]\w*", r"GPTJ[A-Z]\w*", r"GPTNeo[A-Z]\w*",
    r"LayoutLM[A-Z]\w*", r"LLama[A-Z]\w*", r"Megatron[A-Z]\w*",
    r"Mistral[A-Z]\w*", r"Mobile[A-Z]\w*", r"Megatron[A-Z]\w*",
    r"OPT[A-Z]\w*", r"Roberta[A-Z]\w*", r"T5[A-Z]\w*",
    r"google/gemma", r"meta-llama/Llama", r"mistralai/Mistral",
    r"openai/gpt", r"openai/whisper", r"Qwen/AwQ", r"XGLM",
    r"XLNet"
]
```

### Timm Pattern
```python
TIMM_PATTERNS = [
    r"adv_inception", r"beit", r"botnet", r"cait", r"coat",
    r"convit", r"convmixer", r"convnext", r"crossvit", r"cspdarknet",
    r"deit", r"dla", r"dm_nfnet", r"dpn", r"eca_",
    r"ese_vovnet", r"fbnet", r"gernet", r"ghostnet", r"gluon_",
    r"gmixer", r"gmlp", r"hrnet", r"inception", r"jx_nest",
    r"lcnet", r"levit", r"mixer_b", r"mixnet", r"mnasnet",
    r"mobilenet", r"mobilevit", r"nfnet", r"pit_",
    r"pnasnet", r"poolformer", r"regnety", r"repvgg",
    r"res2net", r"resmlp", r"resnest", r"rexnet", r"sebotnet",
    r"selecsls", r"spnasnet", r"swin", r"swsl_resnext",
    r"tf_efficientnet", r"tf_mixnet", r"tinynet", r"tnt_",
    r"twins", r"visformer", r"vit_", r"volo", r"xcit",
]
```

### TorchBench Pattern
```python
TORCHBENCH_PATTERNS = [
    r"BERT_pytorch", r"Background_Matting", r"LearningToPaint",
    r"alexnet", r"dcgan", r"densenet", r"mnasnet",
    r"mobilenet_v", r"nvidia_deeprecommender", r"pytorch_unet",
    r"resnet\d*", r"resnext", r"shufflenet", r"squeezenet", r"vgg"
]
```

## Usage in Priority Analysis

Use these patterns to distinguish **E2E Benchmark** issues from **Custom Model** issues:

```python
def is_benchmark_model(text: str) -> bool:
    """
    Check if issue involves benchmark models.
    Returns True if benchmark model pattern found.
    """
    all_patterns = HF_PATTERNS + TIMM_PATTERNS + TORCHBENCH_PATTERNS
    combined = "|".join(f"({p})" for p in all_patterns)
    return bool(re.search(combined, text, re.IGNORECASE))

def is_custom_model(text: str) -> bool:
    """
    Check if issue involves custom models (NOT benchmark).
    """
    has_model_context = any(kw in text.lower() for kw in [
        "production", "customer", "custom", "enterprise",
        "deployment", "our model", "internal"
    ])
    return has_model_context and not is_benchmark_model(text)
```

## Quick Lookup Sets

```python
BENCHMARK_MODELS_SET = {
    # HuggingFace large models
    "gemma", "llama", "mistral", "qwen", "opt", "gpt",
    # Timm models
    "resnet", "vit", "efficientnet", "convnext", "swin",
    "mobilevit", "mnasnet", "regnet",
    # TorchBench
    "bert_pytorch", "resnet18", "resnet50", "densenet",
    "vgg", "alexnet", "mobilenet_v2"
}
```

## Skill Integration

Import this reference in SKILL_Priority_Analysis.md:

```python
# After importing benchmark models
from skills.triage_skills.BENCHMARK_MODELS import (
    BENCHMARK_MODELS_SET,
    is_benchmark_model,
    is_custom_model
)

# Use in custom model detection
if is_custom_model(issue_body) and not is_benchmark_model(issue_body):
    # This is a P0 custom model issue
    priority_boost = 1
```