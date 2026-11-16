# Does fine-tuning make a VLM less confidently wrong?

LoRA fine-tune of Qwen3-VL-2B on VSR, a spatial reasoning dataset, with my own
training loop and LoRA layer, on one 8 GB GPU. Work in progress.

## Results

### Rank ablation

| run | rank | trainable params | test accuracy | test ECE raw | test ECE scaled | minutes |
|---|---|---|---|---|---|---|
| r4_s0 | 4 | 802,816 | 0.8367 | 0.0387 | 0.0252 | 57.7 |
| r16_s0 | 16 | 3,211,264 | 0.8341 | 0.0196 | 0.0165 | 55.1 |
| r16_s1 | 16 | 3,211,264 | 0.8484 | 0.0456 | 0.0179 | 125.1 |
| r16_s2 | 16 | 3,211,264 | 0.8461 | 0.0488 | 0.0243 | 103.0 |
| r64_s0 | 64 | 12,845,056 | 0.8516 | 0.0512 | 0.0273 | 64.0 |

The three rank 16 rows differ only in seed. Differences between ranks that
are smaller than the spread between those three are noise.

The seed alone moved test accuracy from 0.834 to 0.848. Rank 4 (0.837) falls
inside that range and rank 64 (0.852) only 0.3 points above it, so rank made
no clear difference. Scaled ECE is between 0.017 and 0.027 for every run, and
raw ECE varies more with the seed (0.020 to 0.049) than with the rank. I kept
rank 16 as the main run. Rank 4 would have done as well with a quarter of the
trainable parameters. Rank 64 took 10 to 15% longer than ranks 4 and 16.
r16_s1 and r16_s2 took longer still, but they have the same settings as
r16_s0, so the extra time didn't come from the rank.
