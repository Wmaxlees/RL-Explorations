# RL Explorations

There are two long-standing difficulties that Reinforcement Learning (RL) faces:

1. Sample Efficiency: Training RL algorithms generally takes many environment interactions to learn a useful policy.
1. Generalization: Agents learn policies that are very brittle to the environment and don't generalize well outside of that.

These two issues are important bottlenecks that hold RL back from seeing much real world usage. This repo is a place to
hold code for experiments I do on understanding and improving these two aspects of RL.

## Experiments
### The Bard
An RLAIF trained model using Gemma3-1b as a base and Gemma3 in Ollama as a critic. The model is trained to output the best
poem, as judged by Gemma3, given a noun.

### JEPA-RL
Yan LaCun et al. have recently released several papers around an embedding architecture they call Joint Embedding
Predictive Architectures (JEPA) [paper](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/).
These models, as claimed by LaCun, are supposed to better capture the important underlying structure within the
input data than some other similar architectures (VAE, etc.).

The JEPA-RL experiments revolve around placing JEPAs in place of where previous self-supervised algorithms would
be used to see whether there is any improvement.
