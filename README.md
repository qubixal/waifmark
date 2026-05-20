
# Waifmark-1
A benchmark testing local agentic capabilities and speech persona of small (V)LLMs.

Due to my personal hardware limitations, only models < ~9b are tested.\
I apologise for any inconviences.

Testing procedure and exact results will not be published either unfortunately due to the underbaked benchmarking procedure and tools and rough data organisation.

### What is tested:
- Personality / Maintaining persona in responses
- Basic local agentic tool usage and planning
- Basic reasoning and logic
- English / Chinese fluency (and ability to maintain persona)
- Hallucination checks
- Short-term memory retention
- Safeguard guidelines

### Current Leaderboard and notes:
| Place | Score | Lab/Model | Notes |
|-|-|-|-|
|1|86.3|qwen/qwen3.5 Q4|overall well-scoring|
|2|74.3|qwen/qwen3-2507 instruct Q4|has worse personality in english|
|3|68.8|google/gemma4-e4b Q8|fails hallucination|
|4|67|qwen/qwen3-vl-8b Q4|inconsistent logic|
|5|61|qwen/qwen3-2507 instruct Q8|worse than Q4, bad logic|
|6|60|deepseek/r1-distill-0528-qwen Q8|overthinks|
|7|55|google/gemma4-e4b Q4|fast but long thinking|
|8|54|google/gemma4-e2b Q8|dumber a4b|

no other model was able to achieve a passing mark of ≥50.\
Qwen currently holds 4 models in top 5, Google holds 1.