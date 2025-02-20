# LuxAI S3 MARL environment

The `LuxAIMARLEnv` environment wraps the `LuxAIS3GymEnv` (from the [Lux-S3-Design](https://github.com/Lux-AI-Challenge/Lux-Design-S3/) repository) and offers a Gymnasium-compliant API to enable multi-agent training of a single LuxAI S3 player.

In the LuxAI Challenge Season 3 competition, two players are pitted against each other in a classic RTS game setting, both controlling a set of units that are tasked with collecting 'relic points' from hidden tiles, while trying to stop the opponent from doing the same.

One problem with the competition environment (`LuxAIS3GymEnv`) is that each agent submits at each step a collective action sequence for all their units.

## Installation

To use the `LuxAIMARLEnv`, you need to first clone [Lux-S3-Design](https://github.com/Lux-AI-Challenge/Lux-Design-S3/)

```
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3 
git clone https://github.com/ivarg/lux-marl
cd lux-marl
uv venv -p 3.11
uv sync
source .venv/bin/activate
```

Then clone the [marl-book](https://github.com/ivarg/marl-book/tree/uv) repo

