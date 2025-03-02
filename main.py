import hydra
import jax
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    env = hydra.utils.call(
        cfg.env,
        seed=cfg.seed,
    )

    print(f"env: {env}")

    if cfg.seed is not None:
        rng_key = jax.random.key(cfg.seed)

    else:
        logger.warning("No seed has been set.")


if __name__ == "__main__":
    main()
