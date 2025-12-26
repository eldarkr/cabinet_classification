from omegaconf import OmegaConf
from src.data.prepare import DataPreparationPipeline


def main():
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)
    
    # Initialize and run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
