if __name__ == "__main__":  # pragma: no cover
    from trl import TrlParser

    from linalg_zero.config.configs import DatasetGenerationConfig

    trl = TrlParser(DatasetGenerationConfig)
    config: DatasetGenerationConfig = trl.parse_args_and_config()[0]
    print(config)
