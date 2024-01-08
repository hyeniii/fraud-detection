import logging
import logging.config
import configparser

def load_logging_config(filename, log_file):
    config = configparser.ConfigParser()
    config.read(filename)

    # Modify the filename for the fileHandler
    config['handler_fileHandler']['args'] = f"('{log_file}', 'a')"

    # Apply the modified logging configuration
    logging.config.fileConfig(config)