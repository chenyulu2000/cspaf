import logging
import coloredlogs


class AnaLogger:
    def __init__(self, exp_saved_path, logger_name='main_logger', log_level='DEBUG'):
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        console_fmt = '[%(asctime)s %(levelname)s %(filename)s \033[33mline\033[0m %(lineno)d %(process)d]\n%(message)s'
        file_fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]\n%(message)s'
        level = logging.INFO
        if log_level == 'DEBUG':
            level = logging.DEBUG
        elif log_level == 'ERROR':
            level = logging.ERROR
        coloredlogs.install(level=level)
        self.logger.propagate = False

        fileFormatter = logging.Formatter(file_fmt)
        coloredFormatter = coloredlogs.ColoredFormatter(
            fmt=console_fmt,
            level_styles=dict(
                debug=dict(color='white'),
                info=dict(color='green', bright=True),
                error=dict(color='red', bold=True, bright=True),
            ),
            field_styles=dict(
                name=dict(color='white'),
                asctime=dict(color='white'),
                filename=dict(color='yellow'),
                funcName=dict(color='yellow'),
                lineno=dict(color='yellow'),
            )
        )

        file_handler = logging.FileHandler(filename='{}/exp_log.log'.format(
            exp_saved_path) if exp_saved_path != '' else 'exp_log.log')
        file_handler.setFormatter(fileFormatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt=coloredFormatter)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)
