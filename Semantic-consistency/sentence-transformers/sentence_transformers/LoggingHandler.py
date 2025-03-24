import logging
import tqdm
'''
主要是日志处理程序LoggingHandler和一个日志配置函数install_logger。共同作用于日志的配置和输出格式，特别适合在使用进度条的场景下进行日志记录。
'''
class LoggingHandler(logging.Handler):
    # LoggingHandler 是 logging.Handler 的一个子类，用于重定义日志的输出行为，使其与 tqdm 进度条兼容，避免进度条被日志输出打断。通常，当控制台同时显示进度条和日志信息时，日志的标准输出会影响进度条的显示，LoggingHandler 通过 tqdm.tqdm.write() 方法来写入日志，确保日志输出不会影响进度条的刷新。
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    # 重写 logging.Handler 的 emit() 方法。该方法在每次日志记录触发时调用。
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

# install_logger 函数用于配置给定的日志记录器，设置日志格式、级别、样式等。它使用 coloredlogs 库在控制台中为不同的日志级别设置颜色，增强日志的可读性。除此之外，它还添加了一个自定义的日志级别 NOTICE，位于 INFO 和 WARNING 之间，用于输出特殊的重要信息。
def install_logger(
    given_logger, level = logging.WARNING, fmt="%(levelname)s:%(name)s:%(message)s"
):
    """ Configures the given logger; format, logging level, style, etc """
    import coloredlogs

    def add_notice_log_level():
        """ Creates a new 'notice' logging level """
        # inspired by:
        # https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
        NOTICE_LEVEL_NUM = 25
        logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")

        def notice(self, message, *args, **kws):
            if self.isEnabledFor(NOTICE_LEVEL_NUM):
                self._log(NOTICE_LEVEL_NUM, message, args, **kws)

        logging.Logger.notice = notice

    # Add an extra logging level above INFO and below WARNING
    add_notice_log_level()

    # More style info at:
    # https://coloredlogs.readthedocs.io/en/latest/api.html
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {"color": "white", "faint": True}
    level_styles["notice"] = {"color": "cyan", "bold": True}

    coloredlogs.install(
        logger=given_logger,
        level=level,
        use_chroot=False,
        fmt=fmt,
        level_styles=level_styles,
        field_styles=field_styles,
    )
