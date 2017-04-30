class MinIONBasecallerException(Exception):
    pass


class TooLargeEditDistance(MinIONBasecallerException):
    pass


class MissingRNN1DBasecall(MinIONBasecallerException):
    pass


class BlockSizeYTooSmall(MinIONBasecallerException):
    pass


class InsufficientDataBlocks(MinIONBasecallerException):
    pass
