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


class ZeroLenY(MinIONBasecallerException):
    pass


class RefFileNotFound(MinIONBasecallerException):
    pass


class MissingMincallLogits(MinIONBasecallerException):
    pass


class MissingMincallAlignedRef(MinIONBasecallerException):
    pass
