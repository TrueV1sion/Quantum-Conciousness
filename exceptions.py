# exceptions.py

class SystemProcessingError(Exception):
    """Exception raised for errors in the system processing."""
    pass

class WaveletProcessingError(Exception):
    """Exception raised for errors in wavelet processing."""
    pass

class BridgeEstablishmentError(Exception):
    """Exception raised when bridge establishment fails."""
    pass

class InformationTransferError(Exception):
    """Exception raised when information transfer fails."""
    pass

class PathwayProcessingError(Exception):
    """Exception raised for errors in processing pathways."""
    pass
