from logging import Logger

from grpc import RpcError, StatusCode


def handle_tfs_error(re: RpcError, model_name: str, logger: Logger):
    rpc_error_code = re.code()
    if rpc_error_code == StatusCode.UNAVAILABLE:
        logger.error("Tensorflow serving service is not available.")
    elif rpc_error_code == StatusCode.NOT_FOUND:
        logger.error(f"Tensorflow serving model not found for {model_name}.")
    else:
        logger.error(f"Tensorflow serving error for {model_name}: StatusCode={rpc_error_code}.")
    logger.error(f"Tensorflow serving error. Model: {model_name}. Code:{rpc_error_code}. {re.details()}")
