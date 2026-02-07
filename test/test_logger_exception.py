from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.exception import DocumentPortalException

log.info("Logger working")

try:
    raise DocumentPortalException("Test exception", None)
except Exception as e:
    log.error("Exception caught", error=str(e))
