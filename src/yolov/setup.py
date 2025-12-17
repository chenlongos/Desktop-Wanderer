_is_acl_initialized = False

def is_setup():
    global _is_acl_initialized
    _is_acl_initialized = True

def get_is_acl_initialized():
    return _is_acl_initialized