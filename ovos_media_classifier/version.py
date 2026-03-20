VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_BUILD = 1
VERSION_ALPHA = 1

# START_VERSION_BLOCK
def _get_version():
    return ".".join([str(VERSION_MAJOR), str(VERSION_MINOR), str(VERSION_BUILD)])

__version__ = _get_version() + (f"a{VERSION_ALPHA}" if VERSION_ALPHA else "")
# END_VERSION_BLOCK
