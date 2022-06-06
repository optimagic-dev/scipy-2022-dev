from distutils.spawn import find_executable


def is_installed(executable):
    return bool(find_executable(executable))
