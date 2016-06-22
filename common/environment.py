from __future__ import print_function
import os
import re


def file_parts(s):
    """
    Splits the input path into base path, base name, and extension.
    :param s: full path
    :return: 3-tuple of (base_path, base_name, extension)
    """
    base_path = os.path.dirname(s)
    name_and_extension = os.path.splitext(os.path.basename(s))
    base_name = name_and_extension[0]
    extension = name_and_extension[1]
    return base_path, base_name, extension


def file_extension(s):
    """
    Returns only the extension of the input path.
    :param s: input path
    :return: extension of the file in the path
    """
    return os.path.splitext(os.path.basename(s))[1]


def file_names(directory):
    """
    Returns the file names for the given directory.
    :param directory: directory name
    :return: list of file names in the directory
    """
    files = filter(lambda file: os.path.isfile('/'.join((directory, file))), os.listdir(directory))
    return tuple(files)


def files_with_extension(directory, extensions):
    """
    Returns the file names in the given directory which contain one of the desired extensions.
    :param directory: directory to search
    :param extensions: file extensions to filter on
    :return: files in the directory which contain one of the desired extensions
    """
    files = ['/'.join((directory, f)) for f in os.listdir(directory)
             if os.path.isfile('/'.join((directory, f))) and file_extension(f) in extensions]
    return tuple(files)


def file_names_append(directory, tok):
    """
    Appends the file names in the directory with the specified token.
    :param directory: directory to work in
    :param tok: token to append to each file name
    """
    for f in file_names(directory):
        old_name = '/'.join((directory, f))
        new_name = '/'.join((directory, tok + f))
        os.rename(old_name, new_name)


def file_names_replace(directory, tok, new_tok):
    """
    Performs a string replace with the specified tokens, for the files in the specified directory.
    :param directory: directory to work in
    :param tok: token to be replaced
    :param new_tok: new token
    """
    for f in file_names(directory):
        if f.find(tok) > -1:
            old_name = '/'.join((directory, f))
            new_filename = f.replace(tok, new_tok)
            new_name = '/'.join((directory, new_filename))
            os.rename(old_name, new_name)


def camel_to_snake(s):
    """
    Converts a camelCase string into a snake_case string.
    Adapter from the link below:
    http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
        Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    :param s: camelCase string to convert
    :return: snake_case format of the original string
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def print_lines(num_lines):
    print(('\n'.join([n for n in num_lines * ''])))
