#!/usr/bin/env python3

import os
import re
import sys
import hashlib
from os import listdir
from os.path import isfile, join, abspath

SRC_ROOT = os.path.join(os.path.abspath("."), "src")

# the headers for the built in objects
object_headers = os.path.join(SRC_ROOT, 'builtInPDBObjects', 'headers')

# the computations
computation_headers = os.path.join(SRC_ROOT, 'computations', 'headers')

# physical algorithms
physical_headers = os.path.join(SRC_ROOT, 'executionServer', 'headers', 'physicalAlgorithms')

# this is where we want to dump the generated files
objectTargetDir = os.path.join(SRC_ROOT, 'objectModel', 'headers')

# grab all the headers from the built in files
only_files = [abspath(join(object_headers, f))
              for f in listdir(object_headers) if isfile(join(object_headers, f)) and f[-2:] == '.h']

# grab all the headers from the computations
only_files.extend([abspath(join(computation_headers, f))
                   for f in listdir(computation_headers) if isfile(join(computation_headers, f)) and f[-2:] == '.h'])

# grab the physical algorithms
only_files.extend([abspath(join(physical_headers, f))
                   for f in listdir(physical_headers) if isfile(join(physical_headers, f)) and f[-2:] == '.h'])

def check_if_changed(includes, classes):

    # if the size of the arguments is not three we are not doing anything
    if len(sys.argv) != 3:
        return False

    # hash file name
    hash_file = '%s/.__type_codes_hash_%s' % (sys.argv[2], sys.argv[1])

    # check if hash file exits
    if os.path.isfile(hash_file):

        # if so open it
        with open(hash_file, "rb") as file:
            hash = file.read()
    else:
        hash = ""

    # get the hash
    m = get_hash(classes, includes)

    # return true if they are equal
    return hash != m


def write_out_hash(includes, classes):

    # if the size of the arguments is not three we are not doing anything
    if len(sys.argv) != 3:
        return

    # hash file name
    hash_file = '%s/.__type_codes_hash_%s' % (sys.argv[2], sys.argv[1])

    # get the hash
    m = get_hash(classes, includes)

    with open(hash_file, 'wb') as file:
        file.write(m)


def get_hash(classes, includes):

    # initialize tha hash object
    m = hashlib.sha256()

    # go through all the classes
    for c in classes:
        m.update(c.encode('utf-8'))

    # go through all the includes
    for i in includes:
        m.update(i.encode('utf-8'))

    # return the hash
    return m.digest()


def extract_code(sources):
    # Sort the class names so that we have consistent type ids
    scanned = list(scan_class_names(sources))
    scanned.sort()

    # check if there has been a change
    if check_if_changed(includes=sources, classes=scanned):
        write_files(includes=sources, classes=scanned)
        write_out_hash(includes=sources, classes=scanned)


def scan_class_names(includes):
    for counter, fileName in enumerate(includes, 1):
        datafile = open(fileName)
        # search for a line like:
        # // PRELOAD %ObjectTwo%
        p = re.compile('//\s*PRELOAD\s*%[\w\s\<\>]*%')
        for line in datafile:
            # if we found the line
            if p.search(line):
                # extract the text between the two '%' symbols
                m = p.search(line)
                instance = m.group()
                p = re.compile('%[\w\s<>]*%')
                m = p.search(instance)
                yield (m.group ())[1:-1]


def write_files(includes, classes):
    if len(sys.argv) == 3 and sys.argv[1] == "BuiltinPDBObjects":
        write_include_file(includes)
        write_code_file(classes)
    elif len(sys.argv) == 3 and sys.argv[1] == "BuiltInObjectTypeIDs":
        write_type_codes_file(classes)
    else:
        write_include_file(includes)
        write_code_file(classes)
        write_type_codes_file(classes)


def path_split(p, rest=None):
    if rest is None:
        rest = []
    (h,t) = os.path.split(p)
    if len(h) < 1: return [t]+rest
    if len(t) < 1: return [h]+rest
    return path_split(h, [t] + rest)


def common_path(l1, l2, common=None):
    if common is None:
        common = []

    if len(l1) < 1: return common, l1, l2
    if len(l2) < 1: return common, l1, l2
    if l1[0] != l2[0]: return common, l1, l2
    return common_path(l1[1:], l2[1:], common + [l1[0]])


def rel_path(p1, p2):
    (common,l1,l2) = common_path(path_split(p1), path_split(p2))
    p = []
    if len(l1) > 0:
        p = [ '../' * len(l1) ]
    p = p + l2
    return os.path.join( *p )


def write_include_file(includes):
    #
    # print objectTargetDir + 'BuiltinPDBObjects.h'
    #
    # this is the file where the produced list of includes goes
    include_file = open(os.path.join(objectTargetDir, 'BuiltinPDBObjects.h'), 'w+')
    include_file.write("// Auto-generated by code in SConstruct\n")
    for fileName in includes:
        include_file.write('#include "' + rel_path(objectTargetDir, fileName) + '"\n')


def write_code_file(classes):
    #
    # this is the file where the produced code goes
    code_file = open(os.path.join(objectTargetDir,'BuiltinPDBObjects.cc'), 'w+')
    code_file.write("// Auto-generated by code in SConstruct\n\n")
    code_file.write("// first, record all of the type codes\n")

    for counter, class_name in enumerate(classes, 1):
        code_file.write('objectTypeNamesList [getTypeName <' + class_name + '> ()] = ' + str(2 + counter) + ';\n')

    code_file.write('\n// now, record all of the vTables\n')

    for counter, class_name in enumerate(classes, 1):
        code_file.write('{\n\tconst UseTemporaryAllocationBlock tempBlock{1024 * 24};')
        code_file.write('\n\ttry {\n\t\t')
        code_file.write(class_name + ' tempObject;\n')
        code_file.write('\t\tallVTables [' + str(2 + counter) + '] = tempObject.getVTablePtr ();\n')
        code_file.write('\t} catch (NotEnoughSpace &e) {\n\t\t')
        code_file.write('std :: cout << "Not enough memory to allocate ' + class_name +
                        ' to extract the vTable.\\n";\n\t}\n}\n\n')


def write_type_codes_file(classes):
    #
    # this is the file where all of the built-in type codes goes
    type_codes_file = open(os.path.join(objectTargetDir, 'BuiltInObjectTypeIDs.h'), 'w+')
    type_codes_file.write("// Auto-generated by code in SConstruct\n")
    type_codes_file.write('#define NoMsg_TYPEID 0\n')

    # write out the String and Handle types, since they are not PDB Objects (an optimization)
    type_codes_file.write('#define String_TYPEID 1\n')
    type_codes_file.write('#define Handle_TYPEID 2\n')

    for counter, class_name in enumerate(classes, 1):
        pattern = re.compile('<[\w\s<>]*>')
        if pattern.search (class_name):
            template_arg = pattern.search(class_name)
            class_name = class_name.replace(template_arg.group(), "").strip()
        #
        # Remove the namespace if any
        class_name = class_name.rsplit("::")[-1]
        type_codes_file.write('#define ' + class_name + '_TYPEID ' + str(2 + counter) + '\n')


extract_code(only_files)
