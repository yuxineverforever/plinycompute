#!/usr/bin/env bash
#  Copyright 2018 Rice University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ========================================================================    

usage() {
    echo ""
    echo -e "\033[33;31m""    "Warning: This script deletes stored data. Deleted data cannot be"
             "restored, use it carefully!"\e[0m"

    cat <<EOM

    Description: This script deletes all PlinyCompute storage, catalog metadata,
    and kills both pdb-manager and pdb-worker processes in the machine where it
    is executed.

    Usage: scripts/$(basename $0) <param1>

           param1: <force>
                      This argument is optional, if provided it doesn't prompt user
                      for confirmation when cleaning up stored data.

EOM
   exit -1;
}

[[ "$@" = *--help ]] && { usage; } || [[ "$@" = *-h ]] && { usage; } || [[ ! "$1" = force ]] && { usage; }

# remove shared libraries from the tmp folder only if they exist
if [[ -n $(find /var/tmp/ -name "*.so" 2>/dev/null) ]]; then
    rm -rf /var/tmp/*.so
fi

# remove the any directory starting with pdbRoot only if they exist
if [[ -n $(find ./ -name "pdbRoot*" 2>/dev/null) ]]; then
    rm -rf pdbRoot*
fi

# remove the directory pdbRoot only if they exist
if [[ -e pdbRoot ]]; then
    rm -rf pdbRoot
fi

# remove any directory from /mnt that start with pdbRoot only if they exist
if [[ -n $(find /mnt/pdbRoot -name "*" 2>/dev/null) ]]; then
    rm -rf /mnt/pdbRoot*
fi

# remove the any directory starting with pdbWorker only if they exist
if [[ -n $(find ./ -name "pdbWorker*" 2>/dev/null) ]]; then
    rm -rf pdbWorker*
fi

# remove the directory pdbWorker only if they exist
if [[ -e pdbWorker ]]; then
    rm -rf pdbWorker
fi

# remove any directory from /mnt that start with pdbWorker only if they exist
if [[ -n $(find /mnt/pdbWorker -name "*" 2>/dev/null) ]]; then
    rm -rf /mnt/pdbWorker*
fi

# remove the /tmp/CatalogDir only if they exist
if [[ -e /tmp/CatalogDir ]]; then
    rm -rf /tmp/CatalogDir
fi


# remove the content from CatalogDir only if they exist
if [[ -n $(find ./CatalogDir -name "*" 2>/dev/null) ]]; then
    rm -rf CatalogDir/*
fi

# remove everything from CatalogDir only if they exist
if [[ -n $(find ./CatalogDir -name "*" 2>/dev/null) ]]; then
    rm -rf CatalogDir*
fi

# remove everything from /tmp/CatalogDir only if they exist
if [[ -n $(find /tmp/CatalogDir -name "*" 2>/dev/null) ]]; then
    rm -rf /tmp/CatalogDir*
fi

# remove everything from CatalogDir* only if they exist
if [[ -n $(find CatalogDir* -name "*" 2>/dev/null) ]]; then
    rm -rf CatalogDir*
fi

# remove anything from logs only if they exist
if [[ -n $(find ./logs -name "*" 2>/dev/null) ]]; then
    rm -rf logs/*
fi

# remove anything from statDB only if they exist
if [[ -n $(find ./statDB -name "*" 2>/dev/null) ]]; then
    rm -rf statDB*
fi

# kill all the processes
pkill -9 pdb-worker || true
pkill -9 pdb-manager || true
pkill -9 pdb-node || true

echo -e "All stored data in this node were deleted, and PlinyCompute processes were killed!."
