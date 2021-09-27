#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Status report on a PKDGRAV3 output directory.
    Author: Lorne Whiteway.
"""

import utility 
import sys
import traceback

if __name__ == '__main__':

    try:
    
        if len(sys.argv) != 2:
            raise SystemError("Provide directory name as command line argument.")
        utility.status(sys.argv[1])
        
        
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)
        




    
    
