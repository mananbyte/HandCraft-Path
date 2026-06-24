"""
Safe loading utilities for the PanNuke pipeline.
Consolidates NumPy 2.x saved array load capabilities.
"""

import os
import re
import ast
import numpy as np

def safe_load_npy(filepath, mode='r'):
    """
    Safely load a .npy file by on-the-fly repairing of NumPy 2.x headers.
    Handles shape size serialization differences (e.g. np.int64) that cause
    standard ast.literal_eval to crash.
    """
    with open(filepath, 'rb') as f:
        magic = f.read(6)
        if magic != b'\x93NUMPY':
            raise ValueError("Not a numpy file")
        major = f.read(1)[0]
        minor = f.read(1)[0]
        if major == 1:
            header_len = int.from_bytes(f.read(2), byteorder='little')
            header_start = 10
        elif major == 2:
            header_len = int.from_bytes(f.read(4), byteorder='little')
            header_start = 12
        else:
            raise ValueError(f"Unsupported numpy version {major}.{minor}")
        header_bytes = f.read(header_len)
        header_str = header_bytes.decode('ascii').strip()
        
        # Repair the header string (handle np.int64 serialization issues in NumPy 2.x)
        header_str = re.sub(r'(?:numpy\.|np\.)?int(?:64|32)?\(([0-9]+)\)', r'\1', header_str)
        header_str = re.sub(r'(?:numpy\.)?dtype\([\'"]([^\'"]+)[\'"]\)', r"'\1'", header_str)
        
        header_dict = ast.literal_eval(header_str)
        shape = header_dict['shape']
        fortran_order = header_dict['fortran_order']
        dtype = np.dtype(header_dict['descr'])
        
    return np.memmap(filepath, dtype=dtype, mode=mode, offset=header_start + header_len, shape=shape, order='F' if fortran_order else 'C')
