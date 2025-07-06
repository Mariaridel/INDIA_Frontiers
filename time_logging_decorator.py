#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:52:39 2024

@author: mariaridel
"""

import time
from functools import wraps
from datetime import datetime
from Logger import Logger

# Initialize a global list to store logs
method_logs = []

logger_time = Logger('times_logger').logger

def log_execution_time(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time_epoch = time.time()
        start_time = datetime.fromtimestamp(start_time_epoch).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        
        result = None
        try:
            result = method(*args, **kwargs)
        finally:
            end_time_epoch = time.time()
            end_time = datetime.fromtimestamp(end_time_epoch).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            duration_seconds = end_time_epoch - start_time_epoch
            duration_minutes = duration_seconds / 60
            
            logger_time.info({
                "method": method.__name__,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_minutes
            })
            
            method_logs.append({
                "method": method.__name__,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_minutes
            })
        return result
    return wrapper

def get_method_logs():
    return method_logs

def clear_method_logs():
    global method_logs
    method_logs = []
