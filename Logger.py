#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:32:54 2024

@author: mariaridel
"""

import os
import logging
import logging.handlers
import sys


# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        hashed_args = hash(args)
        if cls not in cls._instances:
            cls._instances[cls] = {}
        if hashed_args not in cls._instances[cls]:
            cls._instances[cls][hashed_args] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls][hashed_args]


class Logger(metaclass=Singleton):
    def __init__(self, logger_name):
        # Create a logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Create a handler for writing logs to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))

        # Create a handler for writing logs to a file
        file_handler = logging.handlers.RotatingFileHandler(
            filename=f'logs/{logger_name}.log',
            # backupCount=2,
            # maxBytes=10240 # 10KB
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))

        # Add both handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)