#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import os
from pathlib import Path


class DeepStrings:
    def __init__(self, search_root=None):
        """
        init initialize the object searching for root base of dataset, then appending
        the paths to lamps and carpets
        """
        if not search_root:
            search_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        found = False
        for root, subdirs, files in os.walk(search_root):
            if found:
                break
            if not found:
                for d in subdirs:
                    if d == "FINALDS":
                        rt = os.path.abspath(os.path.join(root, d))
                        found = True

        self.lamp_paths = [
            rt + "/lamps/chinese/120/RGB",
            rt + "/lamps/french/120/RGB",
            rt + "/lamps/turkish/120/RGB",
        ]

        self.carpet_paths_str = [
            rt + "/carpets_stretched/indian/200/RGB",
            rt + "/carpets_stretched/japanese/200/RGB",
            rt + "/carpets_stretched/scandinavian/200/RGB",
        ]

        self.carpet_paths_bla = [
            rt + "/carpets_blanked/indian/200/RGB",
            rt + "/carpets_blanked/japanese/200/RGB",
            rt + "/carpets_blanked/scandinavian/200/RGB",
        ]
