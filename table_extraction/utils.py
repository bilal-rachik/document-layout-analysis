# -*- coding: utf-8 -*-
from __future__ import division

import re
import os
import sys
import random
import shutil
import string
import tempfile
import warnings
from itertools import groupby
from operator import itemgetter

import numpy as np



PY3 = sys.version_info[0] >= 3
if PY3:
    from urllib.request import urlopen
    from urllib.parse import urlparse as parse_url
    from urllib.parse import uses_relative, uses_netloc, uses_params
else:
    from urllib2 import urlopen
    from urlparse import urlparse as parse_url
    from urlparse import uses_relative, uses_netloc, uses_params


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")



def random_string(length):
    ret = ""
    while length:
        ret += random.choice(
            string.digits + string.ascii_lowercase + string.ascii_uppercase
        )
        length -= 1
    return ret


stream_kwargs = ["columns", "edge_tol", "row_tol", "column_tol"]
lattice_kwargs = [
    "process_background",
    "line_scale",
    "copy_text",
    "shift_text",
    "line_tol",
    "joint_tol",
    "threshold_blocksize",
    "threshold_constant",
    "iterations",
    "resolution",
]


def validate_input(kwargs, flavor="lattice"):
    def check_intersection(parser_kwargs, input_kwargs):
        isec = set(parser_kwargs).intersection(set(input_kwargs.keys()))
        if isec:
            raise ValueError(
                "{} cannot be used with flavor='{}'".format(
                    ",".join(sorted(isec)), flavor
                )
            )

    if flavor == "lattice":
        check_intersection(stream_kwargs, kwargs)
    else:
        check_intersection(lattice_kwargs, kwargs)


def remove_extra(kwargs, flavor="lattice"):
    if flavor == "lattice":
        for key in kwargs.keys():
            if key in stream_kwargs:
                kwargs.pop(key)
    else:
        for key in kwargs.keys():
            if key in lattice_kwargs:
                kwargs.pop(key)
    return kwargs


# https://stackoverflow.com/a/22726782
class TemporaryDirectory(object):
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)


def translate(x1, x2):
    """Translates x2 by x1.
    Parameters
    ----------
    x1 : float
    x2 : float
    Returns
    -------
    x2 : float
    """
    x2 += x1
    return x2


def scale(x, s):
    """Scales x by scaling factor s.
    Parameters
    ----------
    x : float
    s : float
    Returns
    -------
    x : float
    """
    x *= s
    return x


def translate_image(tables, v_segments, h_segments, img_y):
    """Translates  image coordinate space to image
    coordinate space.
    Parameters
    ----------
    tables : dict
        Dict with table boundaries as keys and list of intersections
        in that boundary as value.
    v_segments : list
        List of vertical line segments.
    h_segments : list
        List of horizontal line segments.
    img_y : float
        img_y is height of image.
    Returns
    -------
    tables_new : dict
    v_segments_new : dict
    h_segments_new : dict
    """
    tables_new = {}
    for k in tables.keys():
        x1, y1, x2, y2 = k
        y1 = abs(translate(-img_y, y1))
        y2 = abs(translate(-img_y, y2))
        j_x, j_y = zip(*tables[k])
        j_y = [abs(translate(-img_y, j)) for j in j_y]
        joints = list(zip(j_x, j_y))
        tables_new[(x1, y1, x2, y2)] = joints

    v_segments_new = []
    for v in v_segments:
        x1, x2 = v[0], v[2]
        y1, y2 = (
            abs(translate(-img_y, v[1])),
            abs(translate(-img_y, v[3])),
        )
        v_segments_new.append((x1, y1, x2, y2))

    h_segments_new = []
    for h in h_segments:
        x1, x2 = h[0], h[2]
        y1, y2 = (
            abs(translate(-img_y, h[1])),
            abs(translate(-img_y, h[3]))
        )
        h_segments_new.append((x1, y1, x2, y2))

    return tables_new, v_segments_new, h_segments_new


def get_rotation(chars, horizontal_text, vertical_text):
    """Detects if text in table is rotated or not using the current
    transformation matrix (CTM) and returns its orientation.
    Parameters
    ----------
    horizontal_text : list
        List of PDFMiner LTTextLineHorizontal objects.
    vertical_text : list
        List of PDFMiner LTTextLineVertical objects.
    ltchar : list
        List of PDFMiner LTChar objects.
    Returns
    -------
    rotation : string
        '' if text in table is upright, 'anticlockwise' if
        rotated 90 degree anticlockwise and 'clockwise' if
        rotated 90 degree clockwise.
    """
    rotation = ""
    hlen = len([t for t in horizontal_text if t.get_text().strip()])
    vlen = len([t for t in vertical_text if t.get_text().strip()])
    if hlen < vlen:
        clockwise = sum(t.matrix[1] < 0 and t.matrix[2] > 0 for t in chars)
        anticlockwise = sum(t.matrix[1] > 0 and t.matrix[2] < 0 for t in chars)
        rotation = "anticlockwise" if clockwise < anticlockwise else "clockwise"
    return rotation


def segments_in_bbox(bbox, v_segments, h_segments):
    """Returns all line segments present inside a bounding box.
    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in image coordinate
        space.
    v_segments : list
        List of vertical line segments.
    h_segments : list
        List of vertical horizontal segments.
    Returns
    -------
    v_s : list
        List of vertical line segments that lie inside table.
    h_s : list
        List of horizontal line segments that lie inside table.
    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    v_s = [
        v
        for v in v_segments
        if v[1] > lb[1] - 2 and v[3] < rt[1] + 2 and lb[0] - 2 <= v[0] <= rt[0] + 2
    ]
    h_s = [
        h
        for h in h_segments
        if h[0] > lb[0] - 2 and h[2] < rt[0] + 2 and lb[1] - 2 <= h[1] <= rt[1] + 2
    ]
    return v_s, h_s


def text_in_bbox(bbox, text):
    """Returns all text objects present inside a bounding box.
    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.
    text : List of PDFMiner text objects.
    Returns
    -------
    t_bbox : list
        List of tesseract text objects that lie inside table.
    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    t_bbox = [
        t
        for t in text
        if lb[0] - 2 <= (t.x0 + t.x1) / 2.0 <= rt[0] + 2
        and lb[1] - 2 <= (t.y0 + t.y1) / 2.0 <= rt[1] + 2
    ]
    return t_bbox


def merge_close_lines(ar, line_tol=2):
    """Merges lines which are within a tolerance by calculating a
    moving mean, based on their x or y axis projections.
    Parameters
    ----------
    ar : list
    line_tol : int, optional (default: 2)
    Returns
    -------
    ret : list
    """
    ret = []
    for a in ar:
        if not ret:
            ret.append(a)
        else:
            temp = ret[-1]
            if np.isclose(temp, a, atol=line_tol):
                temp = (temp + a) / 2.0
                ret[-1] = temp
            else:
                ret.append(a)
    return ret


class Text(object):
    def __init__(self, x1, y1, x2, y2, text):
        self.x0 = x1
        self.y0 = y1
        self.x1 = x2
        self.y1 = y2
        self._objs = text

    def __repr__(self):
        return "<Text x1={} y1={} x2={} y2={} text={}>".format(
            round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2), self._objs
        )

    def get_text(self):
        return self._objs


def get_text_box(tess_result, x1, y1, image_height):
    t_box = []
    for idx, text in enumerate(tess_result['text']):
        if len(text) > 0 and tess_result['conf'][idx] > 0:
            x, y, w, h = (
            tess_result['left'][idx], tess_result['top'][idx], tess_result['width'][idx], tess_result['height'][idx])
            x = x + x1
            y = y + y1
            t_box.append(Text(x, abs(translate(-image_height, y)), x + w, abs(translate(-image_height, y + h)), text))
    return t_box


def text_strip(text, strip=""):
    """Strips any characters in `strip` that are present in `text`.
    Parameters
    ----------
    text : str
        Text to process and strip.
    strip : str, optional (default: '')
        Characters that should be stripped from `text`.
    Returns
    -------
    stripped : str
    """
    if not strip:
        return text

    stripped = re.sub(
        r"[{}]".format("".join(map(re.escape, strip))), "", text, re.UNICODE
    )
    return stripped


def get_table_index(
    table, t, split_text=False, flag_size=False, strip_text=""
):
    """Gets indices of the table cell where given text object lies by
    comparing their y and x-coordinates.
    Parameters
    ----------
    table : camelot.core.Table
    t : object
        text Textobject object.
    split_text : bool, optional (default: False)
        Whether or not to split a text line if it spans across
        multiple cells.
    flag_size : bool, optional (default: False)
        Whether or not to highlight a substring using <s></s>
        if its size is different from rest of the string. (Useful for
        super and subscripts)
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    Returns
    -------
    indices : list
        List of tuples of the form (r_idx, c_idx, text) where r_idx
        and c_idx are row and column indices.
    error : float
        Assignment error, percentage of text area that lies outside
        a cell.
        +-------+
        |       |
        |   [Text bounding box]
        |       |
        +-------+
    """
    r_idx, c_idx = [-1] * 2
    for r in range(len(table.rows)):
        if (t.y0 + t.y1) / 2.0 < table.rows[r][0] and (t.y0 + t.y1) / 2.0 > table.rows[
            r
        ][1]:
            lt_col_overlap = []
            for c in table.cols:
                if c[0] <= t.x1 and c[1] >= t.x0:
                    left = t.x0 if c[0] <= t.x0 else c[0]
                    right = t.x1 if c[1] >= t.x1 else c[1]
                    lt_col_overlap.append(abs(left - right) / abs(c[0] - c[1]))
                else:
                    lt_col_overlap.append(-1)
            if len(list(filter(lambda x: x != -1, lt_col_overlap))) == 0:
                text = t.get_text().strip("\n")
                text_range = (t.x0, t.x1)
                col_range = (table.cols[0][0], table.cols[-1][1])
                warnings.warn(
                    "{} {} does not lie in column range {}".format(
                        text, text_range, col_range
                    )
                )
            r_idx = r
            c_idx = lt_col_overlap.index(max(lt_col_overlap))
            break

    # error calculation
    y0_offset, y1_offset, x0_offset, x1_offset = [0] * 4
    if t.y0 > table.rows[r_idx][0]:
        y0_offset = abs(t.y0 - table.rows[r_idx][0])
    if t.y1 < table.rows[r_idx][1]:
        y1_offset = abs(t.y1 - table.rows[r_idx][1])
    if t.x0 < table.cols[c_idx][0]:
        x0_offset = abs(t.x0 - table.cols[c_idx][0])
    if t.x1 > table.cols[c_idx][1]:
        x1_offset = abs(t.x1 - table.cols[c_idx][1])
    X = 1.0 if abs(t.x0 - t.x1) == 0.0 else abs(t.x0 - t.x1)
    Y = 1.0 if abs(t.y0 - t.y1) == 0.0 else abs(t.y0 - t.y1)
    charea = X * Y
    error = ((X * (y0_offset + y1_offset)) + (Y * (x0_offset + x1_offset))) / charea

    return [(r_idx, c_idx, text_strip(t.get_text(), strip_text))], error


def compute_accuracy(error_weights):
    """Calculates a score based on weights assigned to various
    parameters and their error percentages.
    Parameters
    ----------
    error_weights : list
        Two-dimensional list of the form [[p1, e1], [p2, e2], ...]
        where pn is the weight assigned to list of errors en.
        Sum of pn should be equal to 100.
    Returns
    -------
    score : float
    """
    SCORE_VAL = 100
    try:
        score = 0
        if sum([ew[0] for ew in error_weights]) != SCORE_VAL:
            raise ValueError("Sum of weights should be equal to 100.")
        for ew in error_weights:
            weight = ew[0] / len(ew[1])
            for error_percentage in ew[1]:
                score += weight * (1 - error_percentage)
    except ZeroDivisionError:
        score = 0
    return score


def compute_whitespace(d):
    """Calculates the percentage of empty strings in a
    two-dimensional list.
    Parameters
    ----------
    d : list
    Returns
    -------
    whitespace : float
        Percentage of empty cells.
    """
    whitespace = 0
    r_nempty_cells, c_nempty_cells = [], []
    for i in d:
        for j in i:
            if j.strip() == "":
                whitespace += 1
    whitespace = 100 * (whitespace / float(len(d) * len(d[0])))
    return whitespace


